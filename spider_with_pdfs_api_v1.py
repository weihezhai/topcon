import openreview
import json
import time
import csv
from datetime import datetime
import os
import requests
import argparse
from typing import Dict, List, Optional
import urllib.parse
from pathlib import Path
from tqdm import tqdm

class OpenReviewCrawlerWithPDFs:
    def __init__(self):
        # Initialize the OpenReview client with authentication
        self.client = openreview.Client(
            baseurl='https://api.openreview.net',
            username='weihezhai@gmail.com',
            password='aMmeWArff6a3Z2k'
        )
        self.paper_to_file_map = {}  # Maps paper_id to json filename
        self.chunk_size = 500  # Papers per JSON file
        self.review_stats = {'api_v1_extracted': 0, 'fallback_extracted': 0, 'total_papers': 0}
        
    def extract_reviews_from_directreplies(self, paper) -> List[Dict]:
        """Extract reviews from directReplies (API V1 method)"""
        reviews = []
        try:
            if hasattr(paper, 'details') and paper.details and 'directReplies' in paper.details:
                direct_replies = paper.details['directReplies']
                for reply in direct_replies:
                    # Check if this reply is an official review
                    invitation = reply.get('invitation', '')
                    if invitation.endswith('Official_Review') or 'Review' in invitation:
                        # Convert to Note object if it's not already
                        if isinstance(reply, dict):
                            review_note = openreview.Note.from_json(reply)
                        else:
                            review_note = reply
                        reviews.append(review_note)
                        
                if reviews:
                    self.review_stats['api_v1_extracted'] += len(reviews)
                    return reviews
        except Exception as e:
            print(f"Error extracting reviews from directReplies for {getattr(paper, 'id', 'unknown')}: {e}")
        
        return []
    
    def get_conference_papers(self, limit: int = None, venue_year: str = '2024', conference: str = 'neurips') -> List[Dict]:
        """Fetch all papers for specified conference and year"""
        conf_name = conference.upper()
        print(f"Searching for {conf_name} {venue_year} submissions...")
        
        # Define invitation formats for different conferences
        conference_formats = {
            'neurips': {
                '2024': [
                    'NeurIPS.cc/2024/Conference/-/Submission',
                    'NeurIPS.cc/2024/Conference/Submission',
                    'neurips.cc/2024/Conference/-/Submission',
                ],
                'other': [
                    f'NeurIPS.cc/{venue_year}/Conference/-/Submission',
                    f'neurips.cc/{venue_year}/Conference/-/Submission',
                ]
            },
            'iclr': {
                '2024': [
                    'ICLR.cc/2024/Conference/-/Submission',
                    'iclr.cc/2024/Conference/-/Submission',
                ],
                'other': [
                    f'ICLR.cc/{venue_year}/Conference/-/Blind_Submission',  # ICLR 2023 uses Blind_Submission
                    f'ICLR.cc/{venue_year}/Conference/-/Submission',
                    f'iclr.cc/{venue_year}/Conference/-/Submission',
                ]
            },
            'icml': {
                '2024': [
                    'ICML.cc/2024/Conference/-/Submission',
                    'icml.cc/2024/Conference/-/Submission',
                ],
                'other': [
                    f'ICML.cc/{venue_year}/Conference/-/Submission',
                    f'icml.cc/{venue_year}/Conference/-/Submission',
                ]
            },
            'emnlp': {
                '2024': [
                    'EMNLP/2024/Conference/-/Submission',
                    'aclweb.org/EMNLP/2024/Conference/-/Submission',
                ],
                'other': [
                    f'EMNLP/{venue_year}/Conference/-/Submission',
                    f'aclweb.org/EMNLP/{venue_year}/Conference/-/Submission',
                ]
            }
        }
        
        # Get invitation formats for the specified conference
        conf_formats = conference_formats.get(conference.lower(), conference_formats['neurips'])
        if venue_year == '2024':
            possible_invitations = conf_formats['2024']
        else:
            possible_invitations = conf_formats['other']
        
        papers = []
        
        for invitation in possible_invitations:
            try:
                print(f"Trying invitation: {invitation}")
                
                # Use get_all_notes() with directReplies to get reviews automatically
                print(f"Fetching all notes for {invitation} with reviews (using directReplies)...")
                notes = list(self.client.get_all_notes(
                    invitation=invitation,
                    details='directReplies'
                ))
                
                # Apply user-specified limit if provided
                if limit and len(notes) > limit:
                    notes = notes[:limit]
                    print(f"Limited to {limit} papers as requested")
                
                if notes:
                    print(f"Found {len(notes)} papers with invitation: {invitation}")
                    papers = notes
                    break
                else:
                    print(f"No papers found for invitation: {invitation}")
                    
            except Exception as e:
                print(f"Error with invitation {invitation}: {e}")
                continue
    
        # If no papers found for requested year and conference is neurips, try fallback
        if not papers and conference.lower() == 'neurips' and venue_year == '2024':
            print("NeurIPS 2024 not found, trying NeurIPS 2023...")
            try:
                notes = list(self.client.get_all_notes(
                    invitation='NeurIPS.cc/2023/Conference/-/Submission',
                    details='directReplies'
                ))
                
                if limit and len(notes) > limit:
                    notes = notes[:limit]
                    
                papers = notes
                print(f"Found {len(papers)} NeurIPS 2023 papers as fallback")
            except Exception as e:
                print(f"Error fetching NeurIPS 2023: {e}")
    
        return papers
    
    def get_paper_reviews(self, paper_id: str, paper=None) -> List[Dict]:
        """Fetch all reviews for a specific paper using API V1 method first, then fallback"""
        # Try API V1 method first if we have the paper object with directReplies
        if paper:
            api_v1_reviews = self.extract_reviews_from_directreplies(paper)
            if api_v1_reviews:
                return api_v1_reviews
        
        # Fallback to original method
        try:
            # Get all notes in the forum (paper discussion)
            notes = list(self.client.get_notes(forum=paper_id))
            
            # Filter for reviews only - check if invitation attribute exists
            reviews = []
            for note in notes:
                invitation = getattr(note, 'invitation', '') or getattr(note, 'invitations', [''])[0] if hasattr(note, 'invitations') else ''
                if ('Review' in invitation or invitation.endswith('Official_Review')) and note.id != paper_id:
                    reviews.append(note)
            
            if reviews:
                self.review_stats['fallback_extracted'] += len(reviews)
            
            time.sleep(0.1)  # Small delay to be respectful
            return reviews
            
        except Exception as e:
            print(f"Error fetching reviews for {paper_id}: {e}")
            return []
    
    def download_pdf_and_supplements(self, paper, pdf_dir: str) -> Dict[str, bool]:
        """Download PDF and supplementary materials for a given paper"""
        try:
            paper_id = paper.id
            paper_content = getattr(paper, 'content', {})
            results = {'pdf': False, 'supplement': False}
            
            # Create supplementary directory
            supplement_dir = os.path.join(os.path.dirname(pdf_dir), "supplementary")
            os.makedirs(supplement_dir, exist_ok=True)
            
            # Download main PDF
            pdf_url = None
            if 'pdf' in paper_content:
                pdf_field = paper_content['pdf']
                if isinstance(pdf_field, dict) and 'value' in pdf_field:
                    pdf_url = pdf_field['value']
                elif isinstance(pdf_field, str):
                    pdf_url = pdf_field
            
            if pdf_url:
                # If PDF URL is relative, make it absolute
                if pdf_url.startswith('/'):
                    pdf_url = 'https://openreview.net' + pdf_url
                elif not pdf_url.startswith('http'):
                    pdf_url = 'https://openreview.net/' + pdf_url
                
                # Create filename using paper ID
                pdf_filename = f"{paper_id}.pdf"
                pdf_path = os.path.join(pdf_dir, pdf_filename)
                
                # Skip if already downloaded
                if not os.path.exists(pdf_path):
                    try:
                        # Use OpenReview client's attachment method
                        pdf_data = self.client.get_attachment(field_name='pdf', id=paper_id)
                        with open(pdf_path, 'wb') as f:
                            f.write(pdf_data)
                        results['pdf'] = True
                    except Exception as e:
                        # Fallback to URL download if attachment method fails
                        session = requests.Session()
                        session.headers.update({
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        })
                        
                        response = session.get(pdf_url, timeout=30, stream=True)
                        response.raise_for_status()
                        
                        with open(pdf_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        results['pdf'] = True
                else:
                    results['pdf'] = True
            
            # Download supplementary materials (any format, save as .zip)
            supplement_fields = ['supplementary_material', 'supplement', 'supplemental_material', 'code']
            max_size_bytes = 20 * 1024 * 1024  # 20MB limit
            
            for field_name in supplement_fields:
                if field_name in paper_content:
                    supplement_field = paper_content[field_name]
                    if isinstance(supplement_field, dict) and 'value' in supplement_field:
                        try:
                            # Use paper ID as filename with .zip extension
                            supplement_filename = f"{paper_id}.zip"
                            supplement_path = os.path.join(supplement_dir, supplement_filename)
                            
                            if not os.path.exists(supplement_path):
                                # Get attachment data to check size
                                supplement_data = self.client.get_attachment(field_name=field_name, id=paper_id)
                                
                                # Check file size
                                if len(supplement_data) > max_size_bytes:
                                    print(f"Skipping {field_name} for {paper_id}: size {len(supplement_data)/1024/1024:.1f}MB exceeds 20MB limit")
                                    continue
                                
                                # Save the supplementary material as .zip regardless of original format
                                with open(supplement_path, 'wb') as f:
                                    f.write(supplement_data)
                                results['supplement'] = True
                                print(f"Downloaded {field_name} for {paper_id} ({len(supplement_data)/1024/1024:.1f}MB) -> {supplement_filename}")
                                break  # Only download first available supplement
                            else:
                                results['supplement'] = True
                                break  # File already exists
                                
                        except Exception as e:
                            print(f"Failed to download {field_name} for {paper_id}: {e}")
                            continue
            
            time.sleep(0.5)  # Be respectful with download rate
            return results
            
        except Exception as e:
            print(f"Error downloading files for {paper_id}: {e}")
            return {'pdf': False, 'supplement': False}
    
    def search_papers_by_venue(self, venue: str, limit: int = None) -> List[Dict]:
        """Search papers by venue name"""
        try:
            print(f"Searching for papers with venue: {venue}")
            notes = list(self.client.get_notes(
                content={'venue': venue},
                limit=limit
            ))
            print(f"Found {len(notes)} papers for venue: {venue}")
            return notes
        except Exception as e:
            print(f"Error searching by venue {venue}: {e}")
            return []
    
    def save_data_chunked(self, all_data: List[Dict], output_dir: str, conference: str = 'neurips', venue_year: str = '2024'):
        """Save data in chunks and maintain a mapping file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create chunks of data
        chunks = [all_data[i:i + self.chunk_size] for i in range(0, len(all_data), self.chunk_size)]
        
        print(f"💾 Saving {len(all_data)} papers in {len(chunks)} chunks...")
        
        # Save each chunk and update mapping
        chunk_pbar = tqdm(chunks, desc="Saving chunks", unit="chunk")
        for chunk_idx, chunk in enumerate(chunk_pbar):
            chunk_filename = f"{conference}_{venue_year}_chunk_{chunk_idx:03d}_{timestamp}.json"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            # Save chunk
            with open(chunk_path, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, indent=2, ensure_ascii=False)
            
            # Update mapping for each paper in chunk
            for item in chunk:
                paper_id = item['paper'].get('id', '') if isinstance(item['paper'], dict) else getattr(item['paper'], 'id', '')
                if paper_id:
                    self.paper_to_file_map[paper_id] = chunk_filename
            
            chunk_pbar.set_postfix({"Papers": len(chunk), "File": chunk_filename})
        
        chunk_pbar.close()
        
        # Save mapping file
        mapping_file = os.path.join(output_dir, f"{conference}_{venue_year}_paper_mapping_{timestamp}.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.paper_to_file_map, f, indent=2, ensure_ascii=False)
        
        # Save CSV summary (all data combined)
        csv_file = os.path.join(output_dir, f"{conference}_{venue_year}_summary_{timestamp}.csv")
        self.save_csv_summary(all_data, csv_file)
        
        print(f"💾 Data saved in {len(chunks)} chunks to {output_dir}/")
        print(f"🗂️  Paper mapping saved to {mapping_file}")
        print(f"📊 CSV summary saved to {csv_file}")
    
    def crawl_all_data_with_pdfs(self, output_dir: str = "conference_data", limit: int = None, venue_year: str = '2024', conference: str = 'neurips'):
        """Crawl all papers, their reviews, and download PDFs and supplements"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create PDF and supplementary directories
        pdf_dir = os.path.join(output_dir, "pdfs")
        supplement_dir = os.path.join(output_dir, "supplementary")
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(supplement_dir, exist_ok=True)
        
        print("🔍 Fetching papers...")
        papers = self.get_conference_papers(limit, venue_year, conference)
        
        # If still no papers, try searching by venue
        if not papers:
            print("🔍 Trying venue-based search...")
            venue_name = f"{conference.upper()} {venue_year}"
            papers = self.search_papers_by_venue(venue_name, limit)
            
        if not papers and conference.lower() == 'neurips' and venue_year == '2024':
            papers = self.search_papers_by_venue("NeurIPS 2023", limit)
        
        print(f"📊 Found {len(papers)} papers")
        
        if not papers:
            print("❌ No papers found. Exiting...")
            return []
        
        all_data = []
        successful_pdf_downloads = 0
        failed_pdf_downloads = 0
        successful_supplement_downloads = 0
        skipped_supplements = 0
        self.review_stats['total_papers'] = len(papers)
        
        # Create progress bars
        main_pbar = tqdm(papers, desc="🚀 Processing papers", unit="paper")
        
        for i, paper in enumerate(main_pbar):
            paper_id = paper.id
            main_pbar.set_description(f"🚀 Processing {paper_id}")
            
            # Get reviews using API V1 method first
            reviews = self.get_paper_reviews(paper_id, paper)
            
            # Download PDF and supplements
            download_results = self.download_pdf_and_supplements(paper, pdf_dir)
            if download_results['pdf']:
                successful_pdf_downloads += 1
            else:
                failed_pdf_downloads += 1
            
            if download_results['supplement']:
                successful_supplement_downloads += 1
            
            paper_data = {
                'paper': paper.to_json() if hasattr(paper, 'to_json') else paper.__dict__,
                'reviews': [review.to_json() if hasattr(review, 'to_json') else review.__dict__ for review in reviews],
                'pdf_downloaded': download_results['pdf'],
                'supplement_downloaded': download_results['supplement'],
                'crawled_at': datetime.now().isoformat()
            }
            
            all_data.append(paper_data)
            
            # Update progress bar with stats
            main_pbar.set_postfix({
                "PDFs": f"✅{successful_pdf_downloads} ❌{failed_pdf_downloads}",
                "Suppl": f"✅{successful_supplement_downloads}",
                "Reviews": len(reviews)
            })
            
            # Save chunk when we reach chunk_size
            if len(all_data) % self.chunk_size == 0:
                self.save_single_chunk(all_data[-self.chunk_size:], output_dir, conference, venue_year, len(all_data) // self.chunk_size - 1)
                tqdm.write(f"💾 Saved chunk: {len(all_data) // self.chunk_size} ({self.chunk_size} papers)")
        
        # Save remaining data as final chunk
        remaining_papers = len(all_data) % self.chunk_size
        if remaining_papers > 0:
            final_chunk_idx = len(all_data) // self.chunk_size
            self.save_single_chunk(all_data[-remaining_papers:], output_dir, conference, venue_year, final_chunk_idx)
        
        # Save mapping and CSV summary at the end
        self.save_final_metadata(all_data, output_dir, conference, venue_year)
        
        print(f"\n✅ Crawl completed!")
        print(f"📊 Total papers processed: {len(all_data)}")
        print(f"📥 PDFs successfully downloaded: {successful_pdf_downloads}")
        print(f"📎 Supplements successfully downloaded: {successful_supplement_downloads}")
        print(f"❌ PDF download failures: {failed_pdf_downloads}")
        print(f"📈 PDF success rate: {successful_pdf_downloads/len(all_data)*100:.1f}%")
        print(f"📈 Supplement success rate: {successful_supplement_downloads/len(all_data)*100:.1f}%")
        print(f"📊 Reviews extracted via API V1: {self.review_stats['api_v1_extracted']}")
        print(f"📊 Reviews extracted via fallback: {self.review_stats['fallback_extracted']}")
        
        return all_data
    
    def crawl_metadata_only(self, output_dir: str = "conference_data", limit: int = None, venue_year: str = '2024', conference: str = 'neurips'):
        """Crawl all papers and their reviews without downloading PDFs"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("🔍 Fetching papers...")
        papers = self.get_conference_papers(limit, venue_year, conference)
        
        # If still no papers, try searching by venue
        if not papers:
            print("🔍 Trying venue-based search...")
            venue_name = f"{conference.upper()} {venue_year}"
            papers = self.search_papers_by_venue(venue_name, limit)
            
        if not papers and conference.lower() == 'neurips' and venue_year == '2024':
            papers = self.search_papers_by_venue("NeurIPS 2023", limit)
        
        print(f"📊 Found {len(papers)} papers")
        
        if not papers:
            print("❌ No papers found. Exiting...")
            return []
        
        all_data = []
        self.review_stats['total_papers'] = len(papers)
        
        # Create progress bar
        main_pbar = tqdm(papers, desc="🚀 Processing papers", unit="paper")
        
        for i, paper in enumerate(main_pbar):
            paper_id = paper.id
            main_pbar.set_description(f"🚀 Processing {paper_id}")
            
            # Get reviews using API V1 method first
            reviews = self.get_paper_reviews(paper_id, paper)
            
            paper_data = {
                'paper': paper.to_json() if hasattr(paper, 'to_json') else paper.__dict__,
                'reviews': [review.to_json() if hasattr(review, 'to_json') else review.__dict__ for review in reviews],
                'pdf_downloaded': False,
                'crawled_at': datetime.now().isoformat()
            }
            
            all_data.append(paper_data)
            
            # Update progress bar with stats
            main_pbar.set_postfix({"Reviews": len(reviews)})
            
            # Save incrementally every 10 papers
            if (i + 1) % 10 == 0:
                self.save_data_chunked(all_data, output_dir, conference, venue_year)
                tqdm.write(f"💾 Incremental save: {i+1}/{len(papers)} papers processed")
        
        main_pbar.close()
        
        # Final save
        self.save_data_chunked(all_data, output_dir, conference, venue_year)
        
        print(f"\n✅ Crawl completed!")
        print(f"📊 Total papers processed: {len(all_data)}")
        print(f"📊 Reviews extracted via API V1: {self.review_stats['api_v1_extracted']}")
        print(f"📊 Reviews extracted via fallback: {self.review_stats['fallback_extracted']}")
        
        return all_data
    
    def save_csv_summary(self, data: List[Dict], filename: str):
        """Save a CSV summary of papers and reviews"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header - Added supplement_downloaded column
            writer.writerow([
                'paper_id', 'title', 'abstract', 'authors', 
                'num_reviews', 'avg_rating', 'venue', 'pdf_downloaded', 'supplement_downloaded', 'json_file'
            ])
            
            for item in data:
                paper = item['paper']
                reviews = item['reviews']
                pdf_downloaded = item.get('pdf_downloaded', False)
                supplement_downloaded = item.get('supplement_downloaded', False)
                
                # Extract ratings
                ratings = []
                for review in reviews:
                    content = review.get('content', {})
                    if 'rating' in content:
                        try:
                            # Handle different rating formats
                            rating_value = content['rating']
                            if isinstance(rating_value, dict) and 'value' in rating_value:
                                rating_str = str(rating_value['value'])
                            else:
                                rating_str = str(rating_value)
                            
                            if ':' in rating_str:
                                rating = float(rating_str.split(':')[0])
                            else:
                                rating = float(rating_str)
                            ratings.append(rating)
                        except:
                            pass
                
                avg_rating = sum(ratings) / len(ratings) if ratings else None
                
                # Extract paper content - handle both dict and object formats
                if isinstance(paper, dict):
                    paper_content = paper.get('content', {})
                    paper_id = paper.get('id', '')
                else:
                    paper_content = getattr(paper, 'content', {})
                    paper_id = getattr(paper, 'id', '')
                
                # Handle content fields that might be dicts with 'value' key
                def get_content_value(field):
                    if isinstance(field, dict) and 'value' in field:
                        return field['value']
                    return field if field else ''
                
                title = get_content_value(paper_content.get('title', ''))
                abstract = get_content_value(paper_content.get('abstract', ''))
                authors = paper_content.get('authors', [])
                venue = get_content_value(paper_content.get('venue', ''))
                
                # Handle authors list
                if isinstance(authors, dict) and 'value' in authors:
                    authors = authors['value']
                if not isinstance(authors, list):
                    authors = [str(authors)] if authors else []
                
                # Truncate abstract safely
                abstract_str = str(abstract)
                if len(abstract_str) > 500:
                    abstract_str = abstract_str[:500] + '...'
                
                # Get the JSON file containing this paper
                json_file = self.paper_to_file_map.get(paper_id, 'unknown')
                
                writer.writerow([
                    paper_id,
                    title,
                    abstract_str,
                    ', '.join(str(author) for author in authors),
                    len(reviews),
                    round(avg_rating, 2) if avg_rating else None,
                    venue,
                    pdf_downloaded,
                    supplement_downloaded,
                    json_file
                ])
    
    def save_single_chunk(self, chunk_data: List[Dict], output_dir: str, conference: str, venue_year: str, chunk_idx: int):
        """Save a single chunk of data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunk_filename = f"{conference}_{venue_year}_chunk_{chunk_idx:03d}_{timestamp}.json"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        # Update mapping for each paper in chunk
        for item in chunk_data:
            paper_id = item['paper'].get('id', '') if isinstance(item['paper'], dict) else getattr(item['paper'], 'id', '')
            if paper_id:
                self.paper_to_file_map[paper_id] = chunk_filename

    def save_final_metadata(self, all_data: List[Dict], output_dir: str, conference: str, venue_year: str):
        """Save mapping file and CSV summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save mapping file
        mapping_file = os.path.join(output_dir, f"{conference}_{venue_year}_paper_mapping_{timestamp}.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.paper_to_file_map, f, indent=2, ensure_ascii=False)
        
        # Save CSV summary
        csv_file = os.path.join(output_dir, f"{conference}_{venue_year}_summary_{timestamp}.csv")
        self.save_csv_summary(all_data, csv_file)
        
        print(f"🗂️  Paper mapping saved to {mapping_file}")
        print(f"📊 CSV summary saved to {csv_file}")

def test_client():
    """Test the OpenReview client connection"""
    try:
        client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username='weihezhai@gmail.com',
            password='aMmeWArff6a3Z2k'
        )
        
        # Test by getting notes with a specific invitation (ICLR is usually available)
        print("🔧 Testing OpenReview client...")
        test_invitations = [
            'ICLR.cc/2024/Conference/-/Submission',
            'ICLR.cc/2023/Conference/-/Submission',
            'NeurIPS.cc/2023/Conference/-/Submission'
        ]
        
        for invitation in test_invitations:
            try:
                print(f"🔍 Testing invitation: {invitation}")
                notes = list(client.get_notes(invitation=invitation, limit=3))
                if notes:
                    print(f"✅ Successfully connected! Found {len(notes)} notes for {invitation}.")
                    print(f"📄 Sample note ID: {notes[0].id}")
                    print(f"📝 Sample title: {notes[0].content.get('title', 'No title')}")
                    return True
            except Exception as e:
                print(f"❌ Error with {invitation}: {e}")
                continue
        
        return False
    except Exception as e:
        print(f"❌ Client test failed: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Crawl academic conference papers and download PDFs from OpenReview')
    parser.add_argument('--output-dir', '-o', type=str, default='conference_data',
                       help='Output directory for storing crawled data and PDFs (default: conference_data)')
    parser.add_argument('--limit', '-l', type=int, default=0,
                       help='Maximum number of papers to crawl (default: unlimited), 0 means unlimited')
    parser.add_argument('--no-pdfs', action='store_true',
                       help='Skip PDF downloads, only crawl paper metadata and reviews')
    parser.add_argument('--test-connection', action='store_true',
                       help='Only test the OpenReview API connection and exit')
    parser.add_argument('--venue', type=str, default='2024',
                       choices=['2025','2024', '2023', '2022', '2021'], 
                       help='Conference venue year to crawl (default: 2024)')
    parser.add_argument('--conference', '-c', type=str, default='neurips',
                       choices=['neurips', 'iclr', 'icml', 'emnlp'],
                       help='Conference name to crawl (default: neurips)')
    
    args = parser.parse_args()
    
    # Test connection if requested
    if args.test_connection:
        if test_client():
            print("✅ OpenReview API connection successful!")
        else:
            print("❌ Failed to connect to OpenReview API")
        return
    
    # Test connection first
    if not test_client():
        print("❌ Failed to connect to OpenReview API")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {os.path.abspath(args.output_dir)}")
    
    crawler = OpenReviewCrawlerWithPDFs()
    
    # Update venue preference
    conf_name = args.conference.upper()
    print(f"🎯 Targeting {conf_name} {args.venue} papers")
    
    # Crawl papers as requested
    limit_text = "unlimited" if args.limit == 0 else str(args.limit)
    actual_limit = None if args.limit == 0 else args.limit
    
    print(f"🚀 Starting crawl for {limit_text} {conf_name} {args.venue} papers...")
    if args.no_pdfs:
        print("📄 PDF downloads disabled")
        data = crawler.crawl_metadata_only(output_dir=args.output_dir, limit=actual_limit, 
                                         venue_year=args.venue, conference=args.conference)
    else:
        print("📄 PDF downloads enabled")
        data = crawler.crawl_all_data_with_pdfs(output_dir=args.output_dir, limit=actual_limit, 
                                              venue_year=args.venue, conference=args.conference)
    
    print(f"✅ Crawl completed! Collected {len(data)} papers.")

if __name__ == "__main__":
    # Add usage examples as module docstring
    """
    Usage Examples:
    
    # Crawl 1000 NeurIPS 2024 papers with PDFs to custom directory
    python spider_with_pdfs_api_v1.py --output-dir /path/to/output --limit 1000 --conference neurips
    
    # Crawl ICLR 2023 papers without PDFs (uses API V1 method)
    python spider_with_pdfs_api_v1.py --conference iclr --venue 2023 --no-pdfs --limit 500
    
    # Crawl ICML 2023 papers with PDFs
    python spider_with_pdfs_api_v1.py --conference icml --venue 2023 --limit 200
    
    # Crawl EMNLP 2024 papers
    python spider_with_pdfs_api_v1.py --conference emnlp --venue 2024
    
    # Test connection only
    python spider_with_pdfs_api_v1.py --test-connection
    
    # Crawl unlimited papers (all available)
    python spider_with_pdfs_api_v1.py --limit 0 --conference neurips
    """
    main()
