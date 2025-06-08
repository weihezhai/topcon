import openreview
import json
import time
import csv
from datetime import datetime
import os
from typing import Dict, List, Optional

class OpenReviewCrawler:
    def __init__(self):
        # Initialize the OpenReview client with authentication
        self.client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username='weihezhai@gmail.com',
            password='aMmeWArff6a3Z2k'
        )
        
    def get_neurips_2024_papers(self, limit: int = None) -> List[Dict]:
        """Fetch all NeurIPS 2024 papers"""
        print("Searching for NeurIPS 2024 submissions...")
        
        # Try different possible invitation formats for NeurIPS 2024
        possible_invitations = [
            'NeurIPS.cc/2024/Conference/-/Submission',
            'NeurIPS.cc/2024/Conference/Submission',
            'neurips.cc/2024/Conference/-/Submission',
        ]
        
        papers = []
        
        for invitation in possible_invitations:
            try:
                print(f"Trying invitation: {invitation}")
                notes = list(self.client.get_notes(
                    invitation=invitation,
                    limit=limit
                ))
                
                if notes:
                    print(f"Found {len(notes)} papers with invitation: {invitation}")
                    papers = notes
                    break
                else:
                    print(f"No papers found for invitation: {invitation}")
                    
            except Exception as e:
                print(f"Error with invitation {invitation}: {e}")
                continue
        
        # If no NeurIPS 2024 found, try NeurIPS 2023 as fallback
        if not papers:
            print("NeurIPS 2024 not found, trying NeurIPS 2023...")
            try:
                notes = list(self.client.get_notes(
                    invitation='NeurIPS.cc/2023/Conference/-/Submission',
                    limit=limit
                ))
                papers = notes
                print(f"Found {len(papers)} NeurIPS 2023 papers as fallback")
            except Exception as e:
                print(f"Error fetching NeurIPS 2023: {e}")
        
        return papers
    
    def get_paper_reviews(self, paper_id: str) -> List[Dict]:
        """Fetch all reviews for a specific paper"""
        try:
            # Get all notes in the forum (paper discussion)
            notes = list(self.client.get_notes(forum=paper_id))
            
            # Filter for reviews only - check if invitation attribute exists
            reviews = []
            for note in notes:
                invitation = getattr(note, 'invitation', '') or getattr(note, 'invitations', [''])[0] if hasattr(note, 'invitations') else ''
                if 'Review' in invitation and note.id != paper_id:
                    reviews.append(note)
            
            time.sleep(0.1)  # Small delay to be respectful
            return reviews
            
        except Exception as e:
            print(f"Error fetching reviews for {paper_id}: {e}")
            return []
    
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
    
    def crawl_all_data(self, output_dir: str = "neurips2024_data", limit: int = None):
        """Crawl all papers and their reviews"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Fetching papers...")
        papers = self.get_neurips_2024_papers(limit)
        
        # If still no papers, try searching by venue
        if not papers:
            print("Trying venue-based search...")
            papers = self.search_papers_by_venue("NeurIPS 2024", limit)
            
        if not papers:
            papers = self.search_papers_by_venue("NeurIPS 2023", limit)
        
        print(f"Found {len(papers)} papers")
        
        if not papers:
            print("No papers found. Exiting...")
            return []
        
        all_data = []
        
        for i, paper in enumerate(papers):
            paper_id = paper.id
            print(f"Processing paper {i+1}/{len(papers)}: {paper_id}")
            
            reviews = self.get_paper_reviews(paper_id)
            
            paper_data = {
                'paper': paper.to_json() if hasattr(paper, 'to_json') else paper.__dict__,  # Convert note to dict
                'reviews': [review.to_json() if hasattr(review, 'to_json') else review.__dict__ for review in reviews],
                'crawled_at': datetime.now().isoformat()
            }
            
            all_data.append(paper_data)
            
            # Save incrementally every 10 papers
            if i % 10 == 0 and i > 0:
                self.save_data(all_data, output_dir)
        
        # Final save
        self.save_data(all_data, output_dir)
        return all_data
    
    def save_data(self, data: List[Dict], output_dir: str):
        """Save data in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON format (most complete)
        json_file = os.path.join(output_dir, f"neurips_complete_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # CSV format (for analysis)
        csv_file = os.path.join(output_dir, f"neurips_summary_{timestamp}.csv")
        self.save_csv_summary(data, csv_file)
        
        print(f"Data saved to {output_dir}/")
    
    def save_csv_summary(self, data: List[Dict], filename: str):
        """Save a CSV summary of papers and reviews"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'paper_id', 'title', 'abstract', 'authors', 
                'num_reviews', 'avg_rating', 'venue'
            ])
            
            for item in data:
                paper = item['paper']
                reviews = item['reviews']
                
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
                
                writer.writerow([
                    paper_id,
                    title,
                    abstract_str,
                    ', '.join(str(author) for author in authors),
                    len(reviews),
                    round(avg_rating, 2) if avg_rating else None,
                    venue
                ])

def test_client():
    """Test the OpenReview client connection"""
    try:
        client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username='weihezhai@gmail.com',
            password='aMmeWArff6a3Z2k'
        )
        
        # Test by getting notes with a specific invitation (ICLR is usually available)
        print("Testing OpenReview client...")
        test_invitations = [
            'ICLR.cc/2024/Conference/-/Submission',
            'ICLR.cc/2023/Conference/-/Submission',
            'NeurIPS.cc/2023/Conference/-/Submission'
        ]
        
        for invitation in test_invitations:
            try:
                print(f"Testing invitation: {invitation}")
                notes = list(client.get_notes(invitation=invitation, limit=3))
                if notes:
                    print(f"Successfully connected! Found {len(notes)} notes for {invitation}.")
                    print(f"Sample note ID: {notes[0].id}")
                    print(f"Sample title: {notes[0].content.get('title', 'No title')}")
                    return True
            except Exception as e:
                print(f"Error with {invitation}: {e}")
                continue
        
        return False
    except Exception as e:
        print(f"Client test failed: {e}")
        return False

def main():
    # Test connection first
    if not test_client():
        print("Failed to connect to OpenReview API")
        return
    
    crawler = OpenReviewCrawler()
    
    # Start with a small sample for testing
    print("Starting crawl (limit=10 for testing)...")
    data = crawler.crawl_all_data(limit=10)  # Remove limit for full crawl
    
    print(f"Crawl completed! Collected {len(data)} papers with reviews.")

if __name__ == "__main__":
    main()
