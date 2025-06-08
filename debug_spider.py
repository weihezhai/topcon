import requests

def test_openreview_api():
    base_url = "https://api.openreview.net"
    
    # Test basic connectivity
    print("Testing OpenReview API connectivity...")
    
    # Try to get recent notes without specific invitation
    url = f"{base_url}/notes"
    params = {
        'limit': 5
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Basic API test - Status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response content: {response.text[:500]}...")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data.get('notes', []))} notes")
            if data.get('notes'):
                print(f"Sample invitation: {data['notes'][0].get('invitation', 'N/A')}")
        
        # Try different approach - search for any notes with limit
        print("\nTrying different API endpoint...")
        
        # Try the groups endpoint to see available venues
        groups_url = f"{base_url}/groups"
        print(f"Testing groups endpoint: {groups_url}")
        groups_response = requests.get(groups_url, params={'limit': 10})
        print(f"Groups API - Status: {groups_response.status_code}")
        if groups_response.status_code == 200:
            groups_data = groups_response.json()
            print(f"Found {len(groups_data.get('groups', []))} groups")
            for group in groups_data.get('groups', [])[:5]:
                print(f"  - {group.get('id', 'N/A')}")
        
        # Try to find NeurIPS 2023 instead (which should exist)
        print("\nSearching for NeurIPS 2023 invitations...")
        invitation_url = f"{base_url}/invitations"
        invitation_params = {
            'regex': 'NeurIPS.*2023',
            'limit': 20
        }
        
        inv_response = requests.get(invitation_url, params=invitation_params)
        print(f"Invitations API - Status: {inv_response.status_code}")
        print(f"Invitations Response: {inv_response.text[:500]}...")
        
        if inv_response.status_code == 200:
            inv_data = inv_response.json()
            print(f"Found {len(inv_data.get('invitations', []))} NeurIPS 2023 invitations:")
            for inv in inv_data.get('invitations', [])[:10]:
                print(f"  - {inv.get('id', 'N/A')}")
        
        # Try a simpler approach - get notes with a known venue
        print("\nTrying to get notes with simple parameters...")
        simple_url = f"{base_url}/notes"
        simple_params = {
            'content.venue': 'NeurIPS 2023',
            'limit': 5
        }
        simple_response = requests.get(simple_url, params=simple_params)
        print(f"Simple notes search - Status: {simple_response.status_code}")
        print(f"Simple response: {simple_response.text[:300]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_openreview_api()