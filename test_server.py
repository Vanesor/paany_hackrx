import requests

try:
    response = requests.get("http://localhost:10000/api/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {str(e)}")
