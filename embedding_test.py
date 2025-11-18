import requests
import json

BASE_URL = "http://127.0.0.1:8002"

def test_rank():
    url = f"{BASE_URL}/rank_documents"
    payload = {
        "task": "Given a web search query, retrieve relevant passages that answer the query",
        "query": "What is the capital of China?",
        "documents": [
            "Beijing is the capital city of China.",
            "China's largest city is Shanghai.",
            "Tokyo is the capital of Japan.",
            "The Great Wall is located in northern China."
        ],
        "topk": 2
    }

    print("🚀 Sending ranking request...")
    resp = requests.post(url, json=payload)

    if resp.status_code == 200:
        data = resp.json()
        print("\n✅ Response:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(f"❌ Failed: {resp.status_code}")
        print(resp.text)

if __name__ == "__main__":
    test_rank()

