import requests
import time
import statistics
import json
import random

BASE_URL = "http://127.0.0.1:8000"
TEST_ROUNDS = 100   # 请求次数
DOC_COUNT = 100      # 每次文档数量

def generate_test_documents(n: int):
    """生成模拟文档"""
    docs = []
    for i in range(n):
        docs.append(f"This is a test document about AI number {i}. "
                    f"Artificial intelligence and machine learning are discussed here.")
    # 添加一些无关的干扰文本
    docs += [f"Document about random topic {i}: banana, sports, or music." for i in range(n // 5)]
    random.shuffle(docs)
    return docs[:n]

def test_latency():
    url = f"{BASE_URL}/rank_documents"
    payload_template = {
        "task": "Given a web search query, retrieve relevant passages that answer the query",
        "query": "What is artificial intelligence?",
        "documents": generate_test_documents(DOC_COUNT),
        "topk": 10
    }

    latencies = []
    print(f"🚀 Starting benchmark: {TEST_ROUNDS} requests × {DOC_COUNT} docs each...\n")

    for i in range(TEST_ROUNDS):
        start = time.perf_counter()
        response = requests.post(url, json=payload_template)
        end = time.perf_counter()
        latency = (end - start) * 1000  # 转毫秒

        if response.status_code == 200:
            latencies.append(latency)
            print(f"✅ Request {i+1}/{TEST_ROUNDS}: {latency:.2f} ms")
        else:
            print(f"❌ Request {i+1}/{TEST_ROUNDS} failed: {response.status_code}")
            print(response.text)
            latencies.append(None)

    valid_latencies = [x for x in latencies if x is not None]

    if valid_latencies:
        avg_latency = statistics.mean(valid_latencies)
        p95_latency = statistics.quantiles(valid_latencies, n=100)[94]
        print("\n📊 Benchmark Summary:")
        print(f"  • Total requests: {TEST_ROUNDS}")
        print(f"  • Average latency: {avg_latency:.2f} ms")
        print(f"  • 95th percentile latency: {p95_latency:.2f} ms")
        print(f"  • Min: {min(valid_latencies):.2f} ms, Max: {max(valid_latencies):.2f} ms")
    else:
        print("⚠️ All requests failed, check the API endpoint.")

def test_health():
    url = f"{BASE_URL}/health"
    print("🔍 Checking health endpoint...")
    r = requests.get(url)
    print("✅", r.json())

if __name__ == "__main__":
    test_health()
    test_latency()

