import httpx
import os

# Configuration matching the implementation
AI_SERVICE_URL = "http://localhost:8001"
AI_SERVICE_TOKEN = "default-secret-token"

async def test_classification():
    async with httpx.AsyncClient() as client:
        # Test web classification
        payload = {
            "raw_http": "POST /upload HTTP/1.1\nHost: dropbox.com\nfilename=test.csv\n\nsecret_data",
            "metadata": {"channel": "web"}
        }
        print("Testing /classify_web...")
        response = await client.post(
            f"{AI_SERVICE_URL}/classify_web",
            json=payload,
            headers={"X-AI-Token": AI_SERVICE_TOKEN},
            timeout=10.0
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_classification())
