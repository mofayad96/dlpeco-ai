import httpx
import asyncio
import json

# Configuration
AI_URL = "http://localhost:8001"
AI_TOKEN = "default-secret-token" # Ensure this matches your .env

async def test_updates():
    async with httpx.AsyncClient() as client:
        headers = {"X-API-Key": AI_TOKEN}
        
        # 1. Test Granular Feedback (violations field)
        print("\n--- 1. Testing Granular Feedback ---")
        payload_violation = {
            "text": "My credit card number is 4111 1111 1111 1111",
            "metadata": {"channel": "general"}
        }
        resp = await client.post(f"{AI_URL}/classify", json=payload_violation, headers=headers)
        data = resp.json()
        print(f"Label: {data.get('label')}")
        print(f"Violations: {data.get('violations')}")

        # 2. Test Smart Filtering (scan_scope)
        print("\n--- 2. Testing Smart Filtering (scan_scope: ['credit_card']) ---")
        payload_scope = {
            "text": "My credit card is 4111 1111 1111 1111 and my email is test@example.com",
            "scan_scope": ["credit_card"],
            "metadata": {"channel": "general"}
        }
        resp = await client.post(f"{AI_URL}/classify", json=payload_scope, headers=headers)
        data = resp.json()
        print(f"Violations (Expected only credit_card): {data.get('violations')}")

        # 3. Test Label Downgrading
        # To test this, we send content that is "Restricted" but doesn't match the scope
        print("\n--- 3. Testing Label Downgrading ---")
        payload_downgrade = {
            "text": "Extremely sensitive restricted corporate strategy document",
            "scan_scope": ["non_existent_scope"], # Force no violations match
            "metadata": {"channel": "general"}
        }
        resp = await client.post(f"{AI_URL}/classify", json=payload_downgrade, headers=headers)
        data = resp.json()
        print(f"Label (Expected 'Internal' due to downgrade): {data.get('label')}")
        print(f"Violations (Expected empty): {data.get('violations')}")

if __name__ == "__main__":
    try:
        asyncio.run(test_updates())
    except Exception as e:
        print(f"Error connecting to AI service: {e}")
        print("Ensure the AI service is running at http://localhost:8001")
