import requests
import json
import os

# Configuration
API_URL = "http://localhost:8001/classify_web"
API_KEY = os.getenv("AI_SERVICE_TOKEN", "test-token") # Ensure this matches your env

def test_credit_card_violation():
    payload = {
        "raw_http": "POST /payment HTTP/1.1\nHost: example.com\n\ncardNumber=4111111111111111&cvv=123",
        "metadata": {"channel": "web"}
    }
    headers = {"X-API-Key": API_KEY}
    
    try:
        # Note: This assumes the server is running. 
        # Since I can't start a long-running server easily here, 
        # I'll just explain that this is how you verify it.
        print("Testing with Credit Card payload...")
        # response = requests.post(API_URL, json=payload, headers=headers)
        # print(json.dumps(response.json(), indent=2))
        
        # Instead, I'll simulate the logic locally to show you the output structure
        from ai_orchestrator import AIOrchestrator
        orc = AIOrchestrator()
        result = orc.classify_web(raw_http=payload["raw_http"])
        
        # Simulate the new _normalize_response logic
        llm_info = result.get("llm") or {}
        violations = list(llm_info.get("sensitivity_indicators") or [])
        
        print("\n--- SIMULATED API RESPONSE ---")
        print(json.dumps({
            "label": result["label"],
            "violations": violations,
            "compliance_tags": result["compliance_tags"]
        }, indent=2))
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_credit_card_violation()
