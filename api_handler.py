import json
import requests
import time

# System prompt for classification and localization
SYSTEM_PROMPT = """You are a senior forensic document analyst.
Analyze the provided image of an electricity bill for potential digital forgery.

1. PROJECT IDENTIFICATION: Identify 'provider_company'.
2. REGION LOCALIZATION: Locate 'address_block', 'customer_info', 'summary_boxes', and 'billing_footer'.
3. FORGERY AUDIT: 
   - Check for FONT MISMATCHES: Do digits or names look thicker, thinner, or in a different style than the rest of the bill?
   - Check for ALIGNMENT JITTER: Are any digits slightly out of line (pasted/overlayed)?
   - Check for BACKGROUND ARTIFACTS: Are there "blobs" or rectangular patches around text that suggest a "white paint" overlay?

Return your response ONLY as a JSON object:
{
    "provider_company": "Company Name",
    "regions": {
        "address_block": [ymin, xmin, ymax, xmax],
        "customer_info": [ymin, xmin, ymax, xmax],
        ...
    },
    "ai_forgery_data": {
        "score": 0.0 to 1.0, 
        "reasoning": "Detailed visual evidence of forgery or clean audit.",
        "suspicious_regions": ["name_of_region", ...]
    },
    "is_bill": true/false
}

Normalize coordinates [ymin, xmin, ymax, xmax] on a scale of 0 to 1000."""

class OpenRouterClient:
    def __init__(self, api_key, model="google/gemini-2.0-flash-001"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def call_vision(self, b64_image, retries=2):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:8501",
            "X-Title": "Region Consistency Detector"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this bill and identify the company and regions."},
                        {"image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }
            ],
            "response_format": { "type": "json_object" }
        }

        for attempt in range(retries + 1):
            try:
                response = requests.post(self.url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                return json.loads(content)
            
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429 and attempt < retries: # Rate limit
                    time.sleep(2 ** attempt)
                    continue
                return {"error": f"HTTP {response.status_code}: {response.text}"}
            except Exception as e:
                if attempt < retries:
                    time.sleep(1)
                    continue
                return {"error": str(e)}
        
        return {"error": "Maximum retries exceeded"}

def check_openrouter_status(api_key):
    url = "https://openrouter.ai/api/v1/auth/key"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

