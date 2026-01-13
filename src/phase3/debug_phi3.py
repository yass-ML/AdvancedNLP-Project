import requests
import json

def test_phi3():
    model_name = "phi3:mini"
    problem = "Find the value of x if 2x + 5 = 15."
    
    categories = ["Algebra", "Counting & Probability", "Geometry", "Intermediate Algebra", "Number Theory", "Prealgebra", "Precalculus"]
    
    prompt = f"""
        Classify the following math problem into exactly one of these categories: {', '.join(categories)}.
        Return ONLY the category name. Do not include any other text.
        
         Problem: {problem}
        
        Category:
        """
        
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    print(f"Testing {model_name}...")
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        result = data.get("response", "").strip()
        print(f"\nRaw Response:\n'{result}'")
        print(f"Token Count: {len(result.split())} words (approx)")
        
        found = False
        for cat in sorted(categories, key=len, reverse=True):
            if cat.lower() in result.lower():
                print(f"Matched Category: {cat}")
                found = True
                break
        if not found:
            print("No category matched.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_phi3()
