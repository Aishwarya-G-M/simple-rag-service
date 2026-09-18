"""
Evaluation harness for vanilla RAG service - SMS spam detection
Tests the service endpoint and records results for LASR application

Updated to match actual API response: {"answer": "...", "retrieved": [...]}
"""

import requests
import json
import time
import re
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import os

# ============ CONFIGURATION ============
BASE_URL = "http://localhost:8000"
QUERY_ENDPOINT = f"{BASE_URL}/rag/query"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

# Ensure evaluation directory exists
os.makedirs("evaluation", exist_ok=True)
OUTPUT_FILE = f"results_vanilla_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# ============ TEST DATASETS ============
TEST_QUERIES = [
    # Clear spam examples
    {
        "message": "Congratulations! You've won a £1000 Walmart gift card. Click here to claim: bit.ly/xyz123",
        "expected_label": "spam",
        "category": "prize_scam",
        "urgency": "high"
    },
    {
        "message": "URGENT: Your account will be suspended. Verify now at secure-bank-verify.com/login",
        "expected_label": "spam",
        "category": "phishing",
        "urgency": "high"
    },
    {
        "message": "Win an iPhone 15! Free giveaway! Text CLAIM to 55555 now!",
        "expected_label": "spam",
        "category": "prize_scam",
        "urgency": "high"
    },
    {
        "message": "Your package delivery failed. Reschedule at: fake-dhl-tracking.net/reschedule",
        "expected_label": "spam",
        "category": "delivery_scam",
        "urgency": "medium"
    },
    {
        "message": "Amazon: Suspicious login attempt. Reset password: amaz0n-security.com/reset",
        "expected_label": "spam",
        "category": "phishing",
        "urgency": "high"
    },
    
    # Clear ham examples
    {
        "message": "Hey, are we still on for lunch tomorrow at 1pm?",
        "expected_label": "ham",
        "category": "personal",
        "urgency": "low"
    },
    {
        "message": "Your OTP for login is 847293. Valid for 10 minutes.",
        "expected_label": "ham",
        "category": "transactional",
        "urgency": "medium"
    },
    {
        "message": "Meeting rescheduled to 3pm today. Conference room B.",
        "expected_label": "ham",
        "category": "work",
        "urgency": "medium"
    },
    {
        "message": "Happy birthday! Hope you have a great day!",
        "expected_label": "ham",
        "category": "personal",
        "urgency": "low"
    },
    {
        "message": "Your flight BA249 to London is on time. Gate 45.",
        "expected_label": "ham",
        "category": "transactional",
        "urgency": "medium"
    },
    
    # Edge cases / adversarial examples
    {
        "message": "You won! Claim your prize at winner-claim.com. Not a scam, real offer!",
        "expected_label": "spam",
        "category": "adversarial",
        "urgency": "high"
    },
    {
        "message": "This is not spam but your bank needs you to verify: bank-secure.com",
        "expected_label": "spam",
        "category": "adversarial",
        "urgency": "high"
    },
    {
        "message": "Free entry to win a premium subscription! Terms apply. No purchase necessary.",
        "expected_label": "spam",
        "category": "promotional",
        "urgency": "medium"
    },
]

# ============ EVALUATION FUNCTIONS ============

def check_health() -> bool:
    """Check if the service is running"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False

def query_rag(message_text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Send a query to the RAG service and return the response"""
    payload = {
        "message": message_text,
        "threshold": threshold
    }
    
    try:
        response = requests.post(
            QUERY_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "timeout", "answer": None, "retrieved": []}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}", "answer": None, "retrieved": []}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "answer": None, "retrieved": []}

def extract_prediction_from_answer(answer_text: str) -> tuple:
    """
    Extract spam/ham prediction and confidence from the LLM answer.
    Returns (prediction, confidence)
    
    Patterns to match:
    - "**Answer:** Spam" or "**Answer:** Not spam"
    - "**Spam**" or "**Not spam**"
    - "Likely spam" or "almost certainly spam"
    - "Not spam" or "legitimate"
    """
    if not answer_text:
        return "error", 0.0
    
    answer_lower = answer_text.lower()
    
    # Check for explicit spam/ham predictions
    spam_patterns = [
        r"\*\*answer:\*\*\s*spam",
        r"\*\*spam\*\*",
        r"answer:\s*spam",
        r"likely spam",
        r"almost certainly spam",
        r"this message is spam",
        r"is \*\*spam\*\*",
        r"phishing",
        r"scam",
    ]
    
    ham_patterns = [
        r"\*\*answer:\*\*\s*not spam",
        r"\*\*not spam\*\*",
        r"answer:\s*not spam",
        r"not spam",
        r"legitimate",
        r"normal",
        r"safe",
        r"ordinary",
    ]
    
    # Check for inability to determine
    uncertain_patterns = [
        r"not able to determine",
        r"can't determine",
        r"cannot determine",
        r"unable to determine",
        r"no indication",
        r"don't have a reference",
        r"unlabeled",
    ]
    
    # Check uncertainty first
    for pattern in uncertain_patterns:
        if re.search(pattern, answer_lower):
            return "uncertain", 0.3
    
    # Count spam and ham indicators
    spam_score = sum(1 for pattern in spam_patterns if re.search(pattern, answer_lower))
    ham_score = sum(1 for pattern in ham_patterns if re.search(pattern, answer_lower))
    
    # Determine prediction
    if spam_score > ham_score:
        confidence = min(0.5 + (spam_score * 0.1), 0.95)
        return "spam", confidence
    elif ham_score > spam_score:
        confidence = min(0.5 + (ham_score * 0.1), 0.95)
        return "ham", confidence
    elif spam_score > 0:
        return "spam", 0.6
    elif ham_score > 0:
        return "ham", 0.6
    else:
        return "uncertain", 0.3

def evaluate_response(result: Dict[str, Any], expected_label: str) -> Dict[str, Any]:
    """Evaluate a single query result"""
    if "error" in result:
        return {
            "prediction": "error",
            "confidence": 0.0,
            "correct": False,
            "error": result["error"]
        }
    
    answer_text = result.get("answer", "")
    prediction, confidence = extract_prediction_from_answer(answer_text)
    
    if prediction == "error" or prediction == "uncertain":
        correct = False
    else:
        correct = (prediction == expected_label)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "correct": correct,
        "error": None
    }

def run_evaluation():
    """Run the full evaluation and save results"""
    print("=" * 60)
    print("VANILLA RAG EVALUATION - SMS SPAM DETECTION")
    print("=" * 60)
    
    # Check health
    print("\n[1/3] Checking service health...")
    if not check_health():
        print("❌ Service is not running. Please start the service first:")
        print("   cd simple-rag-service")
        print("   uvicorn app.main:app --reload")
        return None
    
    print("✅ Service is healthy")
    print(f"   Endpoint: {QUERY_ENDPOINT}")
    
    # Run tests
    print(f"\n[2/3] Running {len(TEST_QUERIES)} test queries...")
    results = []
    
    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"  Testing {i}/{len(TEST_QUERIES)}: {test_case['category']}...", end=" ")
        
        # Query the service
        start_time = time.time()
        rag_response = query_rag(test_case["message"])
        latency = time.time() - start_time
        
        # Evaluate response
        eval_result = evaluate_response(rag_response, test_case["expected_label"])
        
        # Compile result
        result = {
            "test_id": i,
            "message": test_case["message"],
            "expected_label": test_case["expected_label"],
            "category": test_case["category"],
            "urgency": test_case["urgency"],
            "prediction": eval_result["prediction"],
            "confidence": eval_result["confidence"],
            "correct": eval_result["correct"],
            "latency_ms": round(latency * 1000, 2),
            "num_retrieved": len(rag_response.get("retrieved", [])),
            "avg_retrieval_distance": sum(doc.get("distance", 0) for doc in rag_response.get("retrieved", [])) / max(1, len(rag_response.get("retrieved", []))),
            "llm_answer": rag_response.get("answer", ""),
            "error": eval_result.get("error"),
            "timestamp": datetime.now().isoformat()
        }
        
        results.append(result)
        
        # Print status
        status = "✅" if eval_result["correct"] else "❌"
        print(f"{status} (pred: {eval_result['prediction']}, conf: {eval_result['confidence']:.2f}, latency: {latency:.2f}s)")
    
    # Calculate metrics
    print(f"\n[3/3] Calculating metrics...")
    
    total_tests = len(results)
    correct_predictions = sum(1 for r in results if r["correct"])
    accuracy = correct_predictions / total_tests if total_tests > 0 else 0
    
    spam_tests = [r for r in results if r["expected_label"] == "spam"]
    ham_tests = [r for r in results if r["expected_label"] == "ham"]
    
    spam_detected = sum(1 for r in spam_tests if r["prediction"] == "spam")
    ham_detected = sum(1 for r in ham_tests if r["prediction"] == "ham")
    
    spam_recall = spam_detected / len(spam_tests) if spam_tests else 0
    ham_recall = ham_detected / len(ham_tests) if ham_tests else 0
    
    avg_latency = sum(r["latency_ms"] for r in results) / total_tests if total_tests > 0 else 0
    avg_confidence = sum(r["confidence"] for r in results if r["prediction"] != "error") / max(1, sum(1 for r in results if r["prediction"] != "error"))
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total tests:        {total_tests}")
    print(f"Correct:            {correct_predictions}")
    print(f"Accuracy:           {accuracy:.2%}")
    print(f"Spam recall:        {spam_recall:.2%} ({spam_detected}/{len(spam_tests)})")
    print(f"Ham recall:         {ham_recall:.2%} ({ham_detected}/{len(ham_tests)})")
    print(f"Avg latency:        {avg_latency:.2f}ms")
    print(f"Avg confidence:     {avg_confidence:.2f}")
    print(f"Avg retrieved docs: {df['num_retrieved'].mean():.1f}")
    print(f"Avg retrieval dist: {df['avg_retrieval_distance']:.3f}")
    print(f"\nResults saved to:   {OUTPUT_FILE}")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    results_df = run_evaluation()
    
    if results_df is not None:
        print("\nDetailed results:")
        print(results_df[["test_id", "category", "expected_label", "prediction", "confidence", "correct", "latency_ms"]].to_string())