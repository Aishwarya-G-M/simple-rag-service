# Evaluation Guide - SMS Spam Detection RAG Service

This guide documents how to run systematic evaluations of the vanilla RAG service.

## Quick Start

### Prerequisites

1. **Start the RAG service** (in one terminal):
```bash
cd simple-rag-service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Verify the service is running**:
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy"}
```

3. **Run the evaluation** (in another terminal):
```bash
python test_vanilla_rag.py
```

---

## Test Dataset

The evaluation uses **13 test cases** across 8 categories:

### Spam Examples (8 tests)
| Category | Count | Example |
|----------|-------|---------|
| Prize scam | 2 | "Congratulations! You've won a £1000 Walmart gift card..." |
| Phishing | 2 | "URGENT: Your account will be suspended..." |
| Delivery scam | 1 | "Your package delivery failed..." |
| Adversarial | 2 | "You won! Claim your prize... Not a scam, real offer!" |
| Promotional | 1 | "Free entry to win a premium subscription..." |

### Ham Examples (5 tests)
| Category | Count | Example |
|----------|-------|---------|
| Personal | 2 | "Hey, are we still on for lunch tomorrow?" |
| Transactional | 2 | "Your OTP for login is 847293..." |
| Work | 1 | "Meeting rescheduled to 3pm today..." |

---

## API Specification

### Endpoint
```
POST http://localhost:8000/api/v1/query
```

### Request
```json
{
  "message": "string (required)",
  "threshold": 0.5 (optional, default: 0.5)
}
```

### Response
```json
{
  "answer": "string (LLM reasoning + prediction)",
  "retrieved": [
    {
      "index": 1234,
      "distance": 0.547,
      "text": "retrieved document text..."
    }
  ]
}
```

---

## Evaluation Metrics

The script calculates:

### Primary Metrics
- **Accuracy**: Overall correct predictions / total tests
- **Spam Recall**: Correctly identified spam / total spam tests
- **Ham Recall**: Correctly identified ham / total ham tests

### Performance Metrics
- **Average Latency**: Mean response time in milliseconds
- **Average Confidence**: Mean confidence score across predictions
- **Retrieval Quality**: Average embedding distance of retrieved documents

### Output File
Results are saved to:
```
evaluation/results_vanilla_rag_YYYYMMDD_HHMMSS.csv
```

Columns include:
- `test_id`, `message`, `expected_label`, `category`, `urgency`
- `prediction`, `confidence`, `correct`, `latency_ms`
- `num_retrieved`, `avg_retrieval_distance`
- `llm_answer`, `error`, `timestamp`

---

## Baseline Results

**Run date**: September 18, 2026

### Summary
| Metric | Value |
|--------|-------|
| **Total tests** | 13 |
| **Accuracy** | 92.31% (12/13) |
| **Spam recall** | 100% (8/8) |
| **Ham recall** | 80% (4/5) |
| **Avg latency** | 919ms |
| **Avg confidence** | 0.68 |
| **Avg retrieval distance** | 0.891 |

### Performance by Category
| Category | Accuracy | Tests |
|----------|----------|-------|
| Prize scam | 100% | 2/2 |
| Phishing | 100% | 2/2 |
| Delivery scam | 100% | 1/1 |
| Personal | 100% | 2/2 |
| Work | 100% | 1/1 |
| Adversarial | 100% | 2/2 |
| Promotional | 100% | 1/1 |
| **Transactional** | **50%** | **1/2** ⚠️ |

### Error Analysis
**1 false positive**:
- Test 7: OTP message ("Your OTP for login is 847293...") classified as spam
- Reason: Retrieved documents contained similar-looking promotional codes
- Confidence: 0.8 (overconfident error)

### Sample Output
```
test_id,category,expected_label,prediction,confidence,correct,latency_ms
1,prize_scam,spam,spam,0.6,True,902.01
2,phishing,spam,spam,0.7,True,906.50
3,prize_scam,spam,spam,0.6,True,920.55
4,delivery_scam,spam,spam,0.6,True,922.72
5,phishing,spam,spam,0.7,True,631.24
6,personal,ham,ham,0.9,True,596.96
7,transactional,ham,spam,0.8,False,1845.02
8,work,ham,ham,0.8,True,621.94
9,personal,ham,ham,0.7,True,693.59
10,transactional,ham,ham,0.6,True,945.68
11,adversarial,spam,spam,0.7,True,808.67
12,adversarial,spam,spam,0.6,True,1228.88
13,promotional,spam,spam,0.6,True,920.99
```

---

## How It Works

### Prediction Extraction
The evaluation script parses the LLM's `answer` field using pattern matching:

**Spam indicators**:
- `**Answer:** Spam`, `**Spam**`, `likely spam`, `phishing`, `scam`

**Ham indicators**:
- `**Answer:** Not spam`, `**Not spam**`, `legitimate`, `normal`, `safe`

**Uncertain**:
- `not able to determine`, `can't determine`, `unlabeled`

### Confidence Scoring
- Base confidence: 0.5
- +0.1 per matching indicator (capped at 0.95)
- Uncertain predictions: 0.3

### Example LLM Responses

**Spam detection**:
```
**Spam** – The message follows the same pattern as the labeled examples: 
a congratulatory claim of a prize, a directive to click a link, and a 
generic value ("£1000"). The context indicates that such messages are spam.
```

**Ham detection**:
```
Not spam. The message follows the same conversational pattern as the 
examples, which are normal, non‑spam personal reminders about meetings 
or lunch. The wording, timing, and context all indicate a legitimate, 
ordinary inquiry.
```

---

## Extending the Evaluation

### Adding New Test Cases
Edit `test_vanilla_rag.py` and add to `TEST_QUERIES`:

```python
{
    "message": "Your test message here",
    "expected_label": "spam",  # or "ham"
    "category": "your_category",
    "urgency": "low"  # or "medium", "high"
}
```

### Custom Metrics
Add calculations in the `run_evaluation()` function:
```python
# Example: Calculate precision
spam_predicted = [r for r in results if r["prediction"] == "spam"]
spam_true_positives = sum(1 for r in spam_predicted if r["correct"])
precision = spam_true_positives / len(spam_predicted) if spam_predicted else 0
```

---

## Troubleshooting

### Service not running
```
❌ Service is not running. Please start the service first:
   cd simple-rag-service
   uvicorn app.main:app --reload
```

### Connection refused
- Ensure the service is running on port 8000
- Check firewall settings
- Try `curl http://localhost:8000/health` manually

### 404 errors
- Verify endpoint is `/api/v1/query` (not `/query`)
- Check request payload has `message` and `threshold` fields

### No is_spam field
- The API returns `answer` (text), not structured `is_spam` (boolean)
- The evaluation script extracts predictions from text

---

## Files

- `test_vanilla_rag.py` - Main evaluation script
- `evaluation/` - Output directory for results CSVs
- `data/sms_spam.csv` - Training corpus (5,572 messages)
- `app/main.py` - RAG service API
- `app/faiss_retriever.py` - FAISS-based retrieval
- `app/llm.py` - LLM response generation

---

**Last updated**: September 18, 2026  
**Author**: Aishwarya G M