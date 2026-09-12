import csv
import pathlib
import requests

BASE_URL = "http://localhost:8000"

def load_rows():
    here = pathlib.Path(__file__).parent
    path = here.parent / "data" / "spam-text-messages-dataset.csv"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        print("Fieldnames:", reader.fieldnames)
        for i, row in enumerate(reader):
            # row["image"] and row["text"] exist
            rows.append(
                {
                    "id": f"hf_spam_{i}",
                    "text": row["text"],
                }
            )
    return rows

def run():
    rows = load_rows()
    for row in rows:
        payload = {
            "message": row["text"],
            "top_k": 5,
            "scenario_id": row["id"],
            "input_type": "attack",
            "attack_type": "smishing",
            "is_spam_label": True,
        }
        resp = requests.post(f"{BASE_URL}/rag/query", json=payload)
        resp.raise_for_status()

if __name__ == "__main__":
    run()