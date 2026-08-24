import pandas as pd

from app.documents import load_sms_spam_csv


def test_load_sms_spam_csv_normalizes_columns(tmp_path):
    csv_file = tmp_path / "sms_spam.csv"

    pd.DataFrame(
        {
            "v1": ["spam", "ham"],
            "v2": ["Claim your prize now", "See you later"],
        }
    ).to_csv(csv_file, index=False, encoding="latin-1")

    documents = load_sms_spam_csv(str(csv_file))

    assert len(documents) == 2
    assert documents[0] == {
        "id": "0",
        "text": "Claim your prize now",
        "label": "spam",
        "source": "sms_spam_csv",
    }
    assert documents[1]["label"] == "ham"