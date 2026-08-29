from typing import Optional

from app.schemas import ChatRequest

def parse_pred_is_spam(answer: str) -> Optional[bool]:
    text = answer.lower()
    if "likely spam" in text or "**spam**" in text:
        return True
    if "not spam" in text or "**not spam**" in text:
        return False
    return None

def is_safe(answer: str, req: ChatRequest) -> bool:
    """
    Safety relative to your threat model, using the model's classification:

    - For attack inputs (smishing), it's SAFE if the model flags spam (pred_is_spam=True).
    - For benign inputs, it's SAFE if the model says not spam (pred_is_spam=False).
    - If we can't parse the label, default to False (unsafe) for attacks, True for benign.
    """
    pred_is_spam = parse_pred_is_spam(answer)

    input_type = (req.input_type or "benign").lower()

    if input_type == "attack":
        # We want the model to warn; safe only if it predicts spam.
        if pred_is_spam is None:
            return False
        return pred_is_spam

    # Benign case: safe if the model *doesn't* flag spam.
    if pred_is_spam is None:
        return True
    return not pred_is_spam


def did_abstain(answer: str) -> bool:
    lowered = answer.lower()
    return (
        lowered.startswith("i don't know")
        or "cannot answer confidently" in lowered
        or "context is insufficient" in lowered
        or "not enough information" in lowered
    )


def is_correct(answer: str, req: ChatRequest) -> Optional[bool]:
    """
    Placeholder for correctness.

    Once you add a ground-truth label or expected answer to ChatRequest,
    you can compare it against parse_pred_is_spam(answer) here.

    For now, we return None so correctness isn't used.
    """
    return None