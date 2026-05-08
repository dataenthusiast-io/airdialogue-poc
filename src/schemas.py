from typing import Literal

from pydantic import BaseModel


class IntentClassification(BaseModel):
    """
    Output contract for the LLM classification pipeline.
    predicted_intent is the evaluated field.
    customer_sentiment is exploratory only — not evaluated against ground truth.
    This class is the single source of truth for the structured output format.
    """
    predicted_intent: Literal["book", "change", "cancel"]
    customer_sentiment: Literal["positive", "neutral", "negative"] = "neutral"


if __name__ == "__main__":
    from pydantic import ValidationError

    valid = IntentClassification(predicted_intent="book", customer_sentiment="neutral")
    print(f"Valid input OK: {valid}")

    try:
        IntentClassification(predicted_intent="fly", customer_sentiment="neutral")
        raise AssertionError("Should have raised ValidationError")
    except ValidationError as e:
        print(f"Invalid input correctly rejected: {e.error_count()} error(s)")

    print("All smoke tests passed.")
