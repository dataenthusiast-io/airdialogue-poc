import json
import os
import time

from openai import OpenAI, RateLimitError
from tqdm import tqdm

from src.config import config
from src.schemas import IntentClassification


def _base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(rel_path: str) -> str:
    return os.path.join(_base_dir(), rel_path)


def _load_sample(sample_path: str) -> list[dict]:
    rows = []
    with open(sample_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_processed_ids(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    processed = set()
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                processed.add(json.loads(line)["id"])
    return processed


def _call_api(client: OpenAI, system_prompt: str, dialogue: str) -> tuple[IntentClassification, int, int]:
    response = client.beta.chat.completions.parse(
        model=config.model.name,
        temperature=config.model.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": dialogue},
        ],
        response_format=IntentClassification,
    )
    parsed = response.choices[0].message.parsed
    return parsed, response.usage.prompt_tokens, response.usage.completion_tokens


def _compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        (prompt_tokens / 1000 * config.costs.input_per_1k_tokens)
        + (completion_tokens / 1000 * config.costs.output_per_1k_tokens)
    )


def run_pipeline(
    sample_path: str,
    system_prompt: str,
    output_path: str,
) -> None:
    client = OpenAI()

    sample = _load_sample(sample_path)
    processed_ids = _load_processed_ids(output_path)

    pending = [row for row in sample if row["id"] not in processed_ids]
    if processed_ids:
        print(f"  Resuming: {len(processed_ids)} already done, {len(pending)} remaining")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_calls = 0
    schema_failures = 0
    total_cost = 0.0

    with open(output_path, "a") as out_f:
        for row in tqdm(pending, desc="classifying"):
            error_msg = None
            predicted_intent = None
            customer_sentiment = None
            schema_valid = False
            prompt_tokens = 0
            completion_tokens = 0

            try:
                parsed, prompt_tokens, completion_tokens = _call_api(client, system_prompt, row["dialogue"])
                predicted_intent = parsed.predicted_intent
                customer_sentiment = parsed.customer_sentiment
                schema_valid = True
            except RateLimitError:
                time.sleep(30)
                try:
                    parsed, prompt_tokens, completion_tokens = _call_api(client, system_prompt, row["dialogue"])
                    predicted_intent = parsed.predicted_intent
                    customer_sentiment = parsed.customer_sentiment
                    schema_valid = True
                except Exception as e:
                    error_msg = str(e)
                    schema_failures += 1
            except Exception as e:
                error_msg = str(e)
                schema_failures += 1

            cost = _compute_cost(prompt_tokens, completion_tokens)
            total_cost += cost
            total_calls += 1

            result_row = {
                "id": row["id"],
                "predicted_intent": predicted_intent,
                "customer_sentiment": customer_sentiment,
                "schema_valid": schema_valid,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": round(cost, 8),
                "error": error_msg,
            }
            out_f.write(json.dumps(result_row) + "\n")
            out_f.flush()

            time.sleep(0.5)

    print(f"\n  Pipeline complete:")
    print(f"    Total calls:      {total_calls}")
    print(f"    Schema failures:  {schema_failures}")
    print(f"    Total cost:       ${total_cost:.6f}")
