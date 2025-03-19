import os
import json
from pathlib import Path
from typing import List
from openai import OpenAI

# Load API key from environment.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = "gpt-3.5-turbo"

# File paths.
INPUT_JSON = "../data/selected_scifact_subset_with_claims.json"
OUTPUT_JSON = "../data/selected_scifact_subset_with_coref.json"

# Few-shot example and chain-of-thought instructions for coreference resolution.
FEW_SHOT_EXAMPLE = """
Example:
Abstract: "In this study, researchers examined the effects of a new drug. The drug significantly reduced symptoms, and its benefits were confirmed in a follow-up analysis."
Claims:
1. "The new drug reduces symptoms."
2. "It has confirmed benefits."
Chain-of-Thought: The pronoun "It" in claim 2 refers to "the new drug" mentioned in the abstract. Therefore, the resolved claims should explicitly reference "the new drug."
Resolved Claims:
1. "The new drug reduces symptoms."
2. "The new drug has confirmed benefits."
"""


def build_coref_prompt(abstract: str, claims: List[str]) -> str:
    """
    Build a prompt for GPT-3.5-turbo that instructs the model to perform
    coreference resolution on a group of claims, using the abstract as context.
    """
    prompt = (
        "You are an expert in natural language understanding and natural language inference. "
        "Your task is to perform coreference resolution on a set of claims in the context of a given abstract. "
        "Use a chain-of-thought approach to identify ambiguous references and then provide the final resolved claims as a JSON array. "
        "Follow the detailed instructions and the few-shot example provided.\n\n"
        f"Few-shot Example:\n{FEW_SHOT_EXAMPLE}\n"
        "Now, given the following input:\n"
        f"Abstract: \"{abstract}\"\n"
        "Claims:\n"
    )
    # Format the list of claims as numbered items.
    for i, claim in enumerate(claims, start=1):
        prompt += f"{i}. \"{claim}\"\n"
    prompt += (
        "\nPlease perform coreference resolution and output the resolved claims as a JSON array of strings."
    )
    return prompt


def resolve_coreferences(abstract: str, claims: List[str]) -> List[str]:
    """
    Given an abstract and a list of claims, call the OpenAI Chat API with a detailed prompt
    to perform coreference resolution. Returns a list of resolved claim strings.
    """
    prompt = build_coref_prompt(abstract, claims)
    messages = [
        {"role": "system", "content": "You are an expert in natural language inference and text reasoning."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(model=MODEL_NAME,
                                                  messages=messages,
                                                  temperature=0.3,
                                                  max_tokens=512)
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return claims  # Fallback: return original claims

    resolved_text = response.choices[0].message.content.strip()
    try:
        # Expecting the output to be a JSON array of strings.
        resolved_claims = json.loads(resolved_text)
        if isinstance(resolved_claims, list):
            return resolved_claims
        else:
            print("Resolved output is not a list. Returning original claims.")
            return claims
    except Exception as e:
        print(f"Error parsing resolved claims: {e}")
        # As fallback, try to split by newlines (this is less robust)
        fallback_claims = [line.strip() for line in resolved_text.split("\n") if line.strip()]
        return fallback_claims if fallback_claims else claims


def get_abstract(entry: dict) -> str:
    """
    Retrieve the abstract text from an entry. Prefer "abstract_raw" if available,
    otherwise join the "abstract_sents" list.
    """
    abstract = entry.get("abstract_raw", "").strip()
    if not abstract and "abstract_sents" in entry:
        abstract = " ".join(entry.get("abstract_sents", [])).strip()
    return abstract


def process_entry(entry: dict) -> dict:
    """
    Process a single entry by applying coreference resolution to:
      - Original claims (if available under "original_claims")
      - Generated claims (for each strategy in the "claims" dictionary)
    Updates the entry in-place with the resolved claims.
    """
    abstract = get_abstract(entry)
    if not abstract:
        print(f"Entry with doc_id {entry.get('doc_id')} has no abstract. Skipping coreference resolution.")
        return entry

    # Process original claims if available.
    if "original_claims" in entry:
        original_claims = entry["original_claims"]
        if isinstance(original_claims, list) and original_claims:
            print(f"Processing coreference resolution for original_claims of doc_id {entry.get('doc_id')}.")
            resolved_original = resolve_coreferences(abstract, original_claims)
            entry["original_claims"] = resolved_original

    # Process generated claims grouped by strategy.
    if "claims" in entry and isinstance(entry["claims"], dict):
        for strategy, claim_list in entry["claims"].items():
            if isinstance(claim_list, list) and claim_list:
                print(f"Processing coreference resolution for strategy '{strategy}' in doc_id {entry.get('doc_id')}.")
                resolved_generated = resolve_coreferences(abstract, claim_list)
                entry["claims"][strategy] = resolved_generated

    return entry


def main():
    input_path = Path(INPUT_JSON)
    output_path = Path(OUTPUT_JSON)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {input_path}")

    processed_entries = []
    for entry in data:
        processed_entry = process_entry(entry)
        processed_entries.append(processed_entry)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(processed_entries, f, indent=2, ensure_ascii=False)
    print(f"Coreference resolution complete. Processed data saved to {output_path}")


if __name__ == "__main__":
    main()
