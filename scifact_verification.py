import os
import json
import torch
from collections import defaultdict
from typing import List, Dict
from transformers import pipeline

##############################
#   CONFIG / PATHS
##############################

PREPROCESSED_CLAIMS_PATH = os.path.join("data/scifact/preprocessed_scifact.json")

NLI_MODEL_NAME = "roberta-large-mnli"
MODEL_TO_SFACT = {
    "ENTAILMENT": "SUPPORT",
    "CONTRADICTION": "REFUTE",
    "NEUTRAL": "NOT_MENTIONED"
}


##############################
#   DEVICE CHECK
##############################

def select_device_for_pipeline():
    """
    Check for CUDA, then MPS, else CPU.
    Return an integer or string that huggingface pipeline recognizes:
      - 0 => GPU:0
      - 'mps' => Apple Silicon MPS
      - -1 => CPU
    """
    if torch.cuda.is_available():
        return 0  # means "cuda:0"
    # Some older versions of PyTorch have torch.backends.mps,
    # so we check existence and availability:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return -1  # CPU


##############################
#   HELPERS
##############################

def init_nli_pipeline(model_name=NLI_MODEL_NAME):
    device = select_device_for_pipeline()
    print(f"Using device: {device}")
    return pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device,
        return_all_scores=True
    )


def batch_nli_inference(nli_pipe, pairs: List[str], batch_size=16):
    all_results = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i: i + batch_size]
        results = nli_pipe(batch)
        all_results.extend(results)
    return all_results


def pick_label_across_segments(segment_scores: List[List[Dict]]) -> str:
    max_ent = 0.0
    max_con = 0.0
    max_neu = 0.0

    for label_score_list in segment_scores:
        for d in label_score_list:
            if d["label"] == "ENTAILMENT":
                max_ent = max(max_ent, d["score"])
            elif d["label"] == "CONTRADICTION":
                max_con = max(max_con, d["score"])
            elif d["label"] == "NEUTRAL":
                max_neu = max(max_neu, d["score"])

    if max_ent >= max_con and max_ent >= max_neu:
        return "ENTAILMENT"
    elif max_con >= max_ent and max_con >= max_neu:
        return "CONTRADICTION"
    else:
        return "NEUTRAL"


def get_gold_label_for_claim(claim_dict: Dict, doc_id: str) -> str:
    evidence = claim_dict.get("evidence", {})
    if doc_id not in evidence:
        return "NOT_MENTIONED"
    ev_list = evidence[doc_id]
    return ev_list[0]["label"]


##############################
#   MAIN VERIFICATION
##############################

def main():
    if not os.path.exists(PREPROCESSED_CLAIMS_PATH):
        raise FileNotFoundError(
            f"Preprocessed cache not found at {PREPROCESSED_CLAIMS_PATH}. "
            f"Run scifact_preprocessing.py first."
        )

    with open(PREPROCESSED_CLAIMS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # For quick tests, let's limit the number of claims
    max_claims = 10

    nli_pipe = init_nli_pipeline()

    gold_labels = []
    predicted_labels = []
    count = 0

    for entry in data:
        doc_id_str = entry["doc_id"]
        abstract_sents = entry["abstract_sents"]
        abstract_raw = entry["abstract_raw"]  # if you want to use it for something else
        claim_objs = entry["claims"]

        for claim_dict in claim_objs:
            if count >= max_claims:
                break

            claim_text = claim_dict["claim"]
            gold_label = get_gold_label_for_claim(claim_dict, doc_id_str)

            # For the matrix approach, we typically do (each sentence × claim).
            # If you want to use abstract_raw, you could do a single premise with the entire abstract,
            # but let's keep the segment approach here:
            nli_inputs = [
                f"premise: {seg}\nhypothesis: {claim_text}"
                for seg in abstract_sents
            ]

            all_nli_scores = batch_nli_inference(nli_pipe, nli_inputs, batch_size=8)
            final_label = pick_label_across_segments(all_nli_scores)
            mapped_label = MODEL_TO_SFACT[final_label]

            gold_labels.append(gold_label)
            predicted_labels.append(mapped_label)
            count += 1

            print(f"[{count}] Doc {doc_id_str}, Claim: {claim_text}")
            print(f"    Gold: {gold_label},  Pred: {mapped_label}\n")

        if count >= max_claims:
            break

    total = len(gold_labels)
    correct = sum(1 for g, p in zip(gold_labels, predicted_labels) if g == p)
    accuracy = correct / total if total else 0
    print(f"\nProcessed {total} claims. Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
