#!/usr/bin/env python3
"""
NLI Verification System for SciFact-like Data
---------------------------------------------
This script loads a preprocessed JSON file (e.g., 'preprocessed_scifact.json'),
and runs a SciFact-focused DeBERTa-v3-large model to predict NLI labels
('ENTAILMENT', 'NEUTRAL', 'CONTRADICTION') for each (abstract sentence, claim) pair.

It then aggregates these pairwise results per claim, storing:
 - predicted label (based on an aggregation strategy)
 - maximum/average probability
 - a fallback for claims without a known SciFact label (treated as "hallucinated"
   or new claims).

Caching is used to avoid recomputing NLI scores for the same premise–claim pairs.
"""

import json
import os
import hashlib
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BatchEncoding

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_NAME = "MilosKosRad/DeBERTa-v3-large-SciFact"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

# You can adjust this threshold or approach to suit your aggregator’s logic
ENTAILMENT_THRESHOLD = 0.5

# Path to your data and any caching
DATA_PATH = "data/scifact/preprocessed_scifact.json"
CACHE_DIR = "nli_cache"  # folder where we store results to avoid re-inference

# Make sure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. Model & Caching Classes
# ---------------------------------------------------------
class SciFactNLIModel:
    """
    A wrapper around the DeBERTa v3 large SciFact model, providing:
    - Tokenization
    - NLI label inference
    - In-memory + on-disk caching to avoid repeated computation
    """

    def __init__(self, model_name: str = MODEL_NAME, device: torch.device = DEVICE):
        self.model_name = model_name
        self.device = device

        print(f"Loading tokenizer and model from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # In-memory cache: dict keyed by (premise, hypothesis) -> result
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

    def _make_key(self, premise: str, hypothesis: str) -> str:
        """
        Create a unique cache key based on the premise and hypothesis.
        Could be hashed if they are too large.
        """
        raw_key = premise + "\n||\n" + hypothesis
        # Use a short hash so we don't have huge filenames
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    def predict_nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Predict NLI probabilities for a single premise–hypothesis pair.
        Returns a dict with the label probabilities:
          {"ENTAILMENT": prob_e, "NEUTRAL": prob_n, "CONTRADICTION": prob_c}
        """

        # 1) Check in-memory cache
        cache_key = self._make_key(premise, hypothesis)
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        # 2) Check disk cache
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # Also store in memory_cache
            self.memory_cache[cache_key] = cached
            return cached

        # 3) If not cached, run model inference
        inputs: BatchEncoding = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=256  # Adjust if needed
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()[0]

        # The model is typically [contradiction, neutral, entailment]
        # but we need to verify the label mapping for "MilosKosRad/DeBERTa-v3-large-SciFact".
        # Usually for SciFact, the order is [CONTRADICTION, NEUTRAL, ENTAILMENT].
        # Let's confirm that is indeed the case. We'll map as:
        label_probs = {
            "CONTRADICTION": probs[0],
            "NEUTRAL": probs[1],
            "ENTAILMENT": probs[2]
        }

        # Save to both memory and disk
        self.memory_cache[cache_key] = label_probs
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(label_probs, f)

        return label_probs


# ---------------------------------------------------------
# 2. Aggregator
# ---------------------------------------------------------
def aggregate_nli_scores(sent_label_probs: List[Dict[str, float]],
                         strategy: str = "max_entailment") -> str:
    """
    Given a list of label-probability dicts (one per sentence),
    derive a final label for the claim. Each dict looks like:
       {"ENTAILMENT": 0.7, "NEUTRAL": 0.2, "CONTRADICTION": 0.1}
    Strategy Examples:
      - "max_entailment": pick the max ENT score across all sentences
        to decide if overall ENT > threshold => "ENTAILMENT"
      - "best_label_across_sentences": pick whichever label has the
        highest probability among all sentences
      - "average_entailment": average ENT across all sentences and
        see if it’s above threshold
      - ... etc.
    """

    if not sent_label_probs:
        # If no sentences or no data, default to "NOINFO"
        return "NOINFO"

    if strategy == "max_entailment":
        # Check the highest entailment probability across all sentences
        max_ent = max(lp["ENTAILMENT"] for lp in sent_label_probs)
        if max_ent >= ENTAILMENT_THRESHOLD:
            return "ENTAILMENT"
        else:
            # If it's not strongly entailed, we can check if there's any strong contradiction
            max_contra = max(lp["CONTRADICTION"] for lp in sent_label_probs)
            if max_contra > max_ent:
                return "CONTRADICTION"
            else:
                return "NEUTRAL"

    elif strategy == "best_label_across_sentences":
        # For each sentence, pick the label with the highest probability,
        # then see which label is the most frequent
        all_labels = []
        for lp in sent_label_probs:
            best_label = max(lp.keys(), key=lambda k: lp[k])
            all_labels.append(best_label)
        # pick the majority
        # or you can do: return statistics.mode(all_labels)
        # For simplicity, pick the majority label
        label_counts = {
            "ENTAILMENT": all_labels.count("ENTAILMENT"),
            "NEUTRAL": all_labels.count("NEUTRAL"),
            "CONTRADICTION": all_labels.count("CONTRADICTION"),
        }
        return max(label_counts, key=label_counts.get)

    elif strategy == "average_entailment":
        # Average the ENT score across all sentences
        avg_ent = sum(lp["ENTAILMENT"] for lp in sent_label_probs) / len(sent_label_probs)
        if avg_ent >= ENTAILMENT_THRESHOLD:
            return "ENTAILMENT"
        else:
            # could pick NEUTRAL or CONTRADICTION based on average
            avg_contra = sum(lp["CONTRADICTION"] for lp in sent_label_probs) / len(sent_label_probs)
            if avg_contra > avg_ent:
                return "CONTRADICTION"
            else:
                return "NEUTRAL"

    else:
        # Default fallback
        return "NOINFO"


# ---------------------------------------------------------
# 3. Main Processing Function
# ---------------------------------------------------------
class SciFactVerifier:
    """
    Orchestrates:
     1) Loading the data
     2) Building the N x M matrix of sentence–claim pairs
     3) Using the NLI model to compute label probabilities
     4) Aggregating results for each claim
     5) Handling unknown-labeled (hallucinated) claims
    """

    def __init__(self, model: SciFactNLIModel, aggregator_strategy: str = "max_entailment"):
        self.model = model
        self.aggregator_strategy = aggregator_strategy

    def verify_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single document from SciFact-like JSON structure.
        doc has fields:
          'abstract_sents': list of sentences
          'claims': list of claim objects
        Returns the updated doc with `predicted_label` for each claim.
        """
        abstract_sents = doc.get("abstract_sents", [])
        claims = doc.get("claims", [])

        for claim_obj in claims:
            claim_text = claim_obj["claim"]

            # (A) For each sentence, run NLI
            label_probs_list = []
            for sent in abstract_sents:
                lp = self.model.predict_nli(premise=sent, hypothesis=claim_text)
                label_probs_list.append(lp)

            # (B) Aggregate label
            aggregated_label = aggregate_nli_scores(label_probs_list, self.aggregator_strategy)
            claim_obj["predicted_label"] = aggregated_label

            # (C) If the original data has no label or evidence,
            #     we can treat it as newly generated/hallucinated
            #     but still keep the predicted label from NLI.
            #     For illustration, let's store a new key "is_hallucinated"
            #     if there's truly no ground-truth label:
            is_hallucinated = False
            # For SciFact, sometimes claims have "evidence" key or "label" in the official dataset
            # If it's empty or not present, interpret as "no official label".
            # We'll define a simple heuristic:
            if not claim_obj.get("evidence"):
                is_hallucinated = True

            claim_obj["is_hallucinated"] = is_hallucinated

        return doc

    def verify_corpus(self, corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify all documents in the corpus. Each doc is updated with
        predicted labels for each claim.
        """
        verified_docs = []
        for doc in corpus:
            verified_doc = self.verify_document(doc)
            verified_docs.append(verified_doc)
        return verified_docs


# ---------------------------------------------------------
# 4. Main Script Entry Point
# ---------------------------------------------------------
def main():
    # 1) Load the data
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        scifact_data = json.load(f)

    # scifact_data is expected to be a list of documents, each doc as a dict
    # or a single large list. Adapt if your JSON is structured differently.

    # 2) Initialize model & verifier
    nli_model = SciFactNLIModel(MODEL_NAME, DEVICE)
    verifier = SciFactVerifier(nli_model, aggregator_strategy="max_entailment")

    # 3) Run verification
    verified_results = verifier.verify_corpus(scifact_data)

    # 4) Save or print results
    out_path = "verified_scifact_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verified_results, f, indent=2)
    print(f"Verification results saved to {out_path}")


if __name__ == "__main__":
    main()
