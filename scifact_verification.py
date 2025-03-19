#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import os
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


class NLIModel:
    """
    Wraps a HuggingFace NLI model for SciFact.
    Uses MilosKosRad/DeBERTa-v3-large-SciFact by default.
    """
    def __init__(self, model_name: str = "roberta-large-mnli", device: str = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps"
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        # By default for SciFact-based NLI:
        # 0 -> CONTRADICTION, 1 -> NEUTRAL, 2 -> ENTAILMENT
        self.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def predict(self, premise: str, hypothesis: str) -> List[float]:
        """
        Returns raw probabilities (contradiction, neutral, entailment)
        for the premise-hypothesis pair.
        """
        inputs = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        # Move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape (1, 3)
            probs = F.softmax(logits, dim=1).squeeze()  # shape (3,)

        # probs = [P(contradiction), P(neutral), P(entailment)]
        return probs.cpu().tolist()


class SciFactNliPipeline:
    """
    Orchestrates reading SciFact data, caching NLI predictions,
    and computing per-claim label predictions using a differential scoring method.
    """
    def __init__(
        self,
        data_path: str = "selected_scifact_subset.json",  # <-- NOTE: Now points to your subset
        cache_path: str = "nli_cache_subset.json",
        results_path: str = "nli_verification_results_subset.json",
        skip_no_evidence: bool = False,
        cache_save_interval: int = 200,
        score_threshold: float = 0.1  # Threshold for deciding neutrality
    ):
        """
        Args:
            data_path: path to the selected SciFact JSON subset.
            cache_path: path to the cache JSON for NLI inference.
            results_path: final results (predictions+metrics).
            skip_no_evidence: if True, exclude claims with no evidence.
            cache_save_interval: how many (sentence, claim) inferences before saving the cache.
            score_threshold: differential score below which the claim is considered neutral.
        """
        self.data_path = data_path
        self.cache_path = cache_path
        self.results_path = results_path
        self.skip_no_evidence = skip_no_evidence
        self.cache_save_interval = cache_save_interval
        self.score_threshold = score_threshold

        self.nli_model = NLIModel()

        # 1) Load SciFact data (from the subset file)
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.scifact_data = json.load(f)  # list of doc entries

        # 2) Load or initialize the cache
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as cf:
                self.cache = json.load(cf)
        else:
            self.cache = {}  # { "docid||sentidx||claimid": [prob_contradiction, prob_neutral, prob_entailment] }

        self.results = []  # Will store final results at claim-level

    def run_inference(self):
        """
        For each document in scifact_data, for each claim referencing that doc,
        compute NLI probabilities for all (abstract_sentence, claim) pairs if not cached.
        Then, for each claim, select the sentence with the highest differential score and assign a label.
        """
        inference_counter = 0

        # Total number of (doc -> claim -> sentence) pairs
        total_pairs = sum(len(doc["abstract_sents"]) * len(doc.get("claims", []))
                          for doc in self.scifact_data)
        with tqdm(total=total_pairs, desc="NLI Inference") as pbar:
            for doc in self.scifact_data:
                doc_id = str(doc["doc_id"])
                abstract_sents = doc["abstract_sents"]
                claims = doc.get("claims", [])

                for claim_data in claims:
                    claim_id = str(claim_data["id"])
                    claim_text = claim_data["claim"]

                    # Derive the gold label from the "evidence" structure
                    gold_label, has_evidence = self._get_gold_label_for_doc(claim_data, doc_id)

                    # Optionally skip claims that have no evidence
                    if self.skip_no_evidence and (not has_evidence):
                        pbar.update(len(abstract_sents))
                        continue

                    # Compute differential scores for each sentence in the abstract.
                    best_result = self.aggregate_scores(abstract_sents, claim_text)

                    # Save results for metric calculation
                    self.results.append({
                        "doc_id": doc_id,
                        "claim_id": claim_id,
                        "claim_text": claim_text,
                        "pred_label": best_result["label"],
                        "gold_label": gold_label,
                        "matched_sentence": best_result["sentence"],
                        "p_entail": best_result["p_entail"],
                        "p_contr": best_result["p_contr"],
                        "p_neut": best_result["p_neut"],
                        "NLIScore": best_result["NLIScore"]
                    })

                    # Update progress bar for each sentence evaluated
                    pbar.update(len(abstract_sents))
                    inference_counter += len(abstract_sents)
                    if inference_counter % self.cache_save_interval == 0:
                        self._save_cache()

        self._save_cache()

    def _get_gold_label_for_doc(self, claim_data: Dict[str, Any], doc_id: str):
        """
        In SciFact, each claim can have an 'evidence' dict:
            evidence = {
              "<doc_id>": [
                {"sentences": [...], "label": "SUPPORT" or "CONTRADICT"},
                ...
              ]
            }

        We'll map:
           SUPPORT => "entailment"
           CONTRADICT => "contradiction"
        If there's no evidence at all for doc_id => "neutral".

        Returns:
          (gold_label, has_evidence)
            gold_label (str): "entailment", "contradiction", or "neutral"
            has_evidence (bool): True if evidence exists for doc_id.
        """
        ev = claim_data.get("evidence", {})
        if doc_id not in ev or not ev[doc_id]:
            return "neutral", False

        ev_list = ev[doc_id]
        labels_found = {item["label"].upper() for item in ev_list if "label" in item}

        if "SUPPORT" in labels_found:
            return "entailment", True
        elif "CONTRADICT" in labels_found:
            return "contradiction", True
        else:
            return "neutral", True

    def aggregate_scores(self, abstract_sentences: List[str], claim_text: str) -> Dict:
        """
        For a given claim, compute the differential NLIScore for each abstract sentence
        and select the sentence with the highest score.

        NLIScore = p_entail - p_contr
        If |NLIScore| < self.score_threshold, the label is considered neutral.

        Returns:
            A dictionary with the best matching sentence, its NLI probabilities,
            the computed NLIScore, and the final label.
        """
        best_result = {
            "sentence": None,
            "p_entail": 0.0,
            "p_contr": 0.0,
            "p_neut": 0.0,
            "NLIScore": -float("inf"),
            "label": None
        }
        for sent_idx, sent_text in enumerate(abstract_sentences):
            cache_key = f"{sent_text}||{claim_text}"
            if cache_key in self.cache:
                probs = self.cache[cache_key]
            else:
                probs = self.nli_model.predict(sent_text, claim_text)
                self.cache[cache_key] = probs

            # Recall: probs = [P(contradiction), P(neutral), P(entailment)]
            p_contr, p_neut, p_entail = probs
            nli_score = p_entail - p_contr

            # If this sentence has a higher differential score, update best_result.
            if nli_score > best_result["NLIScore"]:
                # Decide label based on score threshold.
                if abs(nli_score) < self.score_threshold:
                    final_label = "neutral"
                elif nli_score > 0:
                    final_label = "entailment"
                else:
                    final_label = "contradiction"
                best_result = {
                    "sentence": sent_text,
                    "p_entail": p_entail,
                    "p_contr": p_contr,
                    "p_neut": p_neut,
                    "NLIScore": nli_score,
                    "label": final_label
                }
        return best_result

    def calculate_metrics(self):
        """
        After run_inference, compute overall accuracy, precision, recall, F1
        for the predicted vs. gold labels in results.
        """
        gold_labels = [r["gold_label"] for r in self.results]
        pred_labels = [r["pred_label"] for r in self.results]

        label_list = ["contradiction", "neutral", "entailment"]
        label_to_idx = {label: i for i, label in enumerate(label_list)}
        gold_int = [label_to_idx.get(lbl, 1) for lbl in gold_labels]  # default to 'neutral' if unknown
        pred_int = [label_to_idx.get(lbl, 1) for lbl in pred_labels]

        acc = accuracy_score(gold_int, pred_int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_int, pred_int, average="macro", labels=[0,1,2]
        )

        metrics = {
            "accuracy": acc,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        }
        return metrics

    def save_results(self, metrics: Dict[str, float]):
        """
        Save the claim-level predictions and the overall metrics into a single JSON file.
        """
        output_dict = {
            "results": self.results,
            "metrics": metrics
        }
        with open(self.results_path, "w", encoding="utf-8") as rf:
            json.dump(output_dict, rf, indent=2)
        print(f"Saved results to {self.results_path}")

    def _save_cache(self):
        """
        Saves the entire cache as one JSON file.
        """
        with open(self.cache_path, "w", encoding="utf-8") as cf:
            json.dump(self.cache, cf, indent=2)
        print(f"Cache updated => {self.cache_path}")


def main():
    # Example usage:
    # data_path => your subset of SciFact abstracts
    # skip_no_evidence => whether to exclude claims with no evidence from the final results

    pipeline = SciFactNliPipeline(
        data_path="selected_scifact_subset.json",     # <--- Your subset file
        cache_path="nli_cache_subset.json",            # <--- Cache for NLI inferences
        results_path="nli_verification_results_subset.json",
        skip_no_evidence=False,
        cache_save_interval=200,                       # <--- Save cache every 200 inferences
        score_threshold=0.1                            # Differential score threshold for neutrality
    )

    # 1) Run the NLI inference (per-claim evaluation)
    pipeline.run_inference()

    # 2) Compute metrics
    metrics = pipeline.calculate_metrics()
    print("Metrics:", metrics)

    # 3) Save results
    pipeline.save_results(metrics)


if __name__ == "__main__":
    main()
