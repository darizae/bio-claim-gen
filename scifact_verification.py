#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import List, Dict, Any
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
    aggregating sentence-level scores, computing metrics, etc.
    """
    def __init__(
        self,
        data_path: str = "selected_scifact_subset.json",  # <-- NOTE: Now points to your subset
        cache_path: str = "nli_cache_subset.json",
        results_path: str = "nli_verification_results_subset.json",
        skip_no_evidence: bool = False,
        cache_save_interval: int = 200
    ):
        """
        Args:
            data_path: path to the selected SciFact JSON subset (default: "selected_scifact_subset.json").
            cache_path: path to the cache JSON for NLI inference (default: "nli_cache_subset.json").
            results_path: final results (predictions+metrics).
            skip_no_evidence: if True, exclude claims with no evidence for doc_id from final results & metrics.
            cache_save_interval: how many (sentence, claim) inferences before saving the cache.
        """
        self.data_path = data_path
        self.cache_path = cache_path
        self.results_path = results_path
        self.skip_no_evidence = skip_no_evidence
        self.cache_save_interval = cache_save_interval

        self.nli_model = NLIModel()

        # 1) Load SciFact data (from the subset file)
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.scifact_data = json.load(f)  # list of doc entries

        # 2) Load or initialize the cache
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as cf:
                self.cache = json.load(cf)
        else:
            self.cache = {}  # { "docid||sentidx||claimid": [prob_contradict, prob_neutral, prob_entailment] }

        self.results = []  # Will store final results at claim-level

    def run_inference(self):
        """
        For each document in scifact_data, for each claim referencing that doc,
        compute NLI probabilities for all (abstract_sentence, claim) pairs if not cached.
        Then aggregate across sentences to derive a final predicted label.
        """
        # Count how many (sentence, claim) inferences we've processed since last save
        inference_counter = 0

        # Prepare a progress bar over the total # of (doc->claim->sentence) combos
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
                        # Don't process or store them => skip
                        # Just update pbar to reflect those sentences
                        pbar.update(len(abstract_sents))
                        continue

                    sentence_probs = []
                    for sent_idx, sent_text in enumerate(abstract_sents):
                        cache_key = f"{doc_id}||{sent_idx}||{claim_id}"
                        if cache_key in self.cache:
                            probs = self.cache[cache_key]
                        else:
                            probs = self.nli_model.predict(sent_text, claim_text)
                            self.cache[cache_key] = probs
                        sentence_probs.append(probs)

                        pbar.update(1)
                        inference_counter += 1

                        # Save cache every N=cache_save_interval inferences
                        if inference_counter % self.cache_save_interval == 0:
                            self._save_cache()

                    # Aggregate scores
                    pred_label, max_scores = self.aggregate_scores(sentence_probs)

                    # Save results for metric calculation
                    self.results.append({
                        "doc_id": doc_id,
                        "claim_id": claim_id,
                        "claim_text": claim_text,
                        "pred_label": pred_label,
                        "gold_label": gold_label,
                        "max_entailment_score": max_scores["entailment"],
                        "max_contradiction_score": max_scores["contradiction"],
                        "max_neutral_score": max_scores["neutral"]
                    })

        # After finishing *all* docs, do a final cache save
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
            has_evidence (bool): True if doc_id had some evidence annotation
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

    def aggregate_scores(self, sentence_probs: List[List[float]]):
        """
        sentence_probs is a list of length N (# sentences),
        each element = [prob_contradiction, prob_neutral, prob_entailment].

        We'll define a simple aggregator that looks at the maximum
        contradiction, neutral, and entailment across all sentences.

        Then pick whichever category has the highest max prob as final label.

        Returns:
           pred_label (str): "contradiction"/"neutral"/"entailment"
           max_scores (dict): e.g. {"contradiction": 0.9, "neutral": 0.5, "entailment": 0.2}
        """
        max_contradiction = 0.0
        max_neutral = 0.0
        max_entailment = 0.0

        for probs in sentence_probs:
            c, n, e = probs
            if c > max_contradiction:
                max_contradiction = c
            if n > max_neutral:
                max_neutral = n
            if e > max_entailment:
                max_entailment = e

        max_dict = {
            "contradiction": max_contradiction,
            "neutral": max_neutral,
            "entailment": max_entailment
        }

        final_label = max(max_dict, key=max_dict.get)
        return final_label, max_dict

    def calculate_metrics(self):
        """
        After run_inference, compute overall accuracy, precision, recall, F1
        for the predicted vs. gold labels in results.

        We'll do a 3-class classification on: "contradiction", "neutral", "entailment"
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
        Save the claim-level predictions and the overall metrics
        into a single JSON file.
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
    # data_path => your newly selected subset of ~100 abstracts
    # skip_no_evidence => whether to exclude claims with no evidence from the final results

    pipeline = SciFactNliPipeline(
        data_path="selected_scifact_subset.json",     # <--- Our subset file
        cache_path="nli_cache_subset.json",           # <--- Separate cache for the subset
        results_path="nli_verification_results_subset.json",
        skip_no_evidence=False,
        cache_save_interval=200                       # <--- Save cache every 200 inferences
    )

    # 1) Run the NLI inference
    pipeline.run_inference()

    # 2) Compute metrics
    metrics = pipeline.calculate_metrics()
    print("Metrics:", metrics)

    # 3) Save results
    pipeline.save_results(metrics)


if __name__ == "__main__":
    main()
