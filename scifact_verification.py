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
from tqdm import tqdm


class NLIModel:
    """
    Wraps a HuggingFace NLI model for SciFact.
    Uses MilosKosRad/DeBERTa-v3-large-SciFact by default.
    """
    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", device: str = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps"
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        # For SciFact-based NLI: 0->CONTRADICTION, 1->NEUTRAL, 2->ENTAILMENT
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
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape (1, 3)
            probs = F.softmax(logits, dim=1).squeeze()  # shape (3,)
        return probs.cpu().tolist()


def split_into_paragraphs(sentences: List[str], num_sent_per_paragraph: int = 5, sliding: bool = True, stride: int = 1) -> List[Dict[str, Any]]:
    """
    Groups a list of sentences into paragraphs.
    Returns a list of dicts with keys:
      - "text": the concatenated paragraph text.
      - "indices": list of sentence indices that form the paragraph.
      - "granularity": "paragraph".
    """
    paragraphs = []
    if sliding:
        for i in range(0, len(sentences) - num_sent_per_paragraph + 1, stride):
            para_text = " ".join(sentences[i:i + num_sent_per_paragraph])
            paragraphs.append({
                "text": para_text,
                "indices": list(range(i, i + num_sent_per_paragraph)),
                "granularity": "paragraph"
            })
    else:
        for i in range(0, len(sentences), num_sent_per_paragraph):
            para_text = " ".join(sentences[i:i + num_sent_per_paragraph])
            paragraphs.append({
                "text": para_text,
                "indices": list(range(i, min(i + num_sent_per_paragraph, len(sentences)))),
                "granularity": "paragraph"
            })
    return paragraphs


class SciFactNliPipeline:
    """
    Reads SciFact data, computes per-claim predictions using multi-granular spans,
    outputs predictions in leaderboard format, and computes matching metrics against ground truths.
    """
    def __init__(
        self,
        data_path: str = "preprocessed_scifact.json",
        cache_path: str = "nli_cache_full.json",
        results_path: str = "scifact_predictions_full.jsonl",
        skip_no_evidence: bool = False,
        cache_save_interval: int = 200,
        score_threshold: float = 0.1,  # Differential score threshold for neutrality
        num_sent_per_paragraph: int = 5,
        paragraph_sliding: bool = True,
        paragraph_stride: int = 1
    ):
        self.data_path = data_path
        self.cache_path = cache_path
        self.results_path = results_path
        self.skip_no_evidence = skip_no_evidence
        self.cache_save_interval = cache_save_interval
        self.score_threshold = score_threshold
        self.num_sent_per_paragraph = num_sent_per_paragraph
        self.paragraph_sliding = paragraph_sliding
        self.paragraph_stride = paragraph_stride

        self.nli_model = NLIModel()

        # Load SciFact data.
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.scifact_data = json.load(f)

        # Load or initialize the cache.
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as cf:
                self.cache = json.load(cf)
        else:
            self.cache = {}  # Cache key: "span_text||claim_text" -> [P(contradiction), P(neutral), P(entailment)]

        self.results = []  # Intermediate predictions per claim.

    def run_inference(self):
        """
        For each document and its claims, evaluate all spans (sentences and paragraphs).
        For each claim, select the span with the highest differential score (p_entail - p_contr) and assign a predicted label.
        """
        inference_counter = 0
        total_pairs = 0
        for doc in self.scifact_data:
            abstract_sents = doc["abstract_sents"]
            paragraphs = split_into_paragraphs(abstract_sents, self.num_sent_per_paragraph,
                                                self.paragraph_sliding, self.paragraph_stride)
            spans_count = len(abstract_sents) + len(paragraphs)
            total_pairs += spans_count * len(doc.get("claims", []))

        with tqdm(total=total_pairs, desc="NLI Inference") as pbar:
            for doc in self.scifact_data:
                doc_id = str(doc["doc_id"])
                abstract_sents = doc["abstract_sents"]
                # Sentence-level spans.
                sentence_spans = [{
                    "text": sent,
                    "indices": [idx],
                    "granularity": "sentence"
                } for idx, sent in enumerate(abstract_sents)]
                # Paragraph-level spans.
                paragraph_spans = split_into_paragraphs(abstract_sents, self.num_sent_per_paragraph,
                                                        self.paragraph_sliding, self.paragraph_stride)
                spans = sentence_spans + paragraph_spans

                claims = doc.get("claims", [])
                for claim_data in claims:
                    claim_id = int(claim_data["id"])  # Ensure it's an int.
                    claim_text = claim_data["claim"]

                    # Get gold label (for internal evaluation; not used for submission).
                    gold_label, has_evidence = self._get_gold_label_for_doc(claim_data, doc_id)
                    if self.skip_no_evidence and (not has_evidence):
                        pbar.update(len(spans))
                        continue

                    best_result = self.aggregate_scores(spans, claim_text)
                    # Map internal label to leaderboard label.
                    if best_result["label"] == "entailment":
                        pred_label = "SUPPORT"
                    elif best_result["label"] == "contradiction":
                        pred_label = "REFUTES"
                    else:
                        pred_label = "NOINFO"

                    self.results.append({
                        "doc_id": doc_id,
                        "claim_id": claim_id,
                        "claim_text": claim_text,
                        "pred_label": pred_label,
                        "matched_span_indices": best_result["indices"],
                        "p_entail": best_result["p_entail"],
                        "p_contr": best_result["p_contr"],
                        "p_neut": best_result["p_neut"],
                        "NLIScore": best_result["NLIScore"]
                    })

                    pbar.update(len(spans))
                    inference_counter += len(spans)
                    if inference_counter % self.cache_save_interval == 0:
                        self._save_cache()
        self._save_cache()

    def _get_gold_label_for_doc(self, claim_data: Dict[str, Any], doc_id: str):
        """
        Maps SciFact evidence to gold label:
           SUPPORT -> "entailment"
           CONTRADICT -> "contradiction"
           No evidence -> "neutral"
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

    def aggregate_scores(self, spans: List[Dict[str, Any]], claim_text: str) -> Dict:
        """
        For a given claim, compute the differential NLIScore (p_entail - p_contr) for each span.
        Select the span with the highest score and return its associated sentence indices and scores.
        """
        best_result = {
            "indices": None,
            "p_entail": 0.0,
            "p_contr": 0.0,
            "p_neut": 0.0,
            "NLIScore": -float("inf"),
            "label": None
        }
        for span in spans:
            span_text = span["text"]
            cache_key = f"{span_text}||{claim_text}"
            if cache_key in self.cache:
                probs = self.cache[cache_key]
            else:
                probs = self.nli_model.predict(span_text, claim_text)
                self.cache[cache_key] = probs

            # probs = [P(contradiction), P(neutral), P(entailment)]
            p_contr, p_neut, p_entail = probs
            nli_score = p_entail - p_contr

            if nli_score > best_result["NLIScore"]:
                if abs(nli_score) < self.score_threshold:
                    final_label = "neutral"
                elif nli_score > 0:
                    final_label = "entailment"
                else:
                    final_label = "contradiction"
                best_result = {
                    "indices": span["indices"],
                    "p_entail": p_entail,
                    "p_contr": p_contr,
                    "p_neut": p_neut,
                    "NLIScore": nli_score,
                    "label": final_label
                }
        return best_result

    def save_leaderboard_predictions(self):
        """
        Write predictions to a JSONL file.
        Each line corresponds to one claim, formatted as:
        {
          "id": <claim_id as int>,
          "evidence": {
              "<doc_id>": {
                  "sentences": [ list of sentence indices ],
                  "label": <"SUPPORT", "REFUTES", or "NOINFO">
              }
          }
        }
        """
        with open(self.results_path, "w", encoding="utf-8") as fout:
            for result in self.results:
                prediction = {
                    "id": result["claim_id"],
                    "evidence": {
                        result["doc_id"]: {
                            "sentences": result["matched_span_indices"],
                            "label": result["pred_label"]
                        }
                    }
                }
                fout.write(json.dumps(prediction) + "\n")
        print(f"Saved leaderboard predictions to {self.results_path}")

    def compute_matching_metrics(self) -> Dict[str, float]:
        """
        Compares predictions in self.results against ground truth evidence in the loaded SciFact data.
        Computes:
          - Label accuracy.
          - Sentence-level precision, recall, and F1.
        Assumes each claim's ground truth evidence for a document is stored under claim_data["evidence"][doc_id].
        """
        total_claims = 0
        correct_labels = 0
        precision_list = []
        recall_list = []
        f1_list = []

        # Create a mapping of (doc_id, claim_id) -> prediction.
        pred_dict = {}
        for pred in self.results:
            pred_dict[(pred["doc_id"], pred["claim_id"])] = pred

        for doc in self.scifact_data:
            doc_id = str(doc["doc_id"])
            for claim in doc.get("claims", []):
                claim_id = int(claim["id"])
                if (doc_id, claim_id) not in pred_dict:
                    continue
                prediction = pred_dict[(doc_id, claim_id)]
                total_claims += 1

                # Extract ground truth evidence for this doc_id.
                gt_evidence = claim.get("evidence", {}).get(doc_id, [])
                if not gt_evidence:
                    gt_label = "NOINFO"
                    gt_sentences = set()
                else:
                    # Assume all evidence entries share the same label.
                    first_label = gt_evidence[0]["label"].upper()
                    if first_label == "SUPPORT":
                        gt_label = "SUPPORT"
                    elif first_label == "CONTRADICT":
                        gt_label = "REFUTES"
                    else:
                        gt_label = "NOINFO"
                    gt_sentences = set()
                    for entry in gt_evidence:
                        gt_sentences.update(entry.get("sentences", []))

                pred_label = prediction["pred_label"]
                if pred_label == gt_label:
                    correct_labels += 1

                pred_sentences = set(prediction["matched_span_indices"])
                if len(pred_sentences) > 0:
                    precision = len(pred_sentences & gt_sentences) / len(pred_sentences)
                else:
                    precision = 0.0
                if len(gt_sentences) > 0:
                    recall = len(pred_sentences & gt_sentences) / len(gt_sentences)
                else:
                    recall = 0.0
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0

                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

        label_accuracy = correct_labels / total_claims if total_claims > 0 else 0.0
        avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0
        avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
        avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

        metrics = {
            "label_accuracy": label_accuracy,
            "sentence_precision": avg_precision,
            "sentence_recall": avg_recall,
            "sentence_f1": avg_f1
        }
        return metrics

    def _save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as cf:
            json.dump(self.cache, cf, indent=2)
        print(f"Cache updated => {self.cache_path}")


def main():
    pipeline = SciFactNliPipeline(
        #data_path="selected_scifact_subset.json",
       # cache_path="nli_cache_subset.json",
        #results_path="scifact_leaderboard_predictions.jsonl",
        skip_no_evidence=False,
        cache_save_interval=200,
        score_threshold=0.1,
        num_sent_per_paragraph=5,
        paragraph_sliding=True,
        paragraph_stride=1
    )

    # 1) Run inference (per-claim evaluation with multi-granular spans)
    pipeline.run_inference()
    # 2) Save leaderboard-formatted predictions
    pipeline.save_leaderboard_predictions()
    # 3) Compute matching metrics against ground truth.
    metrics = pipeline.compute_matching_metrics()
    print("Matching Metrics:", metrics)


if __name__ == "__main__":
    main()
