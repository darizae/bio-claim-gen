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


def split_into_paragraphs(sentences: List[str], num_sent_per_paragraph: int = 5, sliding: bool = True, stride: int = 1) -> List[str]:
    """
    Groups a list of sentences into paragraphs.
    If sliding is True, use a sliding window with the given stride;
    otherwise, group sentences sequentially without overlap.
    """
    paragraphs = []
    if sliding:
        for i in range(0, len(sentences) - num_sent_per_paragraph + 1, stride):
            paragraphs.append(" ".join(sentences[i:i + num_sent_per_paragraph]))
    else:
        for i in range(0, len(sentences), num_sent_per_paragraph):
            paragraphs.append(" ".join(sentences[i:i + num_sent_per_paragraph]))
    return paragraphs


class SciFactNliPipeline:
    """
    Orchestrates reading SciFact data, caching NLI predictions,
    and computing per-claim label predictions using multi-granular spans.
    """
    def __init__(
        self,
        data_path: str = "preprocessed_scifact.json",
        cache_path: str = "nli_cache_full.json",
        results_path: str = "nli_verification_results_full.json",
        skip_no_evidence: bool = False,
        cache_save_interval: int = 200,
        score_threshold: float = 0.1,  # Differential score threshold for neutrality
        num_sent_per_paragraph: int = 5,  # Number of sentences to group as a paragraph
        paragraph_sliding: bool = True,
        paragraph_stride: int = 1
    ):
        """
        Args:
            data_path: path to the SciFact JSON subset.
            cache_path: path to the cache JSON for NLI inference.
            results_path: final results (predictions+metrics).
            skip_no_evidence: if True, exclude claims with no evidence.
            cache_save_interval: how many (span, claim) inferences before saving the cache.
            score_threshold: differential score below which the claim is considered neutral.
            num_sent_per_paragraph: number of sentences per paragraph for multi-granular evaluation.
            paragraph_sliding: whether to use sliding windows when grouping sentences.
            paragraph_stride: sliding stride for paragraph formation.
        """
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

        # 1) Load SciFact data
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.scifact_data = json.load(f)  # list of document entries

        # 2) Load or initialize the cache.
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as cf:
                self.cache = json.load(cf)
        else:
            self.cache = {}  # Cache format: { "span_text||claim_text": [prob_contradiction, prob_neutral, prob_entailment] }

        self.results = []  # Will store final results at claim-level

    def run_inference(self):
        """
        For each document in scifact_data, for each claim referencing that doc,
        compute NLI probabilities for each abstract span (both sentence and paragraph level)
        if not cached. Then select the span with the highest differential score and assign a label.
        """
        inference_counter = 0

        # Total number of (doc -> claim -> span) pairs (upper bound)
        total_pairs = 0
        for doc in self.scifact_data:
            abstract_sents = doc["abstract_sents"]
            paragraphs = split_into_paragraphs(abstract_sents, self.num_sent_per_paragraph, self.paragraph_sliding, self.paragraph_stride)
            total_pairs += (len(abstract_sents) + len(paragraphs)) * len(doc.get("claims", []))

        with tqdm(total=total_pairs, desc="NLI Inference") as pbar:
            for doc in self.scifact_data:
                doc_id = str(doc["doc_id"])
                abstract_sents = doc["abstract_sents"]
                paragraphs = split_into_paragraphs(abstract_sents, self.num_sent_per_paragraph, self.paragraph_sliding, self.paragraph_stride)
                # Combine spans: both individual sentences and paragraphs.
                spans = [{"text": sent, "granularity": "sentence"} for sent in abstract_sents] + \
                        [{"text": para, "granularity": "paragraph"} for para in paragraphs]
                claims = doc.get("claims", [])

                for claim_data in claims:
                    claim_id = str(claim_data["id"])
                    claim_text = claim_data["claim"]

                    # Derive gold label
                    gold_label, has_evidence = self._get_gold_label_for_doc(claim_data, doc_id)
                    if self.skip_no_evidence and (not has_evidence):
                        pbar.update(len(spans))
                        continue

                    best_result = self.aggregate_scores(spans, claim_text)

                    # Save result for metrics and analysis.
                    self.results.append({
                        "doc_id": doc_id,
                        "claim_id": claim_id,
                        "claim_text": claim_text,
                        "pred_label": best_result["label"],
                        "gold_label": gold_label,
                        "matched_span": best_result["span"],
                        "span_granularity": best_result["granularity"],
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
        Map SciFact evidence to gold label:
           SUPPORT => "entailment"
           CONTRADICT => "contradiction"
           No evidence => "neutral"
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

    def aggregate_scores(self, spans: List[Dict[str, str]], claim_text: str) -> Dict:
        """
        For a given claim, compute the differential NLIScore (p_entail - p_contr)
        for each span (sentence or paragraph) and select the span with the highest score.
        If |NLIScore| < score_threshold, label as neutral.
        """
        best_result = {
            "span": None,
            "granularity": None,
            "p_entail": 0.0,
            "p_contr": 0.0,
            "p_neut": 0.0,
            "NLIScore": -float("inf"),
            "label": None
        }
        for span_info in spans:
            span_text = span_info["text"]
            granularity = span_info["granularity"]
            cache_key = f"{span_text}||{claim_text}"
            if cache_key in self.cache:
                probs = self.cache[cache_key]
            else:
                probs = self.nli_model.predict(span_text, claim_text)
                self.cache[cache_key] = probs

            # Recall: probs = [P(contradiction), P(neutral), P(entailment)]
            p_contr, p_neut, p_entail = probs
            nli_score = p_entail - p_contr

            # Check if this span is the best match so far.
            if nli_score > best_result["NLIScore"]:
                if abs(nli_score) < self.score_threshold:
                    final_label = "neutral"
                elif nli_score > 0:
                    final_label = "entailment"
                else:
                    final_label = "contradiction"
                best_result = {
                    "span": span_text,
                    "granularity": granularity,
                    "p_entail": p_entail,
                    "p_contr": p_contr,
                    "p_neut": p_neut,
                    "NLIScore": nli_score,
                    "label": final_label
                }
        return best_result

    def calculate_metrics(self):
        """
        Compute overall accuracy, precision, recall, F1 for the predicted vs. gold labels.
        """
        gold_labels = [r["gold_label"] for r in self.results]
        pred_labels = [r["pred_label"] for r in self.results]

        label_list = ["contradiction", "neutral", "entailment"]
        label_to_idx = {label: i for i, label in enumerate(label_list)}
        gold_int = [label_to_idx.get(lbl, 1) for lbl in gold_labels]
        pred_int = [label_to_idx.get(lbl, 1) for lbl in pred_labels]

        acc = accuracy_score(gold_int, pred_int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_int, pred_int, average="macro", labels=[0, 1, 2]
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
        Save the claim-level predictions and overall metrics to a JSON file.
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
        Saves the entire cache to a JSON file.
        """
        with open(self.cache_path, "w", encoding="utf-8") as cf:
            json.dump(self.cache, cf, indent=2)
        print(f"Cache updated => {self.cache_path}")


def main():
    # Example usage:
    pipeline = SciFactNliPipeline(
        skip_no_evidence=False,
        cache_save_interval=200,
        score_threshold=0.1,
        num_sent_per_paragraph=5,
        paragraph_sliding=True,
        paragraph_stride=1
    )

    # 1) Run inference (per-claim evaluation with multi-granular spans)
    pipeline.run_inference()

    # 2) Compute metrics
    metrics = pipeline.calculate_metrics()
    print("Metrics:", metrics)

    # 3) Save results
    pipeline.save_results(metrics)


if __name__ == "__main__":
    main()
