import json
import os
from collections import Counter
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# ------------------- CONFIGURATION -------------------
# Path to the preprocessed NLI4CT JSON file.
DATA_PATH = "preprocessed_nli4ct.json"
# Cache file for storing computed NLI predictions.
CACHE_PATH = "nli4ct_cache.json"
# Output file for claim-level verification results.
RESULTS_PATH = "nli4ct_verification_results.json"

# Inference configuration parameters.
CACHE_SAVE_INTERVAL = 200
SCORE_THRESHOLD = 0.1  # Differential score threshold below which a claim is considered neutral.
NUM_SENT_PER_PARAGRAPH = 5
PARAGRAPH_SLIDING = True
PARAGRAPH_STRIDE = 1


# -----------------------------------------------------

class NLIModel:
    """
    Wraps a HuggingFace NLI model.
    Uses roberta-large-mnli by default.
    """

    def __init__(self, model_name: str = "roberta-large-mnli", device: str = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()
        # Mapping: 0 -> contradiction, 1 -> neutral, 2 -> entailment.
        self.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def predict(self, premise: str, hypothesis: str) -> List[float]:
        """
        Returns raw probabilities [P(contradiction), P(neutral), P(entailment)]
        for a premise-hypothesis pair.
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


def split_into_paragraphs(sentences: List[str], num_sent_per_paragraph: int = 5,
                          sliding: bool = True, stride: int = 1) -> List[str]:
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


class NLI4CTNliPipeline:
    """
    Orchestrates running your NLI claim verification system on the NLI4CT preprocessed dataset.
    Groups claims by source document and section, and computes per-claim predictions using multi-granular spans.
    """

    def __init__(self,
                 data_path: str = DATA_PATH,
                 cache_path: str = CACHE_PATH,
                 results_path: str = RESULTS_PATH,
                 cache_save_interval: int = CACHE_SAVE_INTERVAL,
                 score_threshold: float = SCORE_THRESHOLD,
                 num_sent_per_paragraph: int = NUM_SENT_PER_PARAGRAPH,
                 paragraph_sliding: bool = PARAGRAPH_SLIDING,
                 paragraph_stride: int = PARAGRAPH_STRIDE):
        self.data_path = data_path
        self.cache_path = cache_path
        self.results_path = results_path
        self.cache_save_interval = cache_save_interval
        self.score_threshold = score_threshold
        self.num_sent_per_paragraph = num_sent_per_paragraph
        self.paragraph_sliding = paragraph_sliding
        self.paragraph_stride = paragraph_stride

        self.nli_model = NLIModel()

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.nli4ct_data = json.load(f)  # List of grouped source entries.

        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as cf:
                self.cache = json.load(cf)
        else:
            self.cache = {}  # Cache format: "span_text||claim_text": [p_contradiction, p_neutral, p_entailment]

        self.results = []  # Will store final results at claim level.

    def run_inference(self):
        """
        For each source group and for each claim in that group,
        compute NLI probabilities for each span (both sentence and paragraph level)
        and select the span with the highest differential score.
        """
        inference_counter = 0
        # First, estimate the total number of (group -> claim -> span) inferences.
        total_pairs = 0
        for group in self.nli4ct_data:
            source_sents = group.get("source_text_sents", [])
            paragraphs = split_into_paragraphs(source_sents,
                                               self.num_sent_per_paragraph,
                                               self.paragraph_sliding,
                                               self.paragraph_stride)
            spans = source_sents + paragraphs
            total_pairs += len(spans) * len(group.get("claims", []))
        with tqdm(total=total_pairs, desc="NLI4CT Inference") as pbar:
            for group in self.nli4ct_data:
                doc_id = group.get("source_doc_id")
                section = group.get("section")
                source_sents = group.get("source_text_sents", [])
                paragraphs = split_into_paragraphs(source_sents,
                                                   self.num_sent_per_paragraph,
                                                   self.paragraph_sliding,
                                                   self.paragraph_stride)
                # Create span list with granularity info.
                spans = [{"text": sent, "granularity": "sentence"} for sent in source_sents] + \
                        [{"text": para, "granularity": "paragraph"} for para in paragraphs]

                for claim in group.get("claims", []):
                    claim_id = claim.get("claim_id")
                    claim_text = claim.get("claim")
                    # Map gold label to lowercase. Note: in NLI4CT the neutral label might be "None", so we map it to "neutral".
                    gold_label = (claim.get("label") or "neutral").lower()
                    if gold_label == "none":
                        gold_label = "neutral"

                    best_result = self.aggregate_scores(spans, claim_text)
                    self.results.append({
                        "doc_id": doc_id,
                        "section": section,
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

    def aggregate_scores(self, spans: List[Dict[str, str]], claim_text: str) -> Dict:
        """
        For a given claim, compute NLIScore (p_entail - p_contr) for each span and select the span with the highest score.
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

            # probs: [P(contradiction), P(neutral), P(entailment)]
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
        Compute overall accuracy, macro precision, recall, and F1 comparing predicted vs. gold labels.
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
        Save claim-level predictions and overall metrics to a JSON file.
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
        Save the cache to disk.
        """
        with open(self.cache_path, "w", encoding="utf-8") as cf:
            json.dump(self.cache, cf, indent=2)
        print(f"Cache updated -> {self.cache_path}")


def main():
    pipeline = NLI4CTNliPipeline(
        cache_save_interval=CACHE_SAVE_INTERVAL,
        score_threshold=SCORE_THRESHOLD,
        num_sent_per_paragraph=NUM_SENT_PER_PARAGRAPH,
        paragraph_sliding=PARAGRAPH_SLIDING,
        paragraph_stride=PARAGRAPH_STRIDE
    )
    pipeline.run_inference()
    metrics = pipeline.calculate_metrics()
    print("Metrics:", metrics)
    pipeline.save_results(metrics)


if __name__ == "__main__":
    main()
