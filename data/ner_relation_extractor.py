import os
import json
import re
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForSequenceClassification

from relik import Relik

# 2) Adjust this model name to whichever biomedical NER model you prefer
BIOMED_NER_MODEL = "d4data/biomedical-ner-all"
BIOMED_RE_MODEL = "relik-ie/relik-relation-extraction-small"


def load_abstracts(json_path: str = "pubmed_abstracts.json") -> List[Dict[str, Any]]:
    """Load PubMed abstracts from JSON."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File '{json_path}' not found.")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def init_ner_pipeline(model_name: str = BIOMED_NER_MODEL):
    """Initialize the HuggingFace NER pipeline with a domain-specific model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner


def perform_ner_on_abstract(abstract_text: str, ner_pipeline) -> List[Dict[str, Any]]:
    """Use the given NER pipeline to extract entities from a single abstract text."""
    if not abstract_text or abstract_text.strip() == "":
        return []

    ner_results = ner_pipeline(abstract_text)

    # Convert `numpy.float32` scores to standard Python `float`
    for entity in ner_results:
        if "score" in entity:
            entity["score"] = float(entity["score"])  # Convert float32 to float

    return ner_results


def init_re_pipeline(model_name: str = BIOMED_RE_MODEL):
    """
    Initialize the ReLiK relation extraction pipeline.
    This uses ReLiK's library instead of Hugging Face's `transformers`.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    relik_pipeline = Relik.from_pretrained(model_name, device=device)
    return relik_pipeline


def extract_relations(
        abstract_text: str,
        entities: list,
        re_pipeline,
        confidence_threshold: float = 0.7
) -> list:
    """
    Extracts relations using ReLiK's pretrained relation extraction model.
    """
    relations = []
    if not abstract_text.strip():
        return relations

    # Use the ReLiK pipeline
    re_results = re_pipeline(abstract_text)  # This returns a RelikOutput object

    if hasattr(re_results, "triplets") and re_results.triplets:
        for triplet in re_results.triplets:
            relation_label = triplet.label
            confidence = getattr(triplet, "confidence", 1.0)  # Default to 1.0 if missing

            # Apply confidence threshold
            if confidence >= confidence_threshold:
                relations.append({
                    "subject": triplet.subject.text,
                    "relation": relation_label,
                    "object": triplet.object.text,
                    "confidence": confidence
                })

    return relations


def main():
    # 1) Load abstracts
    abstracts = load_abstracts("pubmed_abstracts.json")
    print(f"Loaded {len(abstracts)} abstracts.")

    # 2) Initialize NER
    ner_pipe = init_ner_pipeline(BIOMED_NER_MODEL)
    print("NER pipeline initialized.")

    # 3) Initialize RE pipeline
    #    (Ensure that the chosen model actually has relation-classification labels you expect.)
    re_pipe = init_re_pipeline(BIOMED_RE_MODEL)
    print("RE pipeline initialized.")

    annotated_data = []
    for entry in abstracts:
        pmid = entry.get("pmid", "N/A")
        title = entry.get("title", "")
        abstract_text = entry.get("abstract", "")

        # NER
        ner_results = perform_ner_on_abstract(abstract_text, ner_pipe)

        # RE
        rel_results = extract_relations(abstract_text, ner_results, re_pipeline=re_pipe, confidence_threshold=0.7)

        annotated_data.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract_text,
            "entities": ner_results,
            "relations": rel_results
        })

    # 4) Save final JSON
    output_path = "pubmed_abstracts_with_ner_re.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated_data, f, indent=2)
    print(f"Annotated data (with RE) saved to '{output_path}'.")


if __name__ == "__main__":
    main()
