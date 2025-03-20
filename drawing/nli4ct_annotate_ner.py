# nli4ct_annotate.py
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Use the same NER model as before
NER_MODEL = "d4data/biomedical-ner-all"

INPUT_JSON = "sample_nli4ct.json"
OUTPUT_JSON = "sample_nli4ct_annotated.json"


def init_ner_pipeline(model_name=NER_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def convert_float32_in_dict(d):
    """Recursively convert NumPy float32 in a dict to Python float."""
    for key, value in d.items():
        if isinstance(value, dict):
            convert_float32_in_dict(value)
        elif isinstance(value, list):
            d[key] = [float(item) if hasattr(item, "dtype") and item.dtype == "float32" else item for item in value]
        else:
            if hasattr(value, "dtype") and value.dtype == "float32":
                d[key] = float(value)
    return d


def main():
    # Load the target dataset
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    ner_pipe = init_ner_pipeline()

    annotated_docs = []
    for entry in data:
        source_doc_id = entry["source_doc_id"]
        section = entry["section"]
        # Join the list of sentences into one string
        source_text_sents = entry["source_text_sents"]
        source_text_raw = "\n".join(source_text_sents)
        claims = entry["claims"]

        # Run NER on the source text
        if not source_text_raw.strip():
            entities = []
        else:
            ner_results = ner_pipe(source_text_raw)
            entities = [convert_float32_in_dict(ent) for ent in ner_results]

        # Build the annotated entry
        annotated_docs.append({
            "source_doc_id": source_doc_id,
            "section": section,
            "source_text_sents": source_text_sents,
            "claims": claims,
            "entities": entities
        })

    os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(annotated_docs, f, indent=2)
    print(f"Saved annotated NLI4CT data to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
