# scifact_annotate.py
import os, json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from relik import Relik

NER_MODEL = "d4data/biomedical-ner-all"
RE_MODEL = "relik-ie/relik-relation-extraction-small"

INPUT_JSON = "data/scifact/preprocessed_scifact.json"
OUTPUT_JSON = "data/scifact/scifact_annotated.json"


def init_ner_pipeline(model_name=NER_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def init_re_pipeline(model_name=RE_MODEL):
    # Example device logic: prefer cuda -> mps -> cpu
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return Relik.from_pretrained(model_name, device=device)


def main():
    # Load data from preprocessed SciFact
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    ner_pipe = init_ner_pipeline()
    re_pipe = init_re_pipeline()

    annotated_docs = []
    for entry in data:
        doc_id = entry["doc_id"]
        abstract_raw = entry["abstract_raw"]
        # 1) Run NER
        if not abstract_raw.strip():
            entities = []
            relations = []
        else:
            ner_results = ner_pipe(abstract_raw)
            # 2) Run RE
            #   ReLiK uses the text; you can apply it directly to abstract_raw.
            #   Or you might prefer sentence-level RE. It's up to you.
            re_results = re_pipe(abstract_raw)
            entities = ner_results
            relations = []
            if hasattr(re_results, "triplets"):
                for t in re_results.triplets:
                    if t.confidence >= 0.7:
                        relations.append({
                            "subject": t.subject.text,
                            "relation": t.label,
                            "object": t.object.text,
                            "confidence": t.confidence
                        })

        # Store updated fields
        annotated_docs.append({
            "doc_id": doc_id,
            "abstract_sents": entry["abstract_sents"],
            "abstract_raw": abstract_raw,
            "claims": entry["claims"],  # The original SciFact claims from preprocessed
            "entities": entities,
            "relations": relations
        })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(annotated_docs, f, indent=2)
    print(f"Saved annotated SciFact to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
