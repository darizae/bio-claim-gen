import os
import json
from collections import defaultdict
from typing import Dict, List

SCIFACT_DIR = "data/scifact"

# We will read all three files. Adapt if you only want train + dev, or something else.
CLAIMS_TRAIN = os.path.join(SCIFACT_DIR, "claims_train.jsonl")
CLAIMS_DEV = os.path.join(SCIFACT_DIR, "claims_dev.jsonl")
CLAIMS_TEST = os.path.join(SCIFACT_DIR, "claims_test.jsonl")

CORPUS_FILE = os.path.join(SCIFACT_DIR, "corpus.jsonl")

PREPROCESSED_CLAIMS_PATH = os.path.join("preprocessed_scifact.json")


def load_corpus(path: str) -> Dict[str, Dict]:
    """
    Load SciFact's corpus.jsonl into a dict:
      { doc_id (str): { 'abstract': [sentence1, sentence2, ...], ...}, ...}
    """
    corpus_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            corpus_dict[str(entry["doc_id"])] = entry
    return corpus_dict


def load_claims(path: str) -> List[Dict]:
    """
    Loads SciFact claims from a .jsonl, returns list of dicts.
    """
    claims_data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            claims_data.append(json.loads(line))
    return claims_data


def load_all_claims(train_path: str, dev_path: str, test_path: str) -> List[Dict]:
    """
    Load all claims (train, dev, test) in one big list.
    If you want to keep them separate, you can store them separately, but
    here we'll combine them.
    """
    combined = []
    for p in [train_path, dev_path, test_path]:
        if not os.path.exists(p):
            # If you don't have a test set with labels, you can skip or handle differently.
            # We'll just skip if missing.
            continue
        c = load_claims(p)
        combined.extend(c)
    return combined


def preprocess_scifact_data(corpus_file: str) -> List[Dict]:
    """
    Performs the steps:
      1) Load all claims (train, dev, test)
      2) Keep only single-doc claims (len(cited_doc_ids) == 1)
      3) Group them by doc_id
      4) Load & store the raw abstract (joined sentences) and the segmented version
      5) Returns a list of doc-level entries:
           [
             {
               "doc_id": <str>,
               "abstract_sents": [s1, s2, ...],
               "abstract_raw": "<s1> <s2> ...",
               "claims": [ {...}, {...} ]
             },
             ...
           ]
    """
    # 1) Load all claims
    all_claims = load_all_claims(CLAIMS_TRAIN, CLAIMS_DEV, CLAIMS_TEST)
    corpus_dict = load_corpus(corpus_file)

    # 2) Filter out multi-doc claims, group single-doc claims by doc_id
    doc2claims = defaultdict(list)
    for claim in all_claims:
        doc_ids = claim.get("cited_doc_ids", [])
        if len(doc_ids) == 1:
            single_id = str(doc_ids[0])
            doc2claims[single_id].append(claim)

    # 3) Build the list
    preprocessed = []
    for doc_id, claim_objs in doc2claims.items():
        doc_info = corpus_dict.get(doc_id)
        if not doc_info:
            continue

        abstract_sents = doc_info.get("abstract", [])
        # We'll create a raw version by joining with a space (or newline if you prefer)
        abstract_raw = " ".join(abstract_sents)

        entry = {
            "doc_id": doc_id,
            "abstract_sents": abstract_sents,
            "abstract_raw": abstract_raw,
            "claims": claim_objs
        }
        preprocessed.append(entry)

    return preprocessed


def main():
    # If cached file exists, load it; otherwise create it.
    if os.path.exists(PREPROCESSED_CLAIMS_PATH):
        print("Loading preprocessed data (ALL) from cache...")
        with open(PREPROCESSED_CLAIMS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("Cache not found, preprocessing SciFact data for train/dev/test (ALL)...")
        data = preprocess_scifact_data(CORPUS_FILE)
        # Save to JSON
        with open(PREPROCESSED_CLAIMS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Preprocessed data saved to {PREPROCESSED_CLAIMS_PATH}")

    print(f"Number of doc entries in cache: {len(data)}")


if __name__ == "__main__":
    main()
