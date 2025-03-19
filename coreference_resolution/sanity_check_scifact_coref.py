import json
import difflib
import statistics


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def similarity_ratio(text1, text2):
    """Compute similarity ratio between two strings using difflib."""
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def sanity_check(original_docs, processed_docs):
    abstract_similarities = []
    claim_similarities = []
    missing_docs = []
    mismatched_claims = []

    # Create a lookup dictionary for processed docs by doc_id
    processed_lookup = {doc.get("doc_id", ""): doc for doc in processed_docs}

    for orig_doc in original_docs:
        doc_id = orig_doc.get("doc_id", "")
        proc_doc = processed_lookup.get(doc_id)
        if not proc_doc:
            missing_docs.append(doc_id)
            continue

        # Compare abstracts
        orig_abs = orig_doc.get("abstract_raw", "")
        proc_abs = proc_doc.get("abstract_raw", "")
        abs_ratio = similarity_ratio(orig_abs, proc_abs)
        abstract_similarities.append(abs_ratio)

        # Compare claims
        orig_claims = orig_doc.get("claims", [])
        proc_claims = proc_doc.get("claims", [])
        # Create a lookup for processed claims by claim id (converted to string)
        proc_claim_lookup = {str(claim.get("id")): claim for claim in proc_claims}

        for orig_claim in orig_claims:
            claim_id = str(orig_claim.get("id"))
            proc_claim = proc_claim_lookup.get(claim_id)
            if not proc_claim:
                mismatched_claims.append({"doc_id": doc_id, "claim_id": claim_id})
                continue
            orig_text = orig_claim.get("claim", "")
            proc_text = proc_claim.get("claim", "")
            claim_ratio = similarity_ratio(orig_text, proc_text)
            claim_similarities.append(claim_ratio)

    # Report summary statistics
    print("=== Sanity Check Summary ===")
    print(f"Total original documents: {len(original_docs)}")
    print(f"Total processed documents found: {len(processed_docs) - len(missing_docs)}")
    if missing_docs:
        print(f"Missing processed docs for IDs: {missing_docs}")
    if mismatched_claims:
        print("Mismatched claim IDs in some documents:")
        for m in mismatched_claims:
            print(f"  Document ID: {m['doc_id']}, Missing Claim ID: {m['claim_id']}")
    if abstract_similarities:
        print("\nAbstract Similarity Scores:")
        print(f"  Average similarity: {statistics.mean(abstract_similarities):.3f}")
        print(f"  Min similarity: {min(abstract_similarities):.3f}")
        print(f"  Max similarity: {max(abstract_similarities):.3f}")
    if claim_similarities:
        print("\nClaim Similarity Scores:")
        print(f"  Average similarity: {statistics.mean(claim_similarities):.3f}")
        print(f"  Min similarity: {min(claim_similarities):.3f}")
        print(f"  Max similarity: {max(claim_similarities):.3f}")

    # Optionally, list some individual comparisons for inspection
    print("\nSome individual abstract comparisons:")
    for i, ratio in enumerate(abstract_similarities[:5], start=1):
        print(f"  Document {i}: similarity = {ratio:.3f}")

    print("\nSome individual claim comparisons:")
    for i, ratio in enumerate(claim_similarities[:5], start=1):
        print(f"  Claim {i}: similarity = {ratio:.3f}")


def main():
    # Filenames (adjust as needed)
    original_filename = "sample_scifact.json"
    processed_filename = "sample_scifact_coref.json"

    # Load original and processed documents
    original_docs = load_json(original_filename)
    processed_docs = load_json(processed_filename)

    # Run sanity check on the two datasets
    sanity_check(original_docs, processed_docs)


if __name__ == "__main__":
    main()
