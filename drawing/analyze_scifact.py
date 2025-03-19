import json
from collections import Counter

def main():
    # Load data from a JSON file; adjust the filename as needed.
    with open('sample_scifact.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_documents = len(data)
    total_claims = 0
    # Count how many documents have X number of claims.
    claims_per_doc_counter = Counter()
    # Count evidence labels across all claims.
    claims_label_counter = Counter()
    # Count the number of supporting sentences per claim.
    supporting_sentences_counter = Counter()

    for doc in data:
        claims = doc.get("claims", [])
        num_claims = len(claims)
        total_claims += num_claims
        claims_per_doc_counter[num_claims] += 1

        # Process each claim
        for claim in claims:
            evidence = claim.get("evidence", {})
            # If there is no evidence info, assign label as "NEUTRAL"
            if not evidence:
                claims_label_counter["NEUTRAL"] += 1
            else:
                # evidence is a dictionary, whose keys are cited document ids.
                # Each key maps to a list of evidence items.
                for cited_doc, evidence_items in evidence.items():
                    for ev in evidence_items:
                        # Use the provided label; if none exists, default to "NEUTRAL"
                        label = ev.get("label", "NEUTRAL")
                        claims_label_counter[label] += 1

            # Calculate the total number of supporting sentences for this claim.
            total_sentences = 0
            if evidence:
                for cited_doc, evidence_items in evidence.items():
                    for ev in evidence_items:
                        sentences = ev.get("sentences", [])
                        total_sentences += len(sentences)
            # Add the count (could be zero) to the counter.
            supporting_sentences_counter[total_sentences] += 1

    # Print out the statistical insights:
    print("Statistical Insights:")
    print(f"Total number of documents: {total_documents}")
    print(f"Total number of claims: {total_claims}\n")

    print("Distribution of claims per document:")
    for num_claims, count in sorted(claims_per_doc_counter.items()):
        print(f"  {num_claims} claim(s): {count} document(s)")

    print("\nDistribution of claim evidence labels:")
    for label, count in claims_label_counter.items():
        print(f"  {label}: {count}")

    print("\nDistribution of supporting sentences per claim:")
    for sent_count, count in sorted(supporting_sentences_counter.items()):
        print(f"  {sent_count} supporting sentence(s): {count} claim(s)")

if __name__ == "__main__":
    main()
