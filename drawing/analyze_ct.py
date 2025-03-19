import json
from collections import Counter

def process_trial_json(data):
    """
    Processes JSON objects for trial eligibility-type documents.
    Each document has a "source_doc_id", a "section", a list of "source_text_sents",
    and a list of "claims". Each claim contains a "claim_id", "claim", "label" (which may be empty),
    and "evidence_indices".
    """
    total_documents = len(data)
    total_claims = 0
    section_counter = Counter()
    claims_label_counter = Counter()
    evidence_indices_counter = Counter()  # count by number of evidence indices per claim

    for doc in data:
        # Extract the section (programmatically; could be any value)
        section = doc.get("section", "Unknown")
        section_counter[section] += 1

        claims = doc.get("claims", [])
        total_claims += len(claims)
        for claim in claims:
            # If the label is missing or an empty string, convert it to "NEUTRAL"
            label = claim.get("label")
            if not label or label.strip() == "":
                label = "NEUTRAL"
                claim["label"] = label  # update in the object if needed
            claims_label_counter[label] += 1

            # Count evidence indices distribution per claim
            evidence_indices = claim.get("evidence_indices", [])
            evidence_indices_counter[len(evidence_indices)] += 1

    # Print statistical insights:
    print("Statistical Insights for Trial Eligibility JSON Objects:")
    print(f"Total number of documents: {total_documents}")
    print(f"Total number of claims: {total_claims}\n")

    print("Distribution of documents by section:")
    for section, count in section_counter.items():
        print(f"  {section}: {count} document(s)")

    print("\nDistribution of claim labels:")
    for label, count in claims_label_counter.items():
        print(f"  {label}: {count}")

    print("\nDistribution of number of evidence indices per claim:")
    for num_indices, count in evidence_indices_counter.items():
        print(f"  {num_indices} evidence index{'es' if num_indices != 1 else ''}: {count} claim(s)")

def main():
    # Adjust the filename as needed; here we assume the JSON data is in a file named 'trial_data.json'
    with open('sample_nli4ct.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    process_trial_json(data)

if __name__ == "__main__":
    main()
