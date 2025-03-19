import json
from collections import Counter, defaultdict

# ------------------- CONFIGURATION -------------------
# Path to the preprocessed dataset file.
PREPROCESSED_FILE = "preprocessed_nli4ct.json"


# -----------------------------------------------------

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    data = load_json(PREPROCESSED_FILE)

    total_source_groups = len(data)
    total_claims = 0
    label_counter = Counter()
    section_counter = Counter()
    claims_per_source = []

    # For each source group in the dataset
    for group in data:
        claims = group.get("claims", [])
        num_claims = len(claims)
        total_claims += num_claims
        claims_per_source.append(num_claims)
        section = group.get("section", "Unknown")
        section_counter[section] += num_claims
        for claim in claims:
            label = claim.get("label", "Unknown")
            label_counter[label] += 1

    avg_claims_per_source = total_claims / total_source_groups if total_source_groups else 0

    # Print insights
    print("Insights into Preprocessed NLI4CT Dataset:")
    print("--------------------------------------------------")
    print(f"Total number of source groups: {total_source_groups}")
    print(f"Total number of claims: {total_claims}")
    print(f"Average number of claims per source group: {avg_claims_per_source:.2f}")
    print("\nDistribution of Labels:")
    for label, count in label_counter.items():
        print(f"  {label}: {count}")
    print("\nDistribution of Claims Across Sections:")
    for section, count in section_counter.items():
        print(f"  {section}: {count}")


if __name__ == "__main__":
    main()
