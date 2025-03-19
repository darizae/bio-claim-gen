import os
import json
from collections import defaultdict

# ------------------- CONFIGURATION -------------------
# Paths to the split JSON files that contain the claim entries.
SPLIT_FILES = [
    "train.json",
    "dev.json",
    "test.json",
    "Gold_test.json"
]

# Directory containing the CT JSON files (each document by its NCT ID).
CT_JSON_DIR = "CT json"

# Output file path for the preprocessed dataset.
OUTPUT_FILE = "preprocessed_nli4ct.json"

# Only include entries of type "Single"
CLAIM_TYPE = "Single"


# -----------------------------------------------------

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    # This dictionary will group claims by (primary_id, section_id).
    grouped_claims = defaultdict(list)

    # Process each split file.
    for split_file in SPLIT_FILES:
        if not os.path.exists(split_file):
            print(f"Warning: {split_file} not found. Skipping.")
            continue
        split_data = load_json(split_file)
        for claim_id, entry in split_data.items():
            if entry.get("Type") != CLAIM_TYPE:
                continue  # Skip non-Single entries.
            primary_id = entry.get("Primary_id")
            section_id = entry.get("Section_id")
            statement = entry.get("Statement")
            label = entry.get("Label")
            evidence_indices = entry.get("Primary_evidence_index", [])

            # Create a claim object.
            claim_obj = {
                "claim_id": claim_id,
                "claim": statement,
                "label": label,
                "evidence_indices": evidence_indices
            }
            # Group by (primary_id, section_id)
            key = (primary_id, section_id)
            grouped_claims[key].append(claim_obj)

    # Now, for each group, load the corresponding CT JSON file and extract the section text.
    preprocessed_data = []
    for (primary_id, section_id), claims in grouped_claims.items():
        ct_filepath = os.path.join(CT_JSON_DIR, f"{primary_id}.json")
        if not os.path.exists(ct_filepath):
            print(f"Warning: CT file for {primary_id} not found. Skipping group.")
            continue

        ct_data = load_json(ct_filepath)
        # Extract the section text (assumed to be an array of sentences)
        source_text_sents = ct_data.get(section_id)
        if source_text_sents is None:
            print(f"Warning: Section '{section_id}' not found in {primary_id}. Skipping group.")
            continue

        # Create an aggregated object for this source.
        group_obj = {
            "source_doc_id": primary_id,
            "section": section_id,
            "source_text_sents": source_text_sents,
            "claims": claims
        }
        preprocessed_data.append(group_obj)

    # Save the aggregated preprocessed data to a JSON file.
    save_json(preprocessed_data, OUTPUT_FILE)
    print(f"Preprocessed dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
