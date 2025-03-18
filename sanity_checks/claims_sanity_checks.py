import json
import os

# Update the file name to your new output file.
CLAIMS_JSON = "../data/selected_scifact_subset_with_claims.json"

# Define the expected strategies. Adjust this set if you have a different configuration.
EXPECTED_STRATEGIES = {
    "default_prompt",
    "entities_aware",
    "relations_aware",
    # "kg_based",
}

def load_data(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_required_fields(data):
    """
    Ensure each entry has the required fields for the new JSON structure.
    Required fields are:
      - "doc_id"
      - At least one abstract field ("abstract_raw" or "abstract_sents")
      - "entities"
      - "relations"
      - "claims"
    """
    required_keys = {"doc_id", "entities", "relations", "claims"}
    missing = []
    for entry in data:
        missing_keys = required_keys - entry.keys()
        # Check for at least one abstract field.
        if "abstract_raw" not in entry and "abstract_sents" not in entry:
            missing_keys.add("abstract (raw or sents)")
        if missing_keys:
            missing.append((entry.get("doc_id", "NO DOC_ID"), missing_keys))
    return missing

def check_claims_structure(data):
    """
    Checks that:
      - The "claims" field exists and is a dictionary.
      - For each entry, all expected strategies are present in the "claims" dict.
      - For each strategy in "claims":
          - The value is a list.
          - It is not empty.
          - Each claim is a non-empty string.
    Reports any issues found.
    """
    issues = []
    for entry in data:
        doc_id = entry.get("doc_id", "NO DOC_ID")
        claims = entry.get("claims")
        if claims is None:
            issues.append((doc_id, "Missing 'claims' field"))
            continue
        if not isinstance(claims, dict):
            issues.append((doc_id, "'claims' is not a dictionary"))
            continue

        # Check if any expected strategy is missing.
        missing_strategies = EXPECTED_STRATEGIES - set(claims.keys())
        if missing_strategies:
            issues.append((doc_id, f"Missing claim generation for strategies: {', '.join(missing_strategies)}"))

        # If the claims dict is empty, flag that as well.
        if not claims:
            issues.append((doc_id, "'claims' field is empty, no strategies attempted"))

        for strategy, claim_list in claims.items():
            if not isinstance(claim_list, list):
                issues.append((doc_id, f"Strategy '{strategy}' is not a list"))
                continue
            if len(claim_list) == 0:
                issues.append((doc_id, f"Strategy '{strategy}' is an empty list"))
            else:
                for i, claim in enumerate(claim_list):
                    if not isinstance(claim, str):
                        issues.append((doc_id, f"Strategy '{strategy}' claim index {i} is not a string"))
                    elif not claim.strip():
                        issues.append((doc_id, f"Strategy '{strategy}' claim index {i} is empty or whitespace"))
    return issues

def check_empty_claim_strategies(data):
    """
    Checks if any strategies have an empty array across all entries.
    Returns a list of strategies that are empty in every single entry.
    """
    strategy_counts = {}
    for entry in data:
        claims = entry.get("claims", {})
        for strategy, claim_list in claims.items():
            if strategy not in strategy_counts:
                strategy_counts[strategy] = 0
            if claim_list:
                strategy_counts[strategy] += len(claim_list)
    empty_strategies = [strategy for strategy, count in strategy_counts.items() if count == 0]
    return empty_strategies

def run_sanity_checks():
    print(f"Loading claims file from '{CLAIMS_JSON}'...")
    data = load_data(CLAIMS_JSON)
    total_entries = len(data)
    print(f"Total entries loaded: {total_entries}\n")

    # Check for missing required fields.
    missing_fields = check_required_fields(data)
    if missing_fields:
        print("Entries with missing required fields:")
        for doc_id, keys in missing_fields:
            print(f"  - DOC_ID {doc_id}: missing {', '.join(keys)}")
    else:
        print("All entries have the required fields.")

    # Check the structure and content of the "claims" field.
    claim_issues = check_claims_structure(data)
    if claim_issues:
        print("\nIssues found in 'claims' field:")
        for doc_id, issue in claim_issues:
            print(f"  - DOC_ID {doc_id}: {issue}")
    else:
        print("\nAll claim entries appear well-structured.")

    # Check for empty strategies across all entries.
    empty_strategies = check_empty_claim_strategies(data)
    if empty_strategies:
        print("\nWARNING: The following strategies have empty claim arrays across ALL entries:")
        for strategy in empty_strategies:
            print(f"  - {strategy}")
    else:
        print("\nAll claim strategies have at least some populated claims.")

    # Summary stats by strategy.
    strategy_counts = {}
    for entry in data:
        claims = entry.get("claims", {})
        for strategy, claim_list in claims.items():
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + len(claim_list)
    if strategy_counts:
        print("\nSummary of claim counts by strategy:")
        for strategy, count in strategy_counts.items():
            print(f"  - {strategy}: {count} claim(s) total")
    else:
        print("\nNo claim strategies found.")

    print("\nSanity check complete.")

if __name__ == "__main__":
    run_sanity_checks()
