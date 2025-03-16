import json
import os

CLAIMS_JSON = "pubmed_abstracts_with_claims.json"


def load_data(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_required_fields(data):
    """Ensure each entry has required fields."""
    required_keys = {"pmid", "title", "abstract", "entities", "relations", "claims"}
    missing = []
    for entry in data:
        missing_keys = required_keys - entry.keys()
        if missing_keys:
            missing.append((entry.get("pmid", "NO PMID"), missing_keys))
    return missing


def check_claims_structure(data):
    """
    Checks that:
    - The "claims" field exists and is a dictionary.
    - For each strategy in "claims":
      - The value is a list.
      - It is not empty.
      - Each claim is a non-empty string.
    - Reports if any strategy's array is empty.
    Returns a list of issues found.
    """
    issues = []
    for entry in data:
        pmid = entry.get("pmid", "NO PMID")
        claims = entry.get("claims")
        if claims is None:
            issues.append((pmid, "Missing 'claims' field"))
            continue
        if not isinstance(claims, dict):
            issues.append((pmid, "'claims' is not a dictionary"))
            continue

        # Track if any strategy has an empty array
        for strategy, claim_list in claims.items():
            if not isinstance(claim_list, list):
                issues.append((pmid, f"Strategy '{strategy}' is not a list"))
                continue
            if len(claim_list) == 0:
                issues.append((pmid, f"Strategy '{strategy}' is an empty list"))
            else:
                # Check each claim string in the list.
                for i, claim in enumerate(claim_list):
                    if not isinstance(claim, str):
                        issues.append((pmid, f"Strategy '{strategy}' claim index {i} is not a string"))
                    elif not claim.strip():
                        issues.append((pmid, f"Strategy '{strategy}' claim index {i} is empty or whitespace"))
    return issues


def check_empty_claim_strategies(data):
    """
    Checks if any strategies have an empty array across all abstracts.
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
        for pmid, keys in missing_fields:
            print(f"  - PMID {pmid}: missing {', '.join(keys)}")
    else:
        print("All entries have the required fields.")

    # Check the structure and content of the "claims" field.
    claim_issues = check_claims_structure(data)
    if claim_issues:
        print("\nIssues found in 'claims' field:")
        for pmid, issue in claim_issues:
            print(f"  - PMID {pmid}: {issue}")
    else:
        print("\nAll claim entries appear well-structured.")

    # Check for empty strategies across all abstracts
    empty_strategies = check_empty_claim_strategies(data)
    if empty_strategies:
        print("\nWARNING: The following strategies have empty claim arrays across ALL abstracts:")
        for strategy in empty_strategies:
            print(f"  - {strategy}")
    else:
        print("\nAll claim strategies have at least some populated claims.")

    # Summary stats by strategy:
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
