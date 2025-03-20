import json
import random
import math
from collections import defaultdict


def stratified_sample(docs, group_key_fn, total_sample_size):
    """
    Perform stratified sampling from docs.

    Args:
      docs: list of document objects.
      group_key_fn: function that takes a doc and returns a grouping key.
      total_sample_size: total number of documents to sample.

    Returns:
      A list of sampled documents (of size total_sample_size, if possible).
    """
    # Group documents by the grouping key.
    groups = defaultdict(list)
    for doc in docs:
        key = group_key_fn(doc)
        groups[key].append(doc)

    total_docs = len(docs)
    # Calculate initial allocation (might be fractional)
    allocations = {}
    fractional_parts = {}
    for key, group_docs in groups.items():
        group_count = len(group_docs)
        exact_alloc = (group_count / total_docs) * total_sample_size
        allocations[key] = math.floor(exact_alloc)
        fractional_parts[key] = exact_alloc - allocations[key]

    # Distribute remaining slots based on highest fractional parts.
    allocated = sum(allocations.values())
    remaining = total_sample_size - allocated
    # Sort groups by fractional part descending
    sorted_keys = sorted(fractional_parts, key=lambda k: fractional_parts[k], reverse=True)
    for key in sorted_keys:
        if remaining <= 0:
            break
        allocations[key] += 1
        remaining -= 1

    # Now sample from each group (if a group has fewer docs than allocation, take all)
    sampled_docs = []
    for key, count in allocations.items():
        group_docs = groups[key]
        if count >= len(group_docs):
            sampled_docs.extend(group_docs)
        else:
            sampled_docs.extend(random.sample(group_docs, count))

    # In rare cases the total may be less than total_sample_size due to small groups;
    # if so, fill in randomly from the remaining docs not yet chosen.
    if len(sampled_docs) < total_sample_size:
        sampled_set = set(id(doc) for doc in sampled_docs)
        remaining_docs = [doc for doc in docs if id(doc) not in sampled_set]
        additional = random.sample(remaining_docs, total_sample_size - len(sampled_docs))
        sampled_docs.extend(additional)

    return sampled_docs


def sample_preprocessed_scifact(input_filename, output_filename, sample_size=100):
    # Load the preprocessed_scifact dataset.
    with open(input_filename, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    # For preprocessed_scifact, group by number of claims per document.
    def group_key(doc):
        return len(doc.get("claims", []))

    sampled_docs = stratified_sample(docs, group_key, sample_size)

    # Save the sample to output JSON file.
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(sampled_docs, f, indent=2)
    print(f"Sampled {len(sampled_docs)} documents from preprocessed_scifact saved to {output_filename}")


def sample_preprocessed_nli4ct(input_filename, output_filename, sample_size=100):
    # Load the preprocessed_nli4ct dataset.
    with open(input_filename, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    # For preprocessed_nli4ct, group by the 'section' field.
    def group_key(doc):
        return doc.get("section", "Unknown")

    sampled_docs = stratified_sample(docs, group_key, sample_size)

    # Save the sample to output JSON file.
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(sampled_docs, f, indent=2)
    print(f"Sampled {len(sampled_docs)} documents from preprocessed_nli4ct saved to {output_filename}")


def main():
    # Set random seed for reproducibility if desired.
    random.seed(42)

    # Sample for preprocessed_scifact
    sample_preprocessed_scifact("scifact_annotated.json", "sample_scifact.json", sample_size=100)

    # Sample for preprocessed_nli4ct
    sample_preprocessed_nli4ct("preprocessed_nli4ct.json", "sample_nli4ct.json", sample_size=100)


if __name__ == "__main__":
    main()
