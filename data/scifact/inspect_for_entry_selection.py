#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import random
from collections import Counter, defaultdict


def get_ground_truth_label(claim_data, doc_id):
    """
    Extract a single "dominant" ground-truth label from the claim's evidence for doc_id.
    We'll follow a simplified logic:
      - If doc_id is not in evidence, return "NOINFO".
      - Otherwise, if any evidence block has "SUPPORT", we say "SUPPORT".
      - Otherwise, if any evidence block has "CONTRADICT", we say "CONTRADICT".
      - Otherwise, "NOINFO".

    (You can refine if you need a different approach.)
    """
    ev = claim_data.get("evidence", {})
    if doc_id not in ev:
        return "NOINFO"

    ev_list = ev[doc_id]
    labels_found = {item["label"].upper() for item in ev_list if "label" in item}
    if "SUPPORT" in labels_found:
        return "SUPPORT"
    elif "CONTRADICT" in labels_found:
        return "CONTRADICT"
    else:
        return "NOINFO"


def main():
    data_path = "../../preprocessed_scifact.json"
    with open(data_path, "r", encoding="utf-8") as f:
        scifact_data = json.load(f)

    # 1) Filter: keep only documents that have >= 3 claims
    docs_with_3plus_claims = []
    for doc in scifact_data:
        claims = doc.get("claims", [])
        if len(claims) >= 3:
            docs_with_3plus_claims.append(doc)

    print(f"Total number of docs in dataset: {len(scifact_data)}")
    print(f"Number of docs with >= 3 claims: {len(docs_with_3plus_claims)}")

    # 2) Compute distribution of ground-truth labels among these docs
    #    We'll gather label counts for each doc, and overall.
    overall_label_counter = Counter()
    doc_label_distribution = []

    for doc in docs_with_3plus_claims:
        doc_id = str(doc["doc_id"])
        claims = doc["claims"]
        label_count = Counter()

        for claim_data in claims:
            label = get_ground_truth_label(claim_data, doc_id)
            label_count[label] += 1
            overall_label_counter[label] += 1

        doc_label_distribution.append({
            "doc_id": doc_id,
            "num_claims": len(claims),
            "label_count": dict(label_count)
        })

    print("\n--- Overall Label Distribution (Docs with >=3 claims) ---")
    for lbl, cnt in overall_label_counter.items():
        print(f"{lbl}: {cnt}")
    print("---------------------------------------------------------\n")

    # 3) Optionally, gather other criteria/insights, e.g. abstract length
    #    Let’s record sentence count and approximate token count
    for doc_dist in doc_label_distribution:
        doc_id = doc_dist["doc_id"]
        doc_obj = next(d for d in docs_with_3plus_claims if str(d["doc_id"]) == doc_id)

        abstract_sents = doc_obj["abstract_sents"]
        sentence_count = len(abstract_sents)

        # Approx token count by splitting on whitespace
        token_count = sum(len(s.split()) for s in abstract_sents)

        doc_dist["sentence_count"] = sentence_count
        doc_dist["token_count"] = token_count

    # (You could print or sort by these new stats to see how large the abstracts are.)

    # 4) Try to pick about ~100 abstracts from docs_with_3plus_claims,
    #    with some label-balance constraints (if you want).

    # Example "balanced" approach (very naive):
    #   - We want a fairly even distribution of SUPPORT vs. CONTRADICT vs. NOINFO
    #   - We'll group docs by whichever label is most frequent for that doc
    #   - Then pick from each group in roughly equal proportion
    # This is just an example. You might define "balanced" differently.

    # Group docs by "majority label" in their claims
    doc_groups = defaultdict(list)
    for doc_dist in doc_label_distribution:
        # find the label with the highest count
        label_count_dict = doc_dist["label_count"]
        majority_label = max(label_count_dict, key=label_count_dict.get)
        doc_groups[majority_label].append(doc_dist)

    # Let's say we want about 100 docs total
    desired_total = 100
    # We'll attempt to split equally among the label categories we found
    # sum of sizes in doc_groups might be bigger or smaller than 100; we'll do best effort

    unique_labels = list(doc_groups.keys())
    num_labels = len(unique_labels)
    portion_per_label = desired_total // num_labels

    selected_doc_ids = []
    for lbl in unique_labels:
        docs_for_label = doc_groups[lbl]

        # If fewer docs than needed, take them all; else sample at random
        if len(docs_for_label) <= portion_per_label:
            selected_doc_ids.extend([d["doc_id"] for d in docs_for_label])
        else:
            random.shuffle(docs_for_label)
            selected_doc_ids.extend([d["doc_id"] for d in docs_for_label[:portion_per_label]])

    # If we haven't reached desired_total because some groups were small, we can fill
    # from leftover docs. For brevity, let's skip that step. But it's easy to add if needed.

    print(f"Selected {len(selected_doc_ids)} docs in a naive 'balanced' approach.")

    # 5) You can then retrieve the actual doc objects in that subset if desired
    selected_docs = []
    for doc in docs_with_3plus_claims:
        if str(doc["doc_id"]) in selected_doc_ids:
            selected_docs.append(doc)

    # Print a short summary
    print("\nSummary of selected docs:")
    for doc in selected_docs[:5]:
        print(f"- doc_id={doc['doc_id']}, #claims={len(doc['claims'])}, #abstract_sents={len(doc['abstract_sents'])}")

    # 6) (Optionally) Save the subset to a new JSON for your "Phase 2" experiments
    #   so you can run generation & verification on these selected docs.
    output_subset_path = "selected_scifact_subset.json"
    with open(output_subset_path, "w", encoding="utf-8") as f_out:
        json.dump(selected_docs, f_out, indent=2)

    print(f"\nDone. Saved subset of docs to {output_subset_path}")


if __name__ == "__main__":
    main()
