import os
import json
from pathlib import Path
from typing import List, Dict, Any

from claim_generator import (
    ModelConfig,
    ModelType,
    PromptTemplate,
    create_generator
)

# Load API key from environment (.env should be loaded externally if needed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"

# File paths
INPUT_JSON = "../data/pubmed_abstracts_with_ner.json"
OUTPUT_JSON = "../data/pubmed_abstracts_with_claims.json"


def load_annotated_abstracts(json_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


#############################
#   PROMPT GENERATION LOGIC #
#############################

def get_biomedical_prompt(text: str) -> str:
    return f"""We define a claim as an elementary statement that no longer needs to be further split. 
Your task is to generate claims specific to biomedical texts. 
Additionally, resolve any co-references: do not use pronouns; instead, use the canonical name or entity.

Please produce valid JSON, with a top-level "claims" array, no extra commentary.

SOURCE TEXT:
{text}

OUTPUT:
"""


def get_entity_aware_prompt(text: str, entities: List[Dict[str, Any]]) -> str:
    entity_list = [f"- {ent['word']} (type: {ent['entity_group']})" for ent in entities]
    entity_str = "\n".join(entity_list)
    return f"""Below is some biomedical text and a list of key entities found in it.
Make sure to reference these entities explicitly (no pronouns). 
Resolve co-references so that each entity is called by its canonical name.

Entities:
{entity_str}

SOURCE TEXT:
{text}

Please return valid JSON with a top-level "claims" array.
"""


def get_relation_aware_prompt(text: str, relations: List[Dict[str, Any]]) -> str:
    relation_list = [
        f"- ({rel['subject']} {rel['relation']} {rel['object']}; conf: {rel['confidence']})"
        for rel in relations
    ]
    relations_str = "\n".join(relation_list)
    return f"""Below is some biomedical text and a list of extracted relations.
Try to incorporate these relations into your claims (if supported by the text).
Ensure co-reference resolution (use explicit subjects).

Relations:
{relations_str}

SOURCE TEXT:
{text}

Please return valid JSON with a top-level "claims" array.
"""


def get_kg_prompt_with_entities(text: str, entities: List[Dict[str, Any]]) -> str:
    entity_list = [f"- {ent['word']} (type: {ent['entity_group']})" for ent in entities]
    entity_str = "\n".join(entity_list)
    return f"""Use a knowledge-graph style approach to generate claims from this text.
These key entities were identified:
{entity_str}

Ensure you resolve co-references and produce a final list of short claims 
(with no pronouns) in valid JSON under "claims". 

SOURCE TEXT:
{text}
"""


def get_kg_prompt_with_relations(text: str, relations: List[Dict[str, Any]]) -> str:
    relation_list = [
        f"- {rel['subject']} {rel['relation']} {rel['object']} (conf: {rel['confidence']})"
        for rel in relations
    ]
    relations_str = "\n".join(relation_list)
    return f"""Use a knowledge-graph style approach to generate claims from this text.
We have some known relations:
{relations_str}

Ensure co-reference resolution in each claim. 
Return valid JSON with "claims".

SOURCE TEXT:
{text}
"""


###########################
#  CLAIM GENERATION LOGIC #
###########################

def generate_strategy_claims(strategy: str, text: str, entities: List[Dict[str, Any]],
                             relations: List[Dict[str, Any]]) -> List[str]:
    """
    Generates claims for a single strategy.
    """
    # Use the OPENAI config for non-KG strategies.
    if strategy in ["default_prompt", "biomed_specialized", "entities_aware", "relations_aware"]:
        config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name_or_path=MODEL_NAME,
            api_key=OPENAI_API_KEY,
        )
    else:
        config = ModelConfig(
            model_type=ModelType.KG_TO_CLAIMS,
            model_name_or_path=MODEL_NAME,
            api_key=OPENAI_API_KEY,
        )

    generator = create_generator(config, PromptTemplate.DEFAULT)

    if strategy == "default_prompt":
        prompt = text
    elif strategy == "biomed_specialized":
        prompt = get_biomedical_prompt(text)
    elif strategy == "entities_aware":
        prompt = get_entity_aware_prompt(text, entities)
    elif strategy == "relations_aware":
        prompt = get_relation_aware_prompt(text, relations)
    elif strategy == "kg_based":
        prompt = text
    elif strategy == "kg_based_entities":
        prompt = get_kg_prompt_with_entities(text, entities)
    elif strategy == "kg_based_relations":
        prompt = get_kg_prompt_with_relations(text, relations)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    claims = generator.generate_claims([prompt])[0]
    return claims


def generate_claims_for_abstract(entry: Dict[str, Any], strategies: Dict[str, bool], override_existing: bool) -> Dict[
    str, List[str]]:
    """
    For a given abstract entry, generate claims for the enabled strategies.
    Only generate claims for a strategy if:
      - override_existing is True, OR
      - The entry does not already have claims for that strategy.
    Returns an updated dictionary of claims.
    """
    text = entry.get("abstract", "").strip()
    if not text:
        return {}

    entities = entry.get("entities", [])
    relations = entry.get("relations", [])

    # Use consistent strategy keys for existing claims.
    existing_claims = entry.get("claims", {})

    for strategy, enabled in strategies.items():
        if not enabled:
            continue

        # Only generate for a strategy if it does not already exist (unless overriding)
        if (not override_existing) and (strategy in existing_claims and existing_claims[strategy]):
            print(f"Skipping strategy '{strategy}' for PMID={entry.get('pmid')} (already exists).")
            continue

        print(f"Generating claims for strategy '{strategy}' for PMID={entry.get('pmid')}.")
        try:
            generated = generate_strategy_claims(strategy, text, entities, relations)
            existing_claims[strategy] = generated
        except Exception as e:
            print(f"Error generating claims for strategy '{strategy}' for PMID={entry.get('pmid')}: {e}")

    return existing_claims


def main():
    # ----------------------------
    # User-defined variables:
    # ----------------------------
    NUM_ABSTRACTS_TO_PROCESS = 2  # set to None to process all abstracts
    DUMP_FREQUENCY = 10  # dump output every 10 processed abstracts
    OVERRIDE_EXISTING = False  # if False, only generate for missing strategies

    # Toggle which strategies to run:
    strategies = {
        "default_prompt": True,
        "biomed_specialized": False,
        "entities_aware": False,
        "relations_aware": False,
        "kg_based": False,
        "kg_based_entities": False,
        "kg_based_relations": False,
    }

    print("Loading annotated abstracts...")
    abstracts = load_annotated_abstracts(INPUT_JSON)
    print(f"Loaded {len(abstracts)} records from {INPUT_JSON}.")

    # Load existing entries if available, keyed by PMID for fast lookup.
    output_path = Path(OUTPUT_JSON)
    existing_entries_dict = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing_entries = json.load(f)
        for entry in existing_entries:
            pmid = entry.get("pmid")
            if pmid:
                existing_entries_dict[pmid] = entry
        print(f"Found {len(existing_entries_dict)} existing entries.")
    else:
        existing_entries = []

    processed_count = 0

    # Process abstracts
    for entry in abstracts:
        pmid = entry.get("pmid")
        if not pmid:
            continue

        # Retrieve an existing entry if available
        existing_entry = existing_entries_dict.get(pmid, {})
        current_claims = existing_entry.get("claims", {})

        # Generate claims only for missing or to-be-overridden strategies.
        new_claims = generate_claims_for_abstract(entry, strategies, OVERRIDE_EXISTING)

        # Instead of checking truthiness, check if any enabled strategy key is present
        if not any(strategy in new_claims for strategy in strategies if strategies[strategy]):
            print(f"No new strategies to process for PMID={pmid}. Skipping.")
            continue

        # Merge existing claims with new claims (new_claims overwrite if present)
        updated_claims = {**current_claims, **new_claims}
        entry["claims"] = updated_claims
        existing_entries_dict[pmid] = entry

        processed_count += 1

        # Dump progress every DUMP_FREQUENCY processed abstracts
        if processed_count % DUMP_FREQUENCY == 0:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(list(existing_entries_dict.values()), f, indent=2, ensure_ascii=False)
            print(f"Dumped progress after {processed_count} processed abstracts.")

        # Stop processing if we've reached the limit
        if NUM_ABSTRACTS_TO_PROCESS is not None and processed_count >= NUM_ABSTRACTS_TO_PROCESS:
            break

    # Final dump of all results
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(list(existing_entries_dict.values()), f, indent=2, ensure_ascii=False)
    print(f"Done! Wrote enriched data with claims to {output_path}")


if __name__ == "__main__":
    main()
