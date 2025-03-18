import os
import json
from pathlib import Path
from typing import List, Dict, Any

# Import from our local claim_generator
from claim_generator import (
    ModelConfig,
    ModelType,
    PromptTemplate,
    create_generator
)
from claim_generator.generator import KGToClaimsGenerator

# Import your local prompt-building functions
from claim_generation_prompts import (
    get_default_prompt,
    get_biomedical_prompt,
    get_entity_aware_prompt,
    get_relation_aware_prompt,
    get_kg_parser_prompt_with_entities,
    get_kg_parser_prompt_with_relations
)

# Load API key from environment (.env should be loaded externally if needed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"

# File paths for the new JSON file structure and output files
INPUT_JSON = "../data/scifact/selected_scifact_subset.json"
OUTPUT_JSON = "../data/selected_scifact_subset_with_claims.json"
FLAGGED_JSON = "../data/flagged_claim_generation_entries.json"


def load_entries(json_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def create_kg_generator_with_entities_or_relations(
        model_config: ModelConfig,
        entities: List[dict],
        relations: List[dict],
        mode: str
):
    """
    Returns a KGToClaimsGenerator that uses a custom KG parser prompt
    for entity or relation awareness.
    """
    if mode == "entities":
        def builder_fn(text: str) -> str:
            return get_kg_parser_prompt_with_entities(text, entities)
    elif mode == "relations":
        def builder_fn(text: str) -> str:
            return get_kg_parser_prompt_with_relations(text, relations)
    else:
        builder_fn = None  # will use the default KG prompt internally

    return KGToClaimsGenerator(model_config, kg_prompt_builder=builder_fn)


def generate_strategy_claims(
        strategy: str,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
) -> List[str]:
    """
    Orchestrates which config/prompt to use for a given strategy.
    """
    # Set up config based on strategy
    if strategy in ["default_prompt", "entities_aware", "relations_aware"]:
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

    if strategy == "default_prompt":
        prompt = get_default_prompt(text)
        generator = create_generator(config, PromptTemplate.DEFAULT)
        claims = generator.generate_claims([prompt])[0]

    elif strategy == "entities_aware":
        prompt = get_entity_aware_prompt(text, entities)
        generator = create_generator(config, PromptTemplate.DEFAULT)
        claims = generator.generate_claims([prompt])[0]

    elif strategy == "relations_aware":
        prompt = get_relation_aware_prompt(text, relations)
        generator = create_generator(config, PromptTemplate.DEFAULT)
        claims = generator.generate_claims([prompt])[0]

    elif strategy == "kg_based":
        generator = create_generator(config, PromptTemplate.DEFAULT)
        claims = generator.generate_claims([text])[0]

    elif strategy == "kg_based_entities":
        kg_generator = create_kg_generator_with_entities_or_relations(
            model_config=config,
            entities=entities,
            relations=None,
            mode="entities"
        )
        claims = kg_generator.generate_claims([text])[0]

    elif strategy == "kg_based_relations":
        kg_generator = create_kg_generator_with_entities_or_relations(
            model_config=config,
            entities=None,
            relations=relations,
            mode="relations"
        )
        claims = kg_generator.generate_claims([text])[0]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return claims


def generate_claims_for_entry(
        entry: Dict[str, Any],
        strategies: Dict[str, bool],
        override_existing: bool
) -> (Dict[str, List[str]], List[Dict[str, Any]]):
    """
    Given one entry, run each enabled strategy to generate claims.
    Returns a tuple: (claims dictionary, list of flagged errors [if any]).
    """
    # Use abstract_raw if available; otherwise fallback to joining abstract_sents
    text = entry.get("abstract_raw", "").strip()
    if not text and "abstract_sents" in entry:
        text = " ".join(entry.get("abstract_sents", [])).strip()
    if not text:
        return {}, []

    entities = entry.get("entities", [])
    relations = entry.get("relations", [])

    # Ensure that claims is a dict, not a list
    existing_claims = entry.get("claims", {})
    if not isinstance(existing_claims, dict):
        print(f"Warning: claims for doc_id {entry.get('doc_id')} are not a dict. Resetting to empty dict.")
        existing_claims = {}

    flagged = []

    for strategy, enabled in strategies.items():
        if not enabled:
            continue

        if (not override_existing) and (strategy in existing_claims and existing_claims[strategy]):
            print(f"Skipping strategy '{strategy}' for doc_id={entry.get('doc_id')} (already exists).")
            continue

        print(f"Generating claims for strategy '{strategy}' for doc_id={entry.get('doc_id')}.")
        try:
            new_claims = generate_strategy_claims(strategy, text, entities, relations)
            existing_claims[strategy] = new_claims
        except Exception as e:
            error_detail = {
                "doc_id": entry.get("doc_id"),
                "strategy": strategy,
                "error": str(e)
            }
            flagged.append(error_detail)
            print(f"Error in strategy '{strategy}' for doc_id={entry.get('doc_id')}: {e}")

    return existing_claims, flagged


def main():
    # Configuration parameters
    NUM_ENTRIES_TO_PROCESS = None
    DUMP_FREQUENCY = 10
    OVERRIDE_EXISTING = False

    # Define which strategies to enable. (Set to True for strategies you wish to run.)
    strategies = {
        "default_prompt": False,
        "entities_aware": False,
        "relations_aware": False,
        "kg_based": True,
        # "kg_based_entities": False,
        # "kg_based_relations": False,
    }

    flagged_path = Path(FLAGGED_JSON)
    flagged_entries = []

    # Load existing flagged entries if the file exists
    if flagged_path.exists():
        with flagged_path.open("r", encoding="utf-8") as f:
            try:
                flagged_entries = json.load(f)
                print(f"Loaded {len(flagged_entries)} existing flagged entries.")
            except json.JSONDecodeError:
                print("Existing flagged file is empty or corrupted. Starting fresh.")
                flagged_entries = []

    print("Loading entries...")
    entries = load_entries(INPUT_JSON)
    print(f"Loaded {len(entries)} records from {INPUT_JSON}.")

    output_path = Path(OUTPUT_JSON)

    # Load existing entries if output file exists
    existing_entries_dict = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing_entries = json.load(f)
        for entry in existing_entries:
            doc_id = entry.get("doc_id")
            if doc_id:
                existing_entries_dict[doc_id] = entry
        print(f"Found {len(existing_entries_dict)} existing entries.")

    processed_count = 0
    for entry in entries:
        doc_id = entry.get("doc_id")
        if not doc_id:
            continue

        existing_entry = existing_entries_dict.get(doc_id, {})
        current_claims = existing_entry.get("claims", {})

        # Ensure current_claims is a dictionary; if not, reset it.
        if not isinstance(current_claims, dict):
            print(f"Warning: claims for doc_id {doc_id} are not a dict. Resetting to empty dict.")
            current_claims = {}

        new_claims, flagged = generate_claims_for_entry(entry, strategies, OVERRIDE_EXISTING)

        # Merge new claims with existing ones
        updated_claims = {**current_claims, **new_claims}
        entry["claims"] = updated_claims
        existing_entries_dict[doc_id] = entry

        # Append any flagged errors for diagnostic
        if flagged:
            flagged_entries.extend(flagged)

        processed_count += 1

        if processed_count % DUMP_FREQUENCY == 0:
            # Save progress for enriched entries
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(list(existing_entries_dict.values()), f, indent=2, ensure_ascii=False)
            # Save flagged entries for diagnostic purposes (appending to the existing file)
            with flagged_path.open("w", encoding="utf-8") as f:
                json.dump(flagged_entries, f, indent=2, ensure_ascii=False)
            print(f"Dumped progress after processing {processed_count} entries.")

        if NUM_ENTRIES_TO_PROCESS is not None and processed_count >= NUM_ENTRIES_TO_PROCESS:
            break

    # Final dump of enriched entries and flagged errors
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(list(existing_entries_dict.values()), f, indent=2, ensure_ascii=False)
    with flagged_path.open("w", encoding="utf-8") as f:
        json.dump(flagged_entries, f, indent=2, ensure_ascii=False)
    print(f"Done! Wrote enriched data with claims to {output_path}")
    print(f"Flagged entries written to {flagged_path}")


if __name__ == "__main__":
    main()
