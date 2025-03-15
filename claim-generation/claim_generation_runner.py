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

# File paths
INPUT_JSON = "../data/pubmed_abstracts_with_ner.json"
OUTPUT_JSON = "../data/pubmed_abstracts_with_claims.json"


def load_annotated_abstracts(json_path: str) -> List[Dict[str, Any]]:
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

    # KGToClaimsGenerator must accept kg_prompt_builder in its constructor.
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
    # 1) Build the base config for the strategy.
    if strategy in ["default_prompt", "biomed_specialized", "entities_aware", "relations_aware"]:
        config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name_or_path=MODEL_NAME,
            api_key=OPENAI_API_KEY,
        )
    else:
        # For KG-based strategies
        config = ModelConfig(
            model_type=ModelType.KG_TO_CLAIMS,
            model_name_or_path=MODEL_NAME,
            api_key=OPENAI_API_KEY,
        )

    # 2) Generate claims according to strategy
    if strategy == "default_prompt":
        prompt = get_default_prompt(text)
        generator = create_generator(config, PromptTemplate.DEFAULT)
        claims = generator.generate_claims([prompt])[0]

    elif strategy == "biomed_specialized":
        prompt = get_biomedical_prompt(text)
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
        # Use default KG parser logic (no extra entity/relation hints)
        generator = create_generator(config, PromptTemplate.DEFAULT)
        claims = generator.generate_claims([text])[0]

    elif strategy == "kg_based_entities":
        # Use a custom KG parser prompt that includes known entities
        kg_generator = create_kg_generator_with_entities_or_relations(
            model_config=config,
            entities=entities,
            relations=None,
            mode="entities"
        )
        claims = kg_generator.generate_claims([text])[0]

    elif strategy == "kg_based_relations":
        # Use a custom KG parser prompt that includes known relations
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


def generate_claims_for_abstract(
    entry: Dict[str, Any],
    strategies: Dict[str, bool],
    override_existing: bool
) -> Dict[str, List[str]]:
    """
    Given one abstract, run each strategy to generate claims.
    """
    text = entry.get("abstract", "").strip()
    if not text:
        return {}

    entities = entry.get("entities", [])
    relations = entry.get("relations", [])
    existing_claims = entry.get("claims", {})

    for strategy, enabled in strategies.items():
        if not enabled:
            continue

        # Skip if claims exist and we do not want to override
        if (not override_existing) and (strategy in existing_claims and existing_claims[strategy]):
            print(f"Skipping strategy '{strategy}' for PMID={entry.get('pmid')} (already exists).")
            continue

        print(f"Generating claims for strategy '{strategy}' for PMID={entry.get('pmid')}.")
        try:
            new_claims = generate_strategy_claims(strategy, text, entities, relations)
            existing_claims[strategy] = new_claims
        except Exception as e:
            raise ValueError(
                f"Error generating claims for strategy '{strategy}' for PMID={entry.get('pmid')}: {e}"
            )

    return existing_claims


def main():
    """
    Main entry point for running claim generation.
    """
    NUM_ABSTRACTS_TO_PROCESS = 100
    DUMP_FREQUENCY = 10
    OVERRIDE_EXISTING = False

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

    processed_count = 0
    for entry in abstracts:
        pmid = entry.get("pmid")
        if not pmid:
            continue

        # Retrieve existing entry if it exists
        existing_entry = existing_entries_dict.get(pmid, {})
        current_claims = existing_entry.get("claims", {})

        new_claims = generate_claims_for_abstract(entry, strategies, OVERRIDE_EXISTING)

        # If no new strategies were generated, skip
        if not any(strategy in new_claims for strategy in strategies if strategies[strategy]):
            print(f"No new strategies to process for PMID={pmid}. Skipping.")
            continue

        updated_claims = {**current_claims, **new_claims}
        entry["claims"] = updated_claims
        existing_entries_dict[pmid] = entry

        processed_count += 1

        # Periodically save partial results
        if processed_count % DUMP_FREQUENCY == 0:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(list(existing_entries_dict.values()), f, indent=2, ensure_ascii=False)
            print(f"Dumped progress after {processed_count} processed abstracts.")

        # If we reached our max, stop
        if NUM_ABSTRACTS_TO_PROCESS is not None and processed_count >= NUM_ABSTRACTS_TO_PROCESS:
            break

    # Final save
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(list(existing_entries_dict.values()), f, indent=2, ensure_ascii=False)
    print(f"Done! Wrote enriched data with claims to {output_path}")


if __name__ == "__main__":
    main()
