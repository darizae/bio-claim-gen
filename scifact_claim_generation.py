import os
import json
from pathlib import Path
from typing import List, Dict, Any

# --- Import your local code and prompts ---
from claim_generation.claim_generation_prompts import (
    get_default_prompt,
    get_biomedical_prompt,
    get_entity_aware_prompt,
    get_relation_aware_prompt,
    get_kg_parser_prompt_with_entities,
    get_kg_parser_prompt_with_relations
)
from claim_generator import (
    ModelConfig,
    ModelType,
    PromptTemplate,
    create_generator
)
from claim_generator.generator import KGToClaimsGenerator

# --- Load your OpenAI API key from environment, etc. ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"

# --- File paths: adapt to your folder structure as needed ---
INPUT_JSON  = "data/scifact/scifact_annotated.json"
OUTPUT_JSON = "data/scifact/scifact_with_claims.json"

##########################################
#         PROMPT GENERATION LOGIC        #
##########################################

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
    Re-uses your "prompt-building" logic to produce claims from a text.
    """
    # 1) Configure the LLM model for each strategy
    if strategy in ["default_prompt", "biomed_specialized", "entities_aware", "relations_aware"]:
        config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name_or_path=MODEL_NAME,
            api_key=OPENAI_API_KEY,
        )
    else:
        # For KG-based we assume a special generator
        config = ModelConfig(
            model_type=ModelType.KG_TO_CLAIMS,
            model_name_or_path=MODEL_NAME,
            api_key=OPENAI_API_KEY,
        )

    # 2) Build the prompt and call your generator
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
        # A direct KGToClaims approach with no custom prompt function
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


##########################################
#       MAIN SCRIPT FUNCTIONALITY        #
##########################################

def main():
    """
    1) Load scifact_annotated.json (each entry has doc_id, abstract_raw, entities, relations, etc.)
    2) For each doc, run multiple generation strategies, store results in "generated_claims".
    3) Save final output to scifact_with_claims.json
    """
    # --- Decide which strategies to run ---
    strategies = {
        "default_prompt": True,
        "biomed_specialized": False,
        "entities_aware": False,
        "relations_aware": False,
        "kg_based": False,
        "kg_based_entities": False,
        "kg_based_relations": False,
    }

    # For partial runs, we can limit how many docs we handle
    MAX_DOCS = None  # or some integer if you want to limit
    # If you want to skip re-generation for docs that already have claims, set to False
    OVERRIDE_EXISTING = True

    # 1) Load annotated SciFact data
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Cannot find {INPUT_JSON}. Make sure you have annotated SciFact first.")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) We will store everything in a new list for the final output
    output_data = []
    processed_count = 0

    for entry in data:
        doc_id = entry.get("doc_id")
        if not doc_id:
            continue

        # The text to feed the LLM
        text = entry.get("abstract_raw", "").strip()
        if not text:
            # Possibly skip if empty
            output_data.append(entry)
            continue

        # Entities/relations from the annotated step
        entities = entry.get("entities", [])
        relations = entry.get("relations", [])

        # We'll place new claims under "generated_claims" so we don't overwrite SciFact's own "claims"
        generated_claims = entry.get("generated_claims", {})

        # For each strategy
        for strat, enabled in strategies.items():
            if not enabled:
                continue

            # Check if we skip if there's already something
            if not OVERRIDE_EXISTING and strat in generated_claims and generated_claims[strat]:
                print(f"Skipping generation for doc_id={doc_id}, strategy={strat} (already exists).")
                continue

            print(f"Generating claims for doc_id={doc_id} with strategy={strat}...")
            try:
                claims = generate_strategy_claims(strat, text, entities, relations)
                generated_claims[strat] = claims
            except Exception as e:
                print(f"ERROR generating claims for doc_id={doc_id}, strategy={strat}: {e}")
                generated_claims[strat] = []

        # Attach back to entry
        entry["generated_claims"] = generated_claims
        output_data.append(entry)

        processed_count += 1
        if MAX_DOCS is not None and processed_count >= MAX_DOCS:
            break

    # 3) Save results
    out_path = Path(OUTPUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nDone! Generated claims for {processed_count} documents. Saved to {OUTPUT_JSON}.")


if __name__ == "__main__":
    main()
