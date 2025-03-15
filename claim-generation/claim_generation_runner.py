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

def generate_claims_for_abstract(entry: Dict[str, Any], strategies: Dict[str, bool]) -> Dict[str, List[str]]:
    """
    Given one abstract entry, run selected generation strategies.
    """
    text = entry.get("abstract", "").strip()
    if not text:
        return {}

    entities = entry.get("entities", [])
    relations = entry.get("relations", [])

    claims_results = {}

    # All strategies use the same base config
    base_config = ModelConfig(
        model_type=ModelType.OPENAI,
        model_name_or_path=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0.2
    )

    # (A) DEFAULT PROMPT
    if strategies.get("default", False):
        default_generator = create_generator(base_config, PromptTemplate.DEFAULT)
        default_claims = default_generator.generate_claims([text])[0]
        claims_results["default_prompt"] = default_claims

    # (B) BIOMED SPECIALIZED PROMPT
    if strategies.get("biomed_specialized", False):
        specialized_generator = create_generator(base_config, PromptTemplate.DEFAULT)
        specialized_prompt = get_biomedical_prompt(text)
        specialized_claims = specialized_generator.generate_claims([specialized_prompt])[0]
        claims_results["biomed_specialized"] = specialized_claims

    # (C) ENTITIES-AWARE
    if strategies.get("entities_aware", False):
        entity_prompt_generator = create_generator(base_config, PromptTemplate.DEFAULT)
        entity_prompt = get_entity_aware_prompt(text, entities)
        entity_prompt_claims = entity_prompt_generator.generate_claims([entity_prompt])[0]
        claims_results["entities_aware"] = entity_prompt_claims

    # (D) RELATIONS-AWARE
    if strategies.get("relations_aware", False):
        rel_prompt_generator = create_generator(base_config, PromptTemplate.DEFAULT)
        rel_prompt = get_relation_aware_prompt(text, relations)
        rel_prompt_claims = rel_prompt_generator.generate_claims([rel_prompt])[0]
        claims_results["relations_aware"] = rel_prompt_claims

    # (E) KG-BASED (WITHOUT EXTRA ENTITIES/RELATIONS)
    if strategies.get("kg_based", False):
        kg_config = ModelConfig(
            model_type=ModelType.KG_TO_CLAIMS,
            model_name_or_path=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0.2
        )
        kg_generator = create_generator(kg_config, PromptTemplate.DEFAULT)
        kg_claims = kg_generator.generate_claims([text])[0]
        claims_results["kg_based"] = kg_claims

    # (F) KG-BASED + ENTITIES
    if strategies.get("kg_based_entities", False):
        kg_entities_config = ModelConfig(
            model_type=ModelType.KG_TO_CLAIMS,
            model_name_or_path=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0.2
        )
        kg_entities_generator = create_generator(kg_entities_config, PromptTemplate.DEFAULT)
        kg_entities_prompt = get_kg_prompt_with_entities(text, entities)
        kg_entities_claims = kg_entities_generator.generate_claims([kg_entities_prompt])[0]
        claims_results["kg_based_entities"] = kg_entities_claims

    # (G) KG-BASED + RELATIONS
    if strategies.get("kg_based_relations", False):
        kg_relations_config = ModelConfig(
            model_type=ModelType.KG_TO_CLAIMS,
            model_name_or_path=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0.2
        )
        kg_relations_generator = create_generator(kg_relations_config, PromptTemplate.DEFAULT)
        kg_relations_prompt = get_kg_prompt_with_relations(text, relations)
        kg_relations_claims = kg_relations_generator.generate_claims([kg_relations_prompt])[0]
        claims_results["kg_based_relations"] = kg_relations_claims

    return claims_results


def main():
    # ----------------------------
    # User-defined variables:
    # ----------------------------
    NUM_ABSTRACTS_TO_PROCESS = 2  # set to None to process all abstracts
    DUMP_FREQUENCY = 10             # dump output every 10 processed abstracts
    OVERRIDE_EXISTING = True       # if False, skip abstracts that already have claims

    # Toggle which strategies to run:
    strategies = {
        "default": True,
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

    # If output file exists, load existing entries (to avoid re-processing)
    output_path = Path(OUTPUT_JSON)
    processed_pmids = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing_entries = json.load(f)
        processed_pmids = {entry.get("pmid") for entry in existing_entries if "claims" in entry}
        updated_entries = existing_entries
        print(f"Found {len(processed_pmids)} entries with existing claims.")
    else:
        updated_entries = []

    processed_count = 0

    for entry in abstracts:
        pmid = entry.get("pmid")
        if not pmid:
            continue

        # Skip if already processed and override is False
        if (pmid in processed_pmids) and (not OVERRIDE_EXISTING):
            print(f"Skipping PMID={pmid} (already processed).")
            continue

        print(f"Generating claims for PMID={pmid}...")
        claims_dict = generate_claims_for_abstract(entry, strategies)
        entry["claims"] = claims_dict

        # Replace or append the entry in our output list
        updated_entries = [e for e in updated_entries if e.get("pmid") != pmid] + [entry]

        processed_count += 1

        # Dump progress every DUMP_FREQUENCY processed abstracts
        if processed_count % DUMP_FREQUENCY == 0:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(updated_entries, f, indent=2, ensure_ascii=False)
            print(f"Dumped progress after {processed_count} abstracts.")

        # If a limit is set and reached, stop processing
        if NUM_ABSTRACTS_TO_PROCESS is not None and processed_count >= NUM_ABSTRACTS_TO_PROCESS:
            break

    # Final dump of all results
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(updated_entries, f, indent=2, ensure_ascii=False)
    print(f"Done! Wrote enriched data with claims to {output_path}")


if __name__ == "__main__":
    main()
