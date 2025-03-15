from typing import List

DEFAULT_PROMPT = r"""
We define a claim as an "elementary information unit in a sentence, 
which no longer needs to be further split."

For example, given the following sentence:
INPUT:
"NASA’s Perseverance rover has discovered ancient microbial life on Mars 
according to a recent study published in the journal Science. 
It established a set of new paradigms for space exploration"

OUTPUT:
{{"claims": [
  "NASA’s Perseverance rover discovered ancient microbial life.",
  "Ancient microbial life was discovered on Mars.",
  "The discovery was made according to a recent study.",
  "The study was published in the journal Science.",
  "The study established a set of new paradigms for space exploration."
]}}

Recommendations:
1) Use nouns as subjects (avoid pronouns).
2) Do not invent new words; remain faithful to the input.
3) Output must be valid JSON without any extra commentary.
4) Each distinct fact from the input must form its own claim.
5) Ensure that the JSON is well-formed.

Now, do the same for this input:
INPUT:
{SOURCE_TEXT}
OUTPUT:
"""


def get_default_prompt(source_text: str) -> str:
    return DEFAULT_PROMPT.format(SOURCE_TEXT=source_text)


# Robust Biomedical Prompt
BIOMED_PROMPT = r"""
We define a claim as an "elementary information unit in a sentence, 
which no longer needs to be further split." Follow these principles:

Core Guidelines (from the default prompt):
1) Use nouns as subjects and avoid pronouns.
2) Do not introduce novel words; remain faithful to the input.
3) Output valid JSON with a top-level "claims" array.
4) Each fact in the input must appear as a separate claim.
5) Ensure the JSON is well-formed.

Additional Biomedical Instructions:
1) Accurately represent domain-specific terminology.
2) Resolve co-references by replacing pronouns with explicit entity names.
3) Emphasize medically relevant details from the text.

Now, generate claims for the input below:
INPUT:
{SOURCE_TEXT}
OUTPUT:
"""


def get_biomedical_prompt(source_text: str) -> str:
    return BIOMED_PROMPT.format(SOURCE_TEXT=source_text)


# Robust Entity-Aware Prompt
ENTITY_AWARE_PROMPT = r"""
We define a claim as an "elementary information unit in a sentence, 
which no longer needs to be further split." Adhere to the following guidelines:

Core Guidelines (from the default prompt):
1) Use nouns as subjects and avoid pronouns.
2) Do not invent new words; follow the input text strictly.
3) Provide valid JSON with a "claims" array.
4) Split the input into distinct claims, each representing a fact.
5) Ensure the JSON is correctly formatted.

Additional Entity-Aware Instructions:
1) The following key entities have been identified in the text:
{ENTITY_LIST}
2) Explicitly include these entities in your claims; do not use pronouns.
3) Ensure that each claim clearly references one or more of these entities.

Now, generate claims for the input below:
INPUT:
{SOURCE_TEXT}
OUTPUT:
"""


def get_entity_aware_prompt(source_text: str, entities: list) -> str:
    entity_list = "\n".join(f"- {ent['word']} (type: {ent['entity_group']})" for ent in entities)
    return ENTITY_AWARE_PROMPT.format(SOURCE_TEXT=source_text, ENTITY_LIST=entity_list)


# Robust Relation-Aware Prompt
RELATION_AWARE_PROMPT = r"""
We define a claim as an "elementary information unit in a sentence, 
which no longer needs to be further split." Follow these core guidelines:

Core Guidelines (from the default prompt):
1) Use clear, explicit subjects; avoid pronouns.
2) Remain strictly faithful to the input text.
3) Output must be valid JSON with a top-level "claims" array.
4) Each claim should represent one distinct fact.
5) The JSON output must be well-formed.

Additional Relation-Aware Instructions:
1) Incorporate the following relations into your claims:
{RELATION_LIST}
2) Translate each relation into an independent claim when possible.
3) Resolve any co-references and be explicit about subjects and objects.

Now, generate claims for the input below:
INPUT:
{SOURCE_TEXT}
OUTPUT:
"""


def get_relation_aware_prompt(source_text: str, relations: list) -> str:
    relation_list = "\n".join(
        f"- {rel['subject']} {rel['relation']} {rel['object']} (conf: {rel.get('confidence', 'N/A')})"
        for rel in relations
    )
    return RELATION_AWARE_PROMPT.format(SOURCE_TEXT=source_text, RELATION_LIST=relation_list)


def get_kg_parser_prompt_with_entities(source_text: str, entities: List[dict]) -> str:
    """
    A custom KG parser prompt that includes known entities.
    We'll embed them in the instructions so the parser
    tries to incorporate or align them.
    """
    entity_list = "\n".join(f"- {ent['word']} (type: {ent['entity_group']})" for ent in entities)

    return f"""
("system",
""
You are an expert at extracting a knowledge graph from text.
1) Identify all entities (including these known entities):
{entity_list}

2) Avoid duplicates by merging co-referent entities.

3) For each relationship, produce a triple: [subject, relation, object].
Output only valid JSON: {{ "triples": [...] }}
""
)

("human",
""
The raw text to parse is:
{source_text}
""
)
""".strip()


def get_kg_parser_prompt_with_relations(source_text: str, relations: List[dict]) -> str:
    """
    A custom KG parser prompt that includes known or pre-extracted relations.
    We'll embed them so the parser tries to incorporate them if correct.
    """
    relation_list = "\n".join(
        f"- {rel['subject']} {rel['relation']} {rel['object']} (conf: {rel.get('confidence','N/A')})"
        for rel in relations
    )

    return f"""
("system",
""
Extract a knowledge graph from the text. We already suspect some relations:
{relation_list}

Please confirm or refine them. Each triple is [subject, relation, object].
Output valid JSON with a 'triples' key and no extra text.
""
)

("human",
""
Text:
{source_text}
""
)
""".strip()
