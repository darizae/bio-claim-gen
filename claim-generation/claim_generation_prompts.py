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


BIOMED_PROMPT = r"""
We define a claim as an "elementary information unit in a sentence, 
which no longer needs to be further split." Specialized for biomedical contexts.

For example, given the following sentence:
INPUT:
"Metformin has been shown to significantly reduce blood glucose levels in type 2 diabetes,
according to a recent clinical trial published in The Lancet.
It also demonstrated a favorable safety profile in patients."

OUTPUT:
{{"claims": [
  "Metformin significantly reduces blood glucose levels in type 2 diabetes.",
  "The effect is supported by a recent clinical trial.",
  "The trial was published in The Lancet.",
  "Metformin demonstrated a favorable safety profile in patients."
]}}

Recommendations (adapted from the default):
1) Use nouns as subjects (avoid pronouns).
2) Do not invent new words; remain faithful to the input.
3) Output must be valid JSON without any extra commentary.
4) Each distinct fact in the text must be its own claim.
5) Ensure the JSON is well-formed.

Additional Biomedical Instructions:
1) Accurately represent domain-specific terminology (e.g., drug names, diseases, biomarkers).
2) Resolve co-references by replacing pronouns ("it," "they," etc.) with explicit entities.
3) Emphasize medically relevant details (like dosages, outcomes, side effects).

Now do the same for this input:

INPUT:
{SOURCE_TEXT}

OUTPUT:
"""


def get_biomedical_prompt(source_text: str) -> str:
    return BIOMED_PROMPT.format(SOURCE_TEXT=source_text)


ENTITY_AWARE_PROMPT = r"""
We define a claim as an "elementary information unit in a sentence, 
which no longer needs to be further split." Adhere to the following guidelines:

For example, given the following sentence:
INPUT:
"Barack Obama was the 44th President of the United States.
He was born in Hawaii and studied at Harvard University."

We have identified these entities:
- "Barack Obama" (type: PERSON)
- "44th President" (type: TITLE)
- "Hawaii" (type: LOCATION)
- "Harvard University" (type: ORGANIZATION)

OUTPUT:
{{"claims": [
  "Barack Obama served as the 44th President of the United States.",
  "Barack Obama was born in Hawaii.",
  "Barack Obama studied at Harvard University."
]}}

Core Guidelines (from the default prompt):
1) Use nouns as subjects and avoid pronouns.
2) Do not invent new words; remain faithful to the input.
3) Output must be valid JSON with a top-level "claims" array.
4) Each distinct fact from the input must appear as a separate claim.
5) Ensure the JSON is well-formed.

Additional Entity-Aware Instructions:
1) The following key entities have been identified in the text:
{ENTITY_LIST}
2) Explicitly reference these entities; do not substitute pronouns.
3) Ensure that each claim clearly references at least one of these entities.

Now, generate claims for the input below:
INPUT:
{SOURCE_TEXT}
OUTPUT:
"""


def get_entity_aware_prompt(source_text: str, entities: list) -> str:
    """
    Build the entity-aware prompt, injecting the identified entities.
    """
    entity_list = "\n".join(f"- {ent['word']} (type: {ent['entity_group']})" for ent in entities)
    return ENTITY_AWARE_PROMPT.format(SOURCE_TEXT=source_text, ENTITY_LIST=entity_list)


RELATION_AWARE_PROMPT = r"""
We define a claim as an "elementary information unit in a sentence, 
which no longer needs to be further split." Follow these guidelines:

For example, given the following sentence:
INPUT:
"Joe took 200 mg of Ibuprofen to relieve a headache. 
He reported that this medication was effective."

We have identified this relation:
- (Joe -> took -> Ibuprofen) (conf: 0.95)

OUTPUT:
{{"claims": [
  "Joe took 200 mg of Ibuprofen to relieve a headache.",
  "Ibuprofen was effective for Joe’s headache."
]}}

Core Guidelines (from the default prompt):
1) Use clear, explicit subjects; avoid pronouns.
2) Remain strictly faithful to the input text.
3) Output must be valid JSON with a top-level "claims" array.
4) Each claim should represent one distinct fact.
5) The JSON output must be well-formed.

Additional Relation-Aware Instructions:
1) Incorporate the following relations into your claims:
{RELATION_LIST}
2) Translate each relation into an independent claim if possible.
3) Resolve any co-references and be explicit about subjects and objects.

Now, generate claims for the input below:
INPUT:
{SOURCE_TEXT}
OUTPUT:
"""


def get_relation_aware_prompt(source_text: str, relations: list) -> str:
    """
    Build the relation-aware prompt, injecting known or extracted relations.
    """
    relation_list = "\n".join(
        f"- {rel['subject']} {rel['relation']} {rel['object']} (conf: {rel.get('confidence', 'N/A')})"
        for rel in relations
    )
    return RELATION_AWARE_PROMPT.format(SOURCE_TEXT=source_text, RELATION_LIST=relation_list)


def get_kg_parser_prompt_with_entities(source_text: str, entities: List[dict]) -> str:
    """
    A custom KG parser prompt that includes known entities. We embed them so
    the parser tries to incorporate them in the KG (if they appear in the text).
    """
    entity_list_str = "\n".join(
        f"- {ent['word']} (type: {ent['entity_group']})"
        for ent in entities
    )

    return f"""
("system",
""
You are an expert at extracting information in structured formats to build a knowledge graph.
Step 1 – Entity detection: Identify all entities in the raw text. Entities should be basic and unambiguous, similar to Wikipedia nodes.
Additionally, please incorporate or confirm the following known entities:
{entity_list_str}

Step 2 – Coreference resolution: Merge expressions referring to the same entity to avoid duplicates.
Step 3 – Relation extraction: Identify semantic relationships between the entities.

Format: Return the knowledge graph as a JSON object with a single key \\"triples\\".
The value must be a list of triples, each triple = [subject, relation, object],
for example: [\\"entity 1\\", \\"relation\\", \\"entity 2\\"].

The output must be valid JSON and include nothing but the JSON object.
""
,
"human",
"Use the above instructions to extract a knowledge graph from the input. Return only the JSON object without any extra text or commentary."
,
"human",
""
Important Tips:
- Each triple must contain exactly three non-empty strings.
- Do not add extra commentary outside the JSON.
- Validate that the JSON output is well-formed.
""
),
("human",
""
Here are some example input and output pairs.

Example 1.
Input:
\\"The Walt Disney Company, commonly known as Disney, is an American multinational mass media and entertainment conglomerate that is headquartered at the Walt Disney Studios complex in Burbank, California.\\"
Output:
{{
  "triples": [
    ["The Walt Disney Company", "headquartered at", "Walt Disney Studios complex in Burbank, California"],
    ["The Walt Disney Company", "commonly known as", "Disney"],
    ["The Walt Disney Company", "instance of", "American multinational mass media and entertainment conglomerate"]
  ]
}}

Example 2.
Input:
\\"Amanda Jackson was born in Springfield, Ohio, USA on June 1, 1985. She was a basketball player for the U.S. women’s team.\\"
Output:
{{
  "triples": [
    ["Amanda Jackson", "born in", "Springfield, Ohio, USA"],
    ["Amanda Jackson", "born on", "June 1, 1985"],
    ["Amanda Jackson", "occupation", "basketball player"],
    ["Amanda Jackson", "played for", "U.S. women’s basketball team"]
  ]
}}

Now, process the following input:
<input>{source_text}</input>
""
)
""".strip()


def get_kg_parser_prompt_with_relations(source_text: str, relations: List[dict]) -> str:
    """
    A custom KG parser prompt that includes known or pre-extracted relations.
    We'll embed them so the parser tries to incorporate them if correct.
    """
    relation_list = "\n".join(
        f"- {rel['subject']} {rel['relation']} {rel['object']} (conf: {rel.get('confidence', 'N/A')})"
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
