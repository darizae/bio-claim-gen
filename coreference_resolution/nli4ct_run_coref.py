import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the API key from .env file
load_dotenv()

# Set maximum number of entries to process and the save frequency
MAX_ENTRIES = 200
SAVE_FREQ = 5

# Filenames for the new dataset
INPUT_FILENAME = "sample_nli4ct.json"
OUTPUT_FILENAME = "sample_nli4ct_coref.json"
FAILURE_FILENAME = "failed_entries_nli4ct.json"

# Enhanced prompt prefix with chain-of-thought instructions, few-shot examples,
# and a placeholder for injecting pre-extracted entities.
ENHANCED_PROMPT_PREFIX = (
    "You are a language model expert in coreference resolution and contextual analysis. "
    "Your task is to review a document's eligibility section, its associated claims, and a list of extracted entities, "
    "and then perform coreference resolution across the entire text. "
    "This means you should merge expressions and pronouns referring to the same entity, eliminate duplicates, "
    "and ensure that references in the eligibility section and in the claims are consistent. "
    "Use the provided extracted entities to help guide your resolution.\n\n"
    "Please follow these steps carefully:\n\n"
    "1. **Chain-of-Thought Reasoning:** Think through the context provided by the eligibility text and the claims. "
    "Identify all key entities and concepts, and consider how they are mentioned in different parts of the text.\n\n"
    "2. **Contextual Consistency:** Ensure that any entity mentioned in the eligibility section is consistently referred to in the claims, and vice versa.\n\n"
    "3. **Merge Duplicates:** If the same entity is referred to using different expressions (for example, 'the drug', 'it', or 'this medication'), merge them into one uniform reference.\n\n"
    "4. **Use Extracted Entities:** Below is a list of entities extracted from the eligibility text. "
    "Use these words as guidance to improve the resolution process. Do not modify the entity words; only use them to inform your processing.\n\n"
    "5. **Few-Shot Examples:**\n"
    "   - Example 1:\n"
    "     Input Text: \"A study of aspirin in patients with cardiovascular disease shows that the drug reduces heart attacks.\"\n"
    "     Input Claims: [ {{\"id\": \"1\", \"claim\": \"The medication reduces the incidence of heart attacks.\"}} ]\n"
    "     Expected Output: {{\n"
    "         \"abstract\": \"A study of aspirin in patients with cardiovascular disease shows that aspirin reduces heart attacks.\",\n"
    "         \"claims\": [ {{\"id\": \"1\", \"claim\": \"Aspirin reduces the incidence of heart attacks.\"}} ]\n"
    "     }}\n\n"
    "   - Example 2:\n"
    "     Input Text: \"Researchers found that the protein p53 plays a crucial role in cancer suppression.\"\n"
    "     Input Claims: [ {{\"id\": \"2\", \"claim\": \"It is believed that p53 helps prevent cancer.\"}} ]\n"
    "     Expected Output: {{\n"
    "         \"abstract\": \"Researchers found that the protein p53 plays a crucial role in cancer suppression.\",\n"
    "         \"claims\": [ {{\"id\": \"2\", \"claim\": \"p53 helps prevent cancer.\"}} ]\n"
    "     }}\n\n"
    "IMPORTANT: When you return your final answer, output only a valid JSON object without any markdown formatting or triple backticks. "
    "The JSON object must have two keys: \"abstract\" (the processed eligibility text) and \"claims\" (an array of objects, each with keys \"id\" and \"claim\").\n\n"
    "Now, below you will be given an eligibility section (concatenated from source_text_sents) and a list of claims. "
    "Also provided is a list of extracted entities (words). "
    "Perform coreference resolution on the entire text using the entities to guide your work, and return the resulting JSON object.\n\n"
    "Extracted Entities: {entities}\n\n"
    "Text:\n"
)


def build_prompt(doc):
    """
    Constructs the prompt for a single document.
    It concatenates the source_text_sents into a single text (serving as the eligibility text),
    appends the claims, and injects the extracted entities.
    """
    # Concatenate the source_text_sents into a single string.
    text_sents = doc.get("source_text_sents", [])
    eligibility_text = "\n".join(text_sents).strip()

    claims = doc.get("claims", [])
    # Extract entity words if present.
    entities_list = []
    if "entities" in doc:
        entities_list = [entity.get("word", "") for entity in doc["entities"]]
    entities_str = ", ".join(sorted(set(entities_list))) if entities_list else "None"

    prompt_text = ENHANCED_PROMPT_PREFIX.format(entities=entities_str)
    prompt_text += "Abstract:\n" + eligibility_text + "\n\n"
    prompt_text += "Claims:\n"
    for idx, claim in enumerate(claims, 1):
        claim_id = claim.get("claim_id", f"claim_{idx}")
        claim_text = claim.get("claim", "")
        prompt_text += f"{idx}. [ID: {claim_id}] {claim_text.strip()}\n"
    return prompt_text


def process_document_with_coref(prompt_text):
    """
    Calls the OpenAI API with the given prompt_text once.
    Returns the parsed JSON response with keys "abstract" and "claims".
    Increases max_tokens to 4096.
    """
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You perform text processing tasks."},
        {"role": "user", "content": prompt_text}
    ],
    temperature=0.2,
    max_tokens=4096)
    answer = response.choices[0].message.content
    processed = json.loads(answer)
    return processed


def save_progress(processed_docs, output_filename):
    """Save the current processed documents to output JSON file."""
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, indent=2)
    print(f"Progress saved to {output_filename}")


def log_failure(failure_entry, failure_filename):
    """Append the failure entry (as a dict) to a JSON file for later diagnostics."""
    try:
        if os.path.exists(failure_filename):
            with open(failure_filename, "r", encoding="utf-8") as f:
                failures = json.load(f)
        else:
            failures = []
    except Exception:
        failures = []
    failures.append(failure_entry)
    with open(failure_filename, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    print(f"Logged failure for document ID {failure_entry.get('source_doc_id')} to {failure_filename}")


def load_existing_processed(output_filename):
    """Load already processed documents (if any) and return a dict mapping source_doc_id -> document."""
    if os.path.exists(output_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            processed = json.load(f)
        return {doc.get("source_doc_id", ""): doc for doc in processed}
    return {}


def main():
    # Load input sample file
    with open(INPUT_FILENAME, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Load already processed documents to skip them
    processed_lookup = load_existing_processed(OUTPUT_FILENAME)
    processed_docs = list(processed_lookup.values())
    total_processed = len(processed_docs)

    # Process up to MAX_ENTRIES documents.
    for doc in documents:
        if total_processed >= MAX_ENTRIES:
            break

        doc_id = doc.get("source_doc_id", "unknown")
        if doc_id in processed_lookup:
            print(f"Skipping already processed document ID: {doc_id}")
            continue

        prompt_text = build_prompt(doc)
        print(f"Processing document ID: {doc_id}")
        try:
            processed = process_document_with_coref(prompt_text)
            new_text = processed.get("abstract", "")
            new_claims_processed = processed.get("claims", [])
            # Replace the original source_text_sents by the processed text (as a single string)
            doc["source_text_processed"] = new_text

            # Update each claim by matching the original claim id.
            claim_dict = {str(claim.get("claim_id")): claim for claim in doc.get("claims", [])}
            for proc_claim in new_claims_processed:
                claim_id = str(proc_claim.get("id"))
                if claim_id in claim_dict:
                    claim_dict[claim_id]["claim"] = proc_claim.get("claim", "")
                else:
                    print(f"Warning: claim id {claim_id} not found in document {doc_id}")
            processed_docs.append(doc)
            processed_lookup[doc_id] = doc
            total_processed += 1
        except Exception as e:
            error_info = {
                "source_doc_id": doc_id,
                "error": str(e),
                "metadata": {
                    "num_claims": len(doc.get("claims", [])),
                    "section": doc.get("section", "")
                }
            }
            log_failure(error_info, FAILURE_FILENAME)
            continue

        # Save progress every SAVE_FREQ entries.
        if total_processed % SAVE_FREQ == 0:
            save_progress(processed_docs, OUTPUT_FILENAME)
            time.sleep(1)

    # Final save
    save_progress(processed_docs, OUTPUT_FILENAME)
    print(f"Processing complete. Total documents processed: {total_processed}")


if __name__ == "__main__":
    main()
