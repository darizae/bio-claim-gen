import os
import json

from openai import OpenAI

client = OpenAI()
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import time
from dotenv import load_dotenv

# Load the API key from .env file
load_dotenv()

# Set maximum number of entries to process and the save frequency
MAX_ENTRIES = 200
SAVE_FREQ = 5

# Filenames
INPUT_FILENAME = "sample_scifact.json"
OUTPUT_FILENAME = "sample_scifact_coref.json"
FAILURE_FILENAME = "failed_entries.json"

# Enhanced prompt with chain-of-thought, contextual instructions, and few-shot examples
ENHANCED_PROMPT_PREFIX = (
    "You are a language model expert in coreference resolution and contextual analysis. "
    "Your task is to review an abstract and its associated claims, and then perform coreference resolution across the entire text. "
    "This means you should merge expressions and pronouns referring to the same entity, eliminate duplicates, and ensure that references in the abstract and in the claims are consistent. "
    "Please follow these steps carefully:\n\n"
    "1. **Chain-of-Thought Reasoning:** Think through the context provided by the abstract and the claims. Identify all key entities and concepts, and consider how they are mentioned in different parts of the text.\n\n"
    "2. **Contextual Consistency:** Make sure that any entity mentioned in the abstract is consistently referred to in the claims, and vice versa.\n\n"
    "3. **Merge Duplicates:** If the same entity is referred to using different expressions (for example, 'the drug', 'it', or 'this medication'), merge them into one uniform reference.\n\n"
    "4. **Few-Shot Examples:**\n"
    "   - Example 1:\n"
    "     Input Abstract: \"A study of aspirin in patients with cardiovascular disease shows that the drug reduces heart attacks.\"\n"
    "     Input Claims: [ {\"id\": \"1\", \"claim\": \"The medication reduces the incidence of heart attacks.\"} ]\n"
    "     Expected Output: {\n"
    "         \"abstract\": \"A study of aspirin in patients with cardiovascular disease shows that aspirin reduces heart attacks.\",\n"
    "         \"claims\": [ {\"id\": \"1\", \"claim\": \"Aspirin reduces the incidence of heart attacks.\"} ]\n"
    "     }\n\n"
    "   - Example 2:\n"
    "     Input Abstract: \"Researchers found that the protein p53 plays a crucial role in cancer suppression.\"\n"
    "     Input Claims: [ {\"id\": \"2\", \"claim\": \"It is believed that p53 helps prevent cancer.\"} ]\n"
    "     Expected Output: {\n"
    "         \"abstract\": \"Researchers found that the protein p53 plays a crucial role in cancer suppression.\",\n"
    "         \"claims\": [ {\"id\": \"2\", \"claim\": \"p53 helps prevent cancer.\"} ]\n"
    "     }\n\n"
    "IMPORTANT: When you return your final answer, output only a valid JSON object without any markdown formatting or triple backticks. "
    "The JSON object must have two keys: \"abstract\" (the processed abstract text) and \"claims\" (an array of objects, each with keys \"id\" and \"claim\").\n\n"
    "Now, below you will be given an abstract and a list of claims. Perform coreference resolution on the entire text and return the resulting JSON object.\n\n"
    "Text:\n"
)


def build_prompt(doc):
    """
    Constructs the prompt string for a single document by appending the document's text to the prompt prefix.
    """
    abstract = doc.get("abstract_raw", "")
    claims = doc.get("claims", [])
    prompt_text = ENHANCED_PROMPT_PREFIX
    prompt_text += "Abstract:\n" + abstract.strip() + "\n\n"
    prompt_text += "Claims:\n"
    for idx, claim in enumerate(claims, 1):
        claim_id = claim.get("id", f"claim_{idx}")
        claim_text = claim.get("claim", "")
        prompt_text += f"{idx}. [ID: {claim_id}] {claim_text.strip()}\n"
    return prompt_text


def process_document_with_coref(prompt_text):
    """
    Calls the OpenAI API with the given prompt_text once.
    Returns the parsed JSON response with keys "abstract" and "claims".
    If the call fails (or JSON is invalid), raises an exception.
    """
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You perform text processing tasks."},
        {"role": "user", "content": prompt_text}
    ],
    temperature=0.2,
    max_tokens=2048)
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
    print(f"Logged failure for document ID {failure_entry.get('doc_id')} to {failure_filename}")


def load_existing_processed(output_filename):
    """Load already processed documents (if any) and return a dict mapping doc_id -> document."""
    if os.path.exists(output_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            processed = json.load(f)
        return {doc.get("doc_id", ""): doc for doc in processed}
    return {}


def main():
    input_filename = "sample_scifact.json"  # Input sample file for SciFact
    output_filename = "sample_scifact_coref.json"
    failure_filename = "failed_entries.json"

    # Load the sample data
    with open(input_filename, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Load already processed documents (if any) to skip them
    processed_lookup = load_existing_processed(output_filename)

    processed_docs = list(processed_lookup.values())
    total_processed = len(processed_docs)

    # Process only up to MAX_ENTRIES documents.
    # Skip any document that is already processed (by checking its doc_id).
    for doc in documents:
        if total_processed >= MAX_ENTRIES:
            break

        doc_id = doc.get("doc_id", "unknown")
        if doc_id in processed_lookup:
            print(f"Skipping already processed document ID: {doc_id}")
            continue

        prompt_text = build_prompt(doc)
        print(f"Processing document ID: {doc_id}")
        try:
            processed = process_document_with_coref(prompt_text)
            new_abstract = processed.get("abstract", "")
            new_claims_processed = processed.get("claims", [])
            doc["abstract_raw"] = new_abstract

            # Update each claim by matching the original claim id.
            claim_dict = {str(claim.get("id")): claim for claim in doc.get("claims", [])}
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
                "doc_id": doc_id,
                "error": str(e),
                "metadata": {
                    "num_claims": len(doc.get("claims", []))
                }
            }
            log_failure(error_info, failure_filename)
            continue

        # Iteratively save progress every SAVE_FREQ successful entries.
        if total_processed % SAVE_FREQ == 0:
            save_progress(processed_docs, output_filename)
            time.sleep(1)

    # Final save
    save_progress(processed_docs, output_filename)
    print(f"Processing complete. Total documents processed: {total_processed}")


if __name__ == "__main__":
    main()
