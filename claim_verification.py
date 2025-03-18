import os
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import pipeline

#################################
#  MODEL & FILE PATH CONFIG     #
#################################

# For NLI, we now use a general-purpose model.
NLI_MODEL_NAME = "roberta-large-mnli"

# For QA, we still use a biomedical QA model (e.g., BioBERT on SQuAD).
QA_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1-squad"

# Path to your final data with claims
INPUT_JSON = "data/pubmed_abstracts_with_claims.json"
# Where to save the final data with verification results
OUTPUT_JSON = "data/pubmed_abstracts_with_claims_verified.json"


#################################
#   LOADING & SAVING UTILITIES  #
#################################

def load_abstracts_with_claims(json_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_abstracts_with_claims(abstracts: List[Dict[str, Any]], json_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(abstracts, f, indent=2, ensure_ascii=False)
    print(f"Saved verified data to {json_path}")


####################################
#   NLI-BASED VERIFICATION LOGIC   #
####################################

def init_nli_pipeline(model_name: str = NLI_MODEL_NAME, device: str = None):
    """
    Initialize a text-classification pipeline for NLI.
    This general-purpose model outputs labels such as "ENTAILMENT", "NEUTRAL", and "CONTRADICTION".
    """
    if device is None:
        device_idx = 0 if torch.cuda.is_available() else -1
    else:
        device_idx = 0 if device == "cuda" else -1

    nli_pipe = pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device_idx,
        return_all_scores=True
    )
    return nli_pipe


def verify_claim_nli(nli_pipeline, abstract_text: str, claim_text: str) -> Dict[str, Any]:
    """
    Use the NLI pipeline to see if `claim_text` is entailed by `abstract_text`.
    We concatenate the abstract (premise) and claim (hypothesis) in a simple format.
    Returns the best label and its confidence.
    """
    input_text = f"premise: {abstract_text}\nhypothesis: {claim_text}"
    results = nli_pipeline(input_text)[0]
    best_label_dict = max(results, key=lambda x: x["score"])
    return {
        "label": best_label_dict["label"],
        "score": best_label_dict["score"]
    }


###################################
#   QA-BASED VERIFICATION LOGIC   #
###################################

def init_qa_pipeline(model_name: str = QA_MODEL_NAME, device: str = None):
    """
    Initialize a question-answering pipeline using a biomedical QA model.
    """
    if device is None:
        device_idx = 0 if torch.cuda.is_available() else -1
    else:
        device_idx = 0 if device == "cuda" else -1

    qa_pipe = pipeline(
        task="question-answering",
        model=model_name,
        tokenizer=model_name,
        device=device_idx
    )
    return qa_pipe


def verify_claim_qa(qa_pipeline, abstract_text: str, claim_text: str) -> Dict[str, Any]:
    """
    For QA-based verification, we reframe the claim as a question:
      "Does the text explicitly state that: <claim_text>?"
    and use the abstract as context.
    Returns the answer snippet and confidence score.
    """
    question = f"Does the text explicitly state that: {claim_text}?"
    try:
        output = qa_pipeline(question=question, context=abstract_text)
        if not output or not isinstance(output, dict):
            return {"answer": "unknown", "score": 0.0}
        ans = output.get("answer", "").strip()
        scr = float(output.get("score", 0.0))
        return {"answer": ans, "score": scr}
    except Exception as e:
        return {"answer": "error", "score": 0.0, "error": str(e)}


#################################
#   METRIC / LABEL DECISIONS    #
#################################

def interpret_nli_label(label: str) -> bool:
    """
    Naively interpret the NLI label:
      - "ENTAILMENT" => True
      - Others ("NEUTRAL", "CONTRADICTION") => False
    """
    return "entail" in label.lower()


def interpret_qa_answer(answer: str, score: float) -> bool:
    """
    Naively interpret the QA output:
      - If the answer is non-empty and the score exceeds a threshold, consider it supported.
    """
    if not answer or answer in ["[CLS]", "empty", "unknown"]:
        return False
    return score > 0.2


#################################
#         MAIN LOGIC            #
#################################

def main():
    data = load_abstracts_with_claims(INPUT_JSON)
    print(f"Loaded {len(data)} abstracts from {INPUT_JSON}.")

    print("Initializing NLI pipeline...")
    nli_pipe = init_nli_pipeline(NLI_MODEL_NAME)
    print(f"NLI pipeline ready ({NLI_MODEL_NAME}).")

    print("Initializing QA pipeline...")
    qa_pipe = init_qa_pipeline(QA_MODEL_NAME)
    print(f"QA pipeline ready ({QA_MODEL_NAME}).")

    verified_count = 0
    total_claims = 0

    for entry in data:
        abstract_text = entry.get("abstract", "")
        if not abstract_text.strip():
            continue

        claims_dict = entry.get("claims", {})

        for strategy_name, claim_list in claims_dict.items():
            if not isinstance(claim_list, list):
                continue

            for i, claim_text in enumerate(claim_list):
                total_claims += 1

                nli_result = verify_claim_nli(nli_pipe, abstract_text, claim_text)
                qa_result = verify_claim_qa(qa_pipe, abstract_text, claim_text)

                is_entailment = interpret_nli_label(nli_result["label"])
                is_qa_supported = interpret_qa_answer(qa_result["answer"], qa_result["score"])

                verification_info = {
                    "nli": {
                        "label": nli_result["label"],
                        "score": nli_result["score"],
                        "entailed": is_entailment
                    },
                    "qa": {
                        "answer": qa_result["answer"],
                        "score": qa_result["score"],
                        "supported": is_qa_supported
                    }
                }

                claim_list[i] = {
                    "text": claim_text,
                    "verification": verification_info
                }

                if is_entailment or is_qa_supported:
                    verified_count += 1

        entry["claims"] = claims_dict

    save_abstracts_with_claims(data, OUTPUT_JSON)
    print(f"Total claims processed: {total_claims}")
    print(f"Claims verified (entailed or QA-supported): {verified_count}")


if __name__ == "__main__":
    main()
