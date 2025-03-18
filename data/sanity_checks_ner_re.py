import json
import os
import torch


class RelationExtractionSanityChecker:
    """Runs sanity checks on extracted NER and relation data and prints summary statistics."""

    def __init__(self, filename="pubmed_abstracts_with_ner_re.json"):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' not found.")

        self.filename = filename
        self.data = self.load_data()

        # Print total number of entries
        print(f"Total number of entries found: {len(self.data)}")

    def load_data(self):
        """Loads the JSON file containing extracted relations and entities."""
        with open(self.filename, "r", encoding="utf-8") as f:
            return json.load(f)

    def check_missing_fields(self):
        """Checks for missing essential fields in the annotated abstracts."""
        missing_fields = {"pmid": 0, "title": 0, "abstract": 0, "entities": 0, "relations": 0}

        for entry in self.data:
            if not entry.get("pmid"):
                missing_fields["pmid"] += 1
            if not entry.get("title"):
                missing_fields["title"] += 1
            if not entry.get("abstract"):
                missing_fields["abstract"] += 1
            if "entities" not in entry or not isinstance(entry["entities"], list):
                missing_fields["entities"] += 1
            if "relations" not in entry or not isinstance(entry["relations"], list):
                missing_fields["relations"] += 1

        return missing_fields

    def check_ner_validity(self, min_score=0.5):
        """Ensures that extracted entities have valid structure and reasonable confidence scores."""
        invalid_ner_entries = 0
        low_confidence_entities = 0

        for entry in self.data:
            entities = entry.get("entities", [])
            for entity in entities:
                if not isinstance(entity, dict) or not all(k in entity for k in ["word", "entity_group", "score"]):
                    invalid_ner_entries += 1
                elif entity["score"] < min_score:
                    low_confidence_entities += 1

        return {"invalid_ner_entries": invalid_ner_entries, "low_confidence_entities": low_confidence_entities}

    def check_relation_validity(self, min_confidence=0.7):
        """Ensures that extracted relations have valid structure and meet confidence threshold."""
        invalid_relation_entries = 0
        low_confidence_relations = 0

        for entry in self.data:
            relations = entry.get("relations", [])
            for relation in relations:
                if not isinstance(relation, dict) or not all(k in relation for k in ["subject", "relation", "object", "confidence"]):
                    invalid_relation_entries += 1
                elif relation["confidence"] < min_confidence:
                    low_confidence_relations += 1

        return {"invalid_relation_entries": invalid_relation_entries, "low_confidence_relations": low_confidence_relations}

    def check_duplicates(self):
        """Checks for duplicate abstracts based on PMID."""
        seen_pmids = set()
        duplicates = 0

        for entry in self.data:
            pmid = entry.get("pmid")
            if pmid in seen_pmids:
                duplicates += 1
            else:
                seen_pmids.add(pmid)

        return duplicates

    def check_empty_results(self):
        """Checks if any abstracts have empty NER or relation extractions."""
        empty_entities = sum(1 for entry in self.data if not entry.get("entities"))
        empty_relations = sum(1 for entry in self.data if not entry.get("relations"))

        # List abstracts with empty relations
        abstracts_with_empty_relations = [
            {"pmid": entry["pmid"], "title": entry["title"]}
            for entry in self.data if not entry.get("relations")
        ]

        return {
            "empty_entities": empty_entities,
            "empty_relations": empty_relations,
            "abstracts_with_empty_relations": abstracts_with_empty_relations,
        }

    def check_torch_device(self):
        """Ensures that PyTorch detects the correct device (MPS, CUDA, or CPU)."""
        if torch.cuda.is_available():
            return "CUDA available"
        elif torch.backends.mps.is_available():
            return "MPS (Apple Silicon) available"
        else:
            return "Running on CPU"

    def print_data_statistics(self):
        """Prints additional statistics for presentation."""
        total_entries = len(self.data)
        total_entities = 0
        total_relations = 0
        min_entities = float('inf')
        max_entities = 0
        min_relations = float('inf')
        max_relations = 0
        all_entity_scores = []
        all_relation_confidences = []

        for entry in self.data:
            entities = entry.get("entities", [])
            relations = entry.get("relations", [])
            num_entities = len(entities)
            num_relations = len(relations)

            total_entities += num_entities
            total_relations += num_relations

            min_entities = min(min_entities, num_entities)
            max_entities = max(max_entities, num_entities)

            min_relations = min(min_relations, num_relations)
            max_relations = max(max_relations, num_relations)

            # Gather scores for NER entities
            for entity in entities:
                if isinstance(entity, dict) and "score" in entity:
                    all_entity_scores.append(entity["score"])
            # Gather confidence scores for relations
            for relation in relations:
                if isinstance(relation, dict) and "confidence" in relation:
                    all_relation_confidences.append(relation["confidence"])

        avg_entities = total_entities / total_entries if total_entries else 0
        avg_relations = total_relations / total_entries if total_entries else 0
        avg_entity_score = sum(all_entity_scores) / len(all_entity_scores) if all_entity_scores else 0
        avg_relation_confidence = sum(all_relation_confidences) / len(all_relation_confidences) if all_relation_confidences else 0

        print("\nData Statistics Report:")
        print(f"  - Total abstracts: {total_entries}")
        print(f"  - Total entities: {total_entities}")
        print(f"  - Total relations: {total_relations}")
        print(f"  - Average entities per abstract: {avg_entities:.2f} (min: {min_entities}, max: {max_entities})")
        print(f"  - Average relations per abstract: {avg_relations:.2f} (min: {min_relations}, max: {max_relations})")
        print(f"  - Average NER entity score: {avg_entity_score:.2f}")
        print(f"  - Average relation confidence: {avg_relation_confidence:.2f}")

    def run_checks(self):
        """Runs all sanity checks and prints a report."""
        print(f"\nRunning sanity checks on {len(self.data)} abstracts from '{self.filename}'...\n")

        missing = self.check_missing_fields()
        print("Missing Fields Report:")
        for field, count in missing.items():
            print(f"  - {field}: {count} missing")

        ner_validity = self.check_ner_validity()
        print("\nNER Validity Report:")
        for field, count in ner_validity.items():
            print(f"  - {field}: {count}")

        relation_validity = self.check_relation_validity()
        print("\nRelation Validity Report:")
        for field, count in relation_validity.items():
            print(f"  - {field}: {count}")

        duplicates = self.check_duplicates()
        print(f"\nDuplicate Entries: {duplicates}")

        empty_results = self.check_empty_results()
        print(f"\nEmpty Extraction Results:")
        print(f"  - Empty Entities: {empty_results['empty_entities']}")
        print(f"  - Empty Relations: {empty_results['empty_relations']}")

        if empty_results["empty_relations"] > 0:
            print("\nAbstracts with Empty Relations:")
            for item in empty_results["abstracts_with_empty_relations"]:
                print(f"  - PMID: {item['pmid']}, Title: {item['title']}")

        device_status = self.check_torch_device()
        print(f"\nTorch Device Check: {device_status}")

        # Print additional data statistics for presentation
        self.print_data_statistics()

        print("\nSanity check complete.")


if __name__ == "__main__":
    # Run sanity checks on the extracted data
    checker = RelationExtractionSanityChecker("pubmed_abstracts_with_ner_re.json")
    checker.run_checks()
