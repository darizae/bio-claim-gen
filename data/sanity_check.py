import json
import os


class AbstractSanityChecker:
    """Runs sanity checks on extracted PubMed abstracts."""

    def __init__(self, filename="pubmed_abstracts.json"):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' not found.")

        self.filename = filename
        self.data = self.load_data()

    def load_data(self):
        """Loads the JSON file containing abstracts."""
        with open(self.filename, "r", encoding="utf-8") as f:
            return json.load(f)

    def check_missing_fields(self):
        """Checks for missing essential fields in the abstracts."""
        missing_fields = {"pmid": 0, "title": 0, "abstract": 0, "authors": 0}

        for entry in self.data:
            if not entry.get("pmid"):
                missing_fields["pmid"] += 1
            if not entry.get("title"):
                missing_fields["title"] += 1
            if not entry.get("abstract"):
                missing_fields["abstract"] += 1
            if not isinstance(entry.get("authors", []), list):
                missing_fields["authors"] += 1  # Ensuring authors field is a list

        return missing_fields

    def check_data_types(self):
        """Ensures that certain fields have expected data types."""
        type_issues = {"pmid": 0, "title": 0, "abstract": 0, "authors": 0}

        for entry in self.data:
            if not isinstance(entry.get("pmid", ""), str) or not entry["pmid"].isdigit():
                type_issues["pmid"] += 1
            if not isinstance(entry.get("title", ""), str):
                type_issues["title"] += 1
            if not isinstance(entry.get("abstract", ""), str):
                type_issues["abstract"] += 1
            if not isinstance(entry.get("authors", []), list):
                type_issues["authors"] += 1

        return type_issues

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

    def check_short_abstracts(self, min_length=50):
        """Identifies abstracts that are too short (potentially irrelevant)."""
        short_abstracts = sum(1 for entry in self.data if len(entry.get("abstract", "")) < min_length)
        return short_abstracts

    def run_checks(self):
        """Runs all sanity checks and prints a report."""
        print(f"Running sanity checks on {len(self.data)} abstracts from '{self.filename}'...\n")

        missing = self.check_missing_fields()
        print("Missing Fields Report:")
        for field, count in missing.items():
            print(f"  - {field}: {count} missing")

        type_issues = self.check_data_types()
        print("\nData Type Issues:")
        for field, count in type_issues.items():
            print(f"  - {field}: {count} incorrect")

        duplicates = self.check_duplicates()
        print(f"\nDuplicate Entries: {duplicates}")

        short_abstracts = self.check_short_abstracts()
        print(f"\nShort Abstracts (<50 chars): {short_abstracts}")

        print("\nSanity check complete.")


if __name__ == "__main__":
    # Run sanity checks on the extracted data
    checker = AbstractSanityChecker("pubmed_abstracts.json")
    checker.run_checks()