import os
import json
from dotenv import load_dotenv
from Bio import Entrez, Medline


class Config:
    """Loads configuration from the .env file"""

    def __init__(self, env_file="../.env"):
        load_dotenv(env_file)  # Load environment variables
        self.email = os.getenv("NCBI_EMAIL")

        if not self.email:
            raise ValueError("NCBI_EMAIL is not set in the .env file.")


class PubMedFetcher:
    """Fetches PubMed abstracts based on a search term and dataset size."""

    def __init__(self, email: str, search_term: str, max_results: int = 100):
        self.email = email
        self.search_term = search_term
        self.max_results = max_results
        Entrez.email = self.email  # Set the email for Entrez API

    def fetch_pubmed_ids(self):
        """Searches PubMed and returns a list of article IDs."""
        handle = Entrez.esearch(db="pubmed", term=self.search_term, retmax=self.max_results * 2)  # Fetch extra IDs
        record = Entrez.read(handle)
        handle.close()
        return record.get("IdList", [])

    def fetch_abstracts(self, id_list):
        """
        Fetches abstracts for the given PubMed article IDs.
        Filters out articles with "No abstract found" and keeps fetching more if needed.
        """
        valid_abstracts = []
        batch_size = 50  # Fetch in batches to avoid API overload
        processed_ids = set()

        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i + batch_size]
            fetch_handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="text")
            records = Medline.parse(fetch_handle)
            fetch_handle.close()

            for rec in records:
                pmid = rec.get("PMID", "")
                abstract = rec.get("AB", "No abstract found").strip()

                if abstract.lower() != "no abstract found" and pmid not in processed_ids:
                    processed_ids.add(pmid)
                    valid_abstracts.append({
                        "pmid": pmid,
                        "title": rec.get("TI", "No title found"),
                        "abstract": abstract,
                        "authors": rec.get("AU", []),
                        "year": rec.get("DP", "No date found")
                    })

                if len(valid_abstracts) >= self.max_results:
                    return valid_abstracts  # Stop when we reach the required number

        return valid_abstracts

    def save_to_json(self, data, filename="pubmed_abstracts.json"):
        """Saves fetched abstracts to a JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} valid abstracts to {filename}")

    def run(self):
        """Main function to fetch and save PubMed abstracts."""
        print(f"Searching PubMed for: {self.search_term} (max {self.max_results} results)")
        id_list = self.fetch_pubmed_ids()
        print(f"Found {len(id_list)} article IDs.")

        abstracts = self.fetch_abstracts(id_list)
        print(f"Retrieved {len(abstracts)} valid abstracts (after filtering out missing abstracts).")

        if abstracts:
            self.save_to_json(abstracts)

        return abstracts


if __name__ == "__main__":
    # Load configuration
    config = Config()

    # Define search term and dataset size
    search_term = "Alzheimer's disease[Title/Abstract] AND 2020:2025[dp]"
    max_results = 100  # Adjust as needed

    # Initialize and run the PubMed fetcher
    fetcher = PubMedFetcher(email=config.email, search_term=search_term, max_results=max_results)
    abstracts = fetcher.run()

    # Optionally, process the abstracts further here
    print(f"Finished fetching {len(abstracts)} abstracts.")
