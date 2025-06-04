import requests
import csv

# Step 1: Fetch the list of studies from cBioPortal API
url = "https://www.cbioportal.org/api/studies"
headers = {"Accept": "application/json"}
response = requests.get(url, headers=headers)
studies = response.json()

# Step 2: Extract study ID, name, and PMID (if available)
pmid_data = []
for study in studies:
    pmid = study.get('pmid')
    if pmid:
        pmid_data.append({
            "study_id": study.get("studyId", ""),
            "name": study.get("name", ""),
            "pmid": pmid
        })

# Step 3: Save to CSV
csv_filename = "cbioportal_studies_with_pmids.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["study_id", "name", "pmid"])
    writer.writeheader()
    writer.writerows(pmid_data)

print(f"Saved {len(pmid_data)} studies with PMIDs to {csv_filename}")