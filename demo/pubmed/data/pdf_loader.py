
from langchain_community.document_loaders import PyMuPDFLoader


loader = PyMuPDFLoader("pubmed/paper_on_pmc_but_no_full_xml_btr694.pdf")
data = loader.load()

with open('pasredPDF.txt', 'w') as f:
    f.write(str(data))
