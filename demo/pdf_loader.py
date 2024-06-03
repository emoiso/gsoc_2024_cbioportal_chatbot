from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from pathlib import Path
from typing import List, Union
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader


class PDFLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path]
    ):

        self.file_path = Path(file_path).resolve()

    # def extract_file_part(self, file_path) -> str:
    #     path_str = str(file_path)
    #     parts = path_str.split('/')

    #     # Find the index of "documentation_site"
    #     try:
    #         index = parts.index('documentation_site')
    #     except ValueError:
    #         return "The path does not contain 'documentation_site'"

    #     # Get the part after "documentation_site"
    #     if index + 1 < len(parts):
    #         remaining_parts = '/'.join(parts[index + 1:])
    #         # Remove the file extension
    #         file_name_without_extension = os.path.splitext(remaining_parts)[0]
    #         return file_name_without_extension    
    #     else:
    #         return ""

    def load(self) -> List[Document]:
        """Load and return documents from the Markdown file."""
        docs: List[Document] = []

        # with open(self.file_path, encoding="utf-8") as file:
        #     pdf_document_content = file.read()

        loader = DirectoryLoader(self.file_path, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
        
        # the default splitter for PDFLoader is RecursiveCharacterTextSplitter
        pages = loader.load_and_split()

        for page in pages:  
            docs.append(page)

        #create url for each file and add url in its metadata
        for doc in docs:    
            doc.metadata['source'] = str(self.file_path)
            url = "https://docs.cbioportal.org/user-guide/overview/" #+ self.extract_file_part(self.file_path).lower()
            doc.metadata['url'] = str(url)

        return docs
