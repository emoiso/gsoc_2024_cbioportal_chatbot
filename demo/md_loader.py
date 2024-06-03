from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from pathlib import Path
from typing import List, Union
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


class MarkDownLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path]
    ):

        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]
        self.file_path = Path(file_path).resolve()

    # extract the path of file for creating url
    # for example: url for About_us.md is https://docs.cbioportal.org/about-us/
    # url for documention site is always "https://docs.cbioportal.org/" + "PATH of FILE"
    # this function can extract path of file
    # if cannot be extracted, the url will be https://docs.cbioportal.org/
    def extract_file_part(self, file_path) -> str:
        path_str = str(file_path)
        parts = path_str.split('/')

        # Find the index of "documentation_site"
        try:
            index = parts.index('documentation_site')
        except ValueError:
            return "The path does not contain 'documentation_site'"

        # Get the part after "documentation_site"
        if index + 1 < len(parts):
            remaining_parts = '/'.join(parts[index + 1:])
            # Remove the file extension
            file_name_without_extension = os.path.splitext(remaining_parts)[0]
            return file_name_without_extension    
        else:
            return ""

    def load(self) -> List[Document]:
        """Load and return documents from the Markdown file."""
        docs: List[Document] = []

        with open(self.file_path, encoding="utf-8") as file:
            markdown_document_content = file.read()

        # recursive splitter
        # split page_content into chunks
        # more separator : "\n\n", "\n", " "
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512, 
            chunk_overlap=256,
            separators=["#", "##", "###", "####", "#####", "######"]
        )

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_document_content)
        recursive_splits = text_splitter.split_text(markdown_document_content)
        #convert string to doc, otherwise it cannot have metaData
        recursive_splits_doc = text_splitter.create_documents(recursive_splits)

        for split in recursive_splits_doc:  
            docs.append(split)

        #create url for each file and add url in its metadata
        for doc in docs:    
            doc.metadata['source'] = str(self.file_path)
            url = "https://docs.cbioportal.org/" + self.extract_file_part(self.file_path).lower()
            doc.metadata['url'] = str(url)

        return docs
