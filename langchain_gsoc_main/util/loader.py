
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, Any

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_index.readers.file import MboxReader as MboxFileReader


class MboxLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path]
    ):
        
        self.file_path = Path(file_path).resolve()

    def load(self) -> List[Document]:
        """Load and return documents from the Mbox file."""
        docs: List[Document] = []
        docs.extend(MboxFileReader().load_data(self.file_path))
        return [doc.to_langchain_format() for doc in docs]

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
        ]
        self.file_path = Path(file_path).resolve()

    def load(self) -> List[Document]:
        """Load and return documents from the Markdown file."""
        docs: List[Document] = []

        with open(self.file_path, encoding="utf-8") as file:
            markdown_document_content = file.read()

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_document_content)
        for md_header_split in md_header_splits:
            docs.append(md_header_split)

        return docs


class JSONLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        content_key: Optional[str] = None,
        metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
        text_content: bool = True,
        json_lines: bool = False,
    ):
        """
        Initializes the JSONLoader with a file path, an optional content key to extract specific content,
        and an optional metadata function to extract metadata from each record.
        """
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
        self._metadata_func = metadata_func
        self._text_content = text_content
        self._json_lines = json_lines

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""
        docs: List[Document] = []
        if self._json_lines:
            with self.file_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._parse(line, docs)
        else:
            self._parse(self.file_path.read_text(encoding="utf-8"), docs)
        return docs

    def _parse(self, content: str, docs: List[Document]) -> None:
        """Convert given content to documents."""
        data = json.loads(content)

        # Perform some validation
        # This is not a perfect validation, but it should catch most cases
        # and prevent the user from getting a cryptic error later on.

        for i, sample in enumerate(data, len(docs) + 1):
            text = self._get_text(sample=sample)
            metadata = self._get_metadata(sample=sample, source=str(self.file_path), seq_num=i)
            docs.append(Document(page_content=text, metadata=metadata))

    def _get_text(self, sample: Any) -> str:
        """Convert sample to string format"""
        if self._content_key is not None:
            content = sample.get(self._content_key)
        else:
            content = sample

        if self._text_content and not isinstance(content, str):
            raise ValueError(
                f"Expected page_content is string, got {type(content)} instead. \
                    Set `text_content=False` if the desired input for \
                    `page_content` is not a string"
            )

        # In case the text is None, set it to an empty string
        elif isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content) if content else ""
        else:
            return str(content) if content is not None else ""

    def _get_metadata(self, sample: Dict[str, Any], **additional_fields: Any) -> Dict[str, Any]:
        """
        Return a metadata dictionary base on the existence of metadata_func
        :param sample: single data payload
        :param additional_fields: key-word arguments to be added as metadata values
        :return:
        """
        if self._metadata_func is not None:
            return self._metadata_func(sample, additional_fields)
        else:
            return additional_fields


def json_demo():
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["name"] = record.get("name")
        metadata["summary"] = record.get("summary")
        metadata["url"] = record.get("url")
        metadata["category"] = record.get("category")
        metadata["updated_at"] = record.get("updated_at")

        return metadata

    loader = JSONLoader("../sample_data/sample.json", "content", metadata_func=metadata_func)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=256
    )
    docs = loader.load_and_split(text_splitter=text_splitter)
    for doc in docs:
        print("===")
        print(type(doc.page_content))
        # print(doc.page_content)


def mbox_demo():
    loader = MboxLoader("../sample_data/sample.mbox")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=256
    )
    docs = loader.load_and_split(text_splitter=text_splitter)
    for doc in docs:
        print("===")
        print(doc)

if __name__ == '__main__':
    json_demo()
