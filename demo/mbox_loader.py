from langchain.document_loaders.base import BaseLoader
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, Any
from langchain.docstore.document import Document
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
