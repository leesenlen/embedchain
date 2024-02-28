import hashlib
import os

from embedchain.helpers.json_serializable import register_deserializable
from embedchain.loaders.base_loader import BaseLoader
from embedchain.utils.misc import clean_string


@register_deserializable
class TextFileLoader(BaseLoader):
    def load_data(self, url: str):
        """Load data from a text file located at a local path."""
        if not os.path.exists(url):
            raise FileNotFoundError(f"The file at {url} does not exist.")

        with open(url, "r", encoding="utf-8") as file:
            content = clean_string(file.read())

        doc_id = hashlib.sha256((content).encode()).hexdigest()

        meta_data = {"url": url}

        return {
            "doc_id": doc_id,
            "data": [
                {
                    "content": content,
                    "meta_data": metadata,
                }
            ]
        }
