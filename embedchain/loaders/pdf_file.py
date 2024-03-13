import hashlib

try:
    from langchain_community.document_loaders import PyPDFLoader,sql_database
except ImportError:
    raise ImportError(
        'PDF File requires extra dependencies. Install with `pip install --upgrade "embedchain[dataloaders]"`'
    ) from None
from embedchain.helpers.json_serializable import register_deserializable
from embedchain.loaders.base_loader import BaseLoader
from embedchain.utils.misc import clean_string


@register_deserializable
class PdfFileLoader(BaseLoader):
    def load_data(self, url):
        """Load data from a PDF file."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",  # noqa:E501
        }
        loader = PyPDFLoader(url, headers=headers)
        data = []
        all_content = []
        pages = loader.load_and_split()
        if not len(pages):
            raise ValueError("No data found")
        metadata = {}
        metadata["url"] = url
        for page in pages:
            content = page.page_content
            content = clean_string(content)
            all_content.append(content)

        text = "".join(all_content)    
        data.append(
                {
                    "content": text,
                    "meta_data": metadata,
                }
            )
        doc_id = hashlib.sha256((text).encode()).hexdigest()
        return {
            "doc_id": doc_id,
            "data": data
        }
