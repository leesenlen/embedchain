import hashlib
import subprocess
import os

try:
    from langchain.document_loaders import Docx2txtLoader
except ImportError:
    raise ImportError(
        'Docx file requires extra dependencies. Install with `pip install --upgrade "embedchain[dataloaders]"`'
    ) from None
from embedchain.helpers.json_serializable import register_deserializable
from embedchain.loaders.base_loader import BaseLoader


@register_deserializable
class DocFileLoader(BaseLoader):
    def load_data(self, url):
        """Load data from a .doc file."""
        tmp_file = url+"x"
        self.convert_doc_to_docx(url)
        
        loader = Docx2txtLoader(tmp_file)
        output = []
        data = loader.load()
        content = data[0].page_content
        meta_data = data[0].metadata
        meta_data["url"] = url
        output.append({"content": content, "meta_data": meta_data})
        doc_id = hashlib.sha256((content + url).encode()).hexdigest()
        hash_data = hashlib.sha256((content).encode()).hexdigest()
        os.remove(tmp_file)
        return {
            "doc_id": doc_id,
            "data": output,
            "hash": hash_data
        }
    

    def convert_doc_to_docx(self, doc_path):
        try:
            # 使用unoconv将.doc文件转换为.docx
            subprocess.run(["unoconv", "-f", "docx", "-o", ".", doc_path])
        except Exception as e:
            print(f"Error: {e}")


