import hashlib

try:
    import markdown2
except ImportError:
    raise ImportError(
        'md file requires extra dependencies. Install with `pip install markdown2`'
    ) from None
from embedchain.helpers.json_serializable import register_deserializable
from embedchain.loaders.base_loader import BaseLoader
from embedchain.utils.misc import clean_string

@register_deserializable
class MarkdownLoader(BaseLoader):
    def load_data(self, url):
        """Load data from a .md file."""
        output = []
        meta_data = {}
        content = clean_string(self.extract_text_from_markdown(url))

        meta_data["url"] = url
        output.append({"content": content, "meta_data": meta_data})
        doc_id = hashlib.sha256((content).encode()).hexdigest()
        return {
            "doc_id": doc_id,
            "data": output
        }
    
    def extract_text_from_markdown(self, md_file_path):
        with open(md_file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()

        # 将 Markdown 转换为 HTML
        html_content = markdown2.markdown(md_content)

        # 使用 BeautifulSoup 从 HTML 中提取纯文本
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text()

        return text_content
