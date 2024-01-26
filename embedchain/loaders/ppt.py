import hashlib

try:
    from pptx import Presentation
except ImportError:
    raise ImportError(
        'PPT file requires extra dependencies. Install with `pip install --upgrade "embedchain[python-pptx]"`'
    ) from None
from embedchain.helpers.json_serializable import register_deserializable
from embedchain.loaders.base_loader import BaseLoader
from embedchain.utils.misc import clean_string

@register_deserializable
class PPTLoader(BaseLoader):
    def load_data(self, url):
        """Load data from a .pptx file."""
        output = []
        meta_data = {}
        content = clean_string(self.extract_text_from_ppt(url))

        meta_data["url"] = url
        output.append({"content": content, "meta_data": meta_data})
        doc_id = hashlib.sha256((content).encode()).hexdigest()
        return {
            "doc_id": doc_id,
            "data": output
        }
    
    def extract_text_from_ppt(self, ppt_file_path):
        presentation = Presentation(ppt_file_path)

        all_text = []

        for slide_number, slide in enumerate(presentation.slides, start=1):
            slide_text = []

            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text.append(shape.text)

            if slide_text:
                all_text.append(f"Slide {slide_number}:")
                all_text.extend(slide_text)
                all_text.append("\n")

        return "\n".join(all_text)
