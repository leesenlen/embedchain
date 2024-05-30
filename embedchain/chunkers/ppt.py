import os
import re
import copy
from typing import Optional, Any
from embedchain.rag.nlp import tokenize_table, naive_merge, tokenize_chunks, rag_tokenizer, tokenize
from embedchain.deepdoc.parser.ppt_parser import RAGFlowPptParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

from embedchain.chunkers.base_chunker import BaseChunker
from embedchain.config.add_config import ChunkerConfig
from embedchain.helpers.json_serializable import register_deserializable


@register_deserializable
class PPTChunker(BaseChunker, RAGFlowPptParser):
    """Chunker for ppt files."""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        super().__init__(RAGFlowPptParser)
        if config is None:
            config = ChunkerConfig(chunk_size=1000, chunk_overlap=0, length_function=len)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=config.length_function,
        )
        super().__init__(text_splitter)

    def __garbage(self, txt):
        txt = txt.lower().strip()
        if re.match(r"[0-9\.,%/-]+$", txt):
            return True
        if len(txt) < 3:
            return True
        return False

    @staticmethod
    def generate_doc_id(app_id, content):
        return str(app_id) + "-" + hashlib.sha256(content.encode()).hexdigest()

    def chunks(self, loader, src, metadata: Optional[dict[str, Any]] = None, config: Optional[ChunkerConfig] = None):
        min_chunk_size = config.min_chunk_size if config is not None else 1
        documents = []
        chunk_ids = []
        idMap = {}
        filename = os.path.basename(src)
        app_id = metadata.get("app_id", 1)
        knowledge_id = metadata.get("knowledge_id", 1)
        subject = metadata.get("subject", None)
        pn = 0
        from_page = int(metadata.get("from_page", 0))
        to_page = int(metadata.get("to_page", 100000))
        eng = metadata.get("eng", False)
        delimiter = metadata.get("delimiter", "\n!?。；！？")
        lines = []
        txts = self.__call__(filename, from_page, to_page)

        import aspose.slides as slides
        import aspose.pydrawing as drawing
        imgs = []
        # with slides.Presentation(BytesIO(fnm)) as presentation:
        #     for i, slide in enumerate(presentation.slides[from_page: to_page]):
        #         buffered = BytesIO()
        #         slide.get_thumbnail(
        #             0.5, 0.5).save(
        #             buffered, drawing.imaging.ImageFormat.jpeg)
        #         imgs.append(Image.open(buffered))
        # assert len(imgs) == len(
        #     txts), "Slides text and image do not match: {} vs. {}".format(len(imgs), len(txts))
        # callback(0.9, "Image extraction finished")
        # self.is_english = is_english(txts)
        doc = {
            "docnm_kwd": filename,
            "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
        }
        doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
        res = []
        for pn, (txt, img) in enumerate(self.__call__(filename, from_page, 1000000)):
            d = copy.deepcopy(doc)
            pn += from_page
            d["image"] = img
            d["page_num_int"] = [pn + 1]
            d["top_int"] = [0]
            d["position_int"] = [(pn + 1, 0, img.size[0], 0, img.size[1])]
            tokenize(d, txt, eng)
            res.append(d)
        return res
