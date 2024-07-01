import os
import re
import copy
from PIL import Image
from io import BytesIO
from typing import Optional, Any
from embedchain.rag.nlp import tokenize_table, naive_merge, tokenize_chunks, rag_tokenizer, tokenize, is_english
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
        metadatas = []
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
        txts = self.__call__(src, from_page, to_page)

        import aspose.slides as slides
        import aspose.pydrawing as drawing
        with open(src, "rb") as f:
            fnm = f.read()
        imgs = []
        with slides.Presentation(BytesIO(fnm)) as presentation:
            for i, slide in enumerate(presentation.slides[from_page: to_page]):
                buffered = BytesIO()
                slide.get_thumbnail(
                    0.5, 0.5).save(
                    buffered, drawing.imaging.ImageFormat.jpeg)
                imgs.append(Image.open(buffered))
        doc_id = self.generate_doc_id(app_id, "".join(txts))
        assert len(imgs) == len(
            txts), "Slides text and image do not match: {} vs. {}".format(len(imgs), len(txts))
        self.is_english = is_english(txts)

        for index, (txt, img) in enumerate(zip(txts, imgs)):
            # TODO: 添加图片处理
            chunk = txt
            chunk_id = str(doc_id) + "-" + hashlib.sha256(chunk.encode()).hexdigest()
            meta_data = {}
            url = src
            # add data type to meta data to allow query using data type
            meta_data["app_id"] = app_id
            meta_data["doc_id"] = doc_id
            meta_data["knowledge_id"] = knowledge_id
            meta_data["hash"] = doc_id
            meta_data["data_type"] = self.data_type.value
            meta_data["subject"] = subject if subject is not None else os.path.basename(url)
            meta_data["status"] = 1
            meta_data['segment_number'] = index
            if idMap.get(chunk_id) is None and len(chunk) >= min_chunk_size:
                idMap[chunk_id] = True
                chunk_ids.append(chunk_id)
                documents.append(f"主题：{meta_data['subject']}。段落内容：{chunk}")
                metadatas.append(meta_data)
        return {
            "documents": documents,
            "ids": chunk_ids,
            "metadatas": metadatas,
            "doc_id": doc_id
        }


