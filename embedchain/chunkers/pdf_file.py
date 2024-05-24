import re
import os
import time
import hashlib
from typing import Optional, Any
import logging
from embedchain.deepdoc.parser import PdfParser
from embedchain.rag.nlp import rag_tokenizer, naive_merge, tokenize_table, tokenize_chunks, find_codec
from langchain.text_splitter import RecursiveCharacterTextSplitter

from embedchain.chunkers.base_chunker import BaseChunker
from embedchain.config.add_config import ChunkerConfig
from embedchain.helpers.json_serializable import register_deserializable


@register_deserializable
class PdfFileChunker(BaseChunker, PdfParser):
    """Chunker for PDF file."""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        PdfParser.__init__(self)
        if config is None:
            config = ChunkerConfig(chunk_size=1000, chunk_overlap=0, length_function=len)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=config.length_function,
        )
        super().__init__(text_splitter)

    def chunks(self, loader, src, metadata: Optional[dict[str, Any]] = None, config: Optional[ChunkerConfig] = None):
        documents = []
        chunk_ids = []
        idMap = {}
        min_chunk_size = config.min_chunk_size if config is not None else 1
        logging.info(f"[INFO] Skipping chunks smaller than {min_chunk_size} characters")

        if metadata is None:
            metadata = {}
        app_id = metadata.get("app_id", 1)
        knowledge_id = metadata.get("knowledge_id", 1)
        subject = metadata.get("subject", None)
        # OCR,布局识别
        cks = self.ocr_and_layout_recognition(src, config)
        doc_id = self.generate_doc_id(app_id, "".join(each["content_with_weight"] for each in cks))
        metadatas = []
        for ck in cks:
            if ch.get("image"):
                # TODO 图片存储
                ...
            chunk = ck["content_with_weight"]
            number = 0
            chunk_id = str(doc_id) + "-" + hashlib.sha256(chunk.encode()).hexdigest()
            meta_data = {}
            url = src
            # add data type to meta data to allow query using data type
            meta_data["app_id"] = app_id
            meta_data["doc_id"] = doc_id
            meta_data["knowledge_id"] = knowledge_id
            meta_data["hash"] = doc_id
            meta_data["data_type"] = self.data_type.value
            # TODO: 主题可以是一段文本的summary
            meta_data["subject"] = subject if subject is not None else os.path.basename(url)
            meta_data["status"] = 1
            meta_data['segment_number'] = number
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

    def ocr_and_layout_recognition(self, src: str, config: ChunkerConfig) -> dict:
        """
        对PDF文件进行OCR解析，表格识别，布局识别，字体合并等操作
        """
        cks = []
        zoomin = 3
        doc = {
            "docnm_kwd": src,
            "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", src))
        }
        doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
        start = time.time()
        logging.info("OCR is running...")
        self.__images__(src)
        logging.info("OCR finished")
        logging.info(f"OCR cost time: {time.time() - start :.4f}s")
        # 布局识别: 表格识别，文字合并等等
        start = time.time()
        self._layouts_rec(zoomin)
        logging.info("Layout analysis finished.")
        self._table_transformer_job(zoomin)
        logging.info("Table analysis finished.")
        self._text_merge()
        logging.info("Text merging finished")
        tbls = self._extract_table_figure(True, zoomin, True, True)
        self._concat_downward()
        sections = [(b["text"], self._line_tag(b, zoomin)) for b in self.boxes]
        logging.info(f"布局识别耗时: {time.time() - start :.4f}s")
        print(self.boxes)
        cks = tokenize_table(tbls, doc, False)
        chunks = naive_merge(
            sections, config.chunk_size, delimiter="\n!?。；！？")
        cks.extend(tokenize_chunks(chunks, doc, False, pdf_parser=None))
        return cks

    @staticmethod
    def generate_doc_id(app_id, content):
        return str(app_id) + "-" + hashlib.sha256(content.encode()).hexdigest()

