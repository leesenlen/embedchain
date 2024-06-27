from typing import Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import os
import re
import datetime
import logging
import markdown
from bs4 import BeautifulSoup
from embedchain.rag.nlp import rag_tokenizer
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.text_splitter import Document
from embedchain.chunkers.base_chunker import BaseChunker
from embedchain.config.add_config import ChunkerConfig
from embedchain.helpers.json_serializable import register_deserializable


@register_deserializable
class MarkdownChunker(BaseChunker):
    """Chunker for md files."""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        if config is None:
            config = ChunkerConfig(chunk_size=1000, chunk_overlap=0, length_function=len)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=config.length_function,
        )
        super().__init__(text_splitter)

    def chunks(self, loader, src, metadata: Optional[dict[str, Any]] = None, config: Optional[ChunkerConfig] = None):
        with open(src, "r", encoding="utf-8") as file:
            content = file.read()
        headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2")
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(content)
        # TODO: 如果按照标题切分，chunk_size > config.chunk_size. 需要进一步拆分
        documents = []
        chunk_ids = []
        idMap = {}
        min_chunk_size = config.min_chunk_size if config is not None else 1
        logging.info(f"[INFO] Skipping chunks smaller than {min_chunk_size} characters")

        if metadata is None:
            metadata = {}
        app_id = metadata.get("app_id", 1)
        knowledge_id = metadata.get("knowledge_id", 1)
        subject = metadata.get("subject", os.path.basename(src))
        doc_id = self.generate_doc_id(app_id, "".join(each.page_content for each in md_header_splits))
        metadatas = []
        extra_data = []
        for idx, chunk in enumerate(md_header_splits):
            chunk_content = chunk.page_content
            chunk_id = str(doc_id) + "-" + hashlib.sha256(chunk_content.encode()).hexdigest()
            meta_data = {}
            extra_data.append(self.generate_extra_data(subject, chunk))
            meta_data.update(metadata)
            # add data type to meta data to allow query using data type
            meta_data["app_id"] = app_id
            meta_data["doc_id"] = doc_id
            meta_data["knowledge_id"] = knowledge_id
            meta_data["hash"] = doc_id
            meta_data["data_type"] = self.data_type.value
            meta_data["subject"] = subject if subject is not None else os.path.basename(src)
            meta_data["status"] = 1
            meta_data['segment_number'] = idx
            if idMap.get(chunk_id) is None and len(chunk_content) >= min_chunk_size:
                idMap[chunk_id] = True
                chunk_ids.append(chunk_id)
                documents.append(f"主题：{meta_data['subject']}。段落内容：{chunk_content}")
                metadatas.append(meta_data)

        return {
            "documents": documents,
            "ids": chunk_ids,
            "metadatas": metadatas,
            "doc_id": doc_id,
            "extra_data": extra_data
        }

    @staticmethod
    def generate_doc_id(app_id, content):
        return str(app_id) + "-" + hashlib.sha256(content.encode()).hexdigest()

    def generate_extra_data(self, subject: str, chunk: Document) -> dict:
        """
            rag2.0的关键，根据文本，subject生成extra_data,以提升es检索的准确性，需要包含以下字段
            1. docnm_kwd 主题/标题
            2. title_tks 分词后的主题/标题
            3. content_with_weight  带格式文本
            4. create_time   创建时间
            5. create_timestamp_flt  创建时间戳
            6. content_ltks   不带格式且分词的chunk
            7. content_sm_ltks  精细分词后的chunk
        """
        header = chunk.metadata.get("Header1", "")
        if not header:
            header = chunk.metadata.get("Header2", "")
        if header:
            subject = subject + " " + header
        extra_data = {
            "docnm_kwd": subject,
            "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", subject))
        }
        content = chunk.page_content
        extra_data["content_with_weight"] = content
        extra_data["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
        extra_data["create_timestamp_flt"] = datetime.datetime.now().timestamp()
        extra_data["content_ltks"] = rag_tokenizer.tokenize(self.clear_markdown(content))
        extra_data["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(extra_data["content_ltks"])
        return extra_data

    def clear_markdown(self, content):
        """
        清除markdown格式
        """
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, "html.parser")
        processed_content = soup.get_text()
        return processed_content
