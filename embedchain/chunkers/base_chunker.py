import hashlib
import logging
import traceback
import requests
from typing import Optional, Any
import os

from embedchain.config.add_config import ChunkerConfig
from embedchain.helpers.json_serializable import JSONSerializable
from embedchain.models.data_type import DataType


class BaseChunker(JSONSerializable):
    def __init__(self, text_splitter):
        """Initialize the chunker."""
        self.text_splitter = text_splitter
        self.data_type = None

    def chunks(self, loader, src, metadata: Optional[dict[str, Any]] = None, config: Optional[ChunkerConfig] = None):
        documents = []
        chunk_ids = []
        idMap = {}
        min_chunk_size = config.min_chunk_size if config is not None else 1
        logging.info(f"[INFO] Skipping chunks smaller than {min_chunk_size} characters")

        if metadata is None:
            metadata = {}
        app_id = metadata.get("app_id", 1) #应用ID
        knowledge_id = metadata.get("knowledge_id", 1) #默认知识库ID
        subject = metadata.get("subject", None) #主题

        data_result = loader.load_data(src)
        data_records = data_result["data"]
        hash_data = data_result["doc_id"]
        doc_id = str(app_id) + "-" + data_result["doc_id"]

        metadatas = []
        for data in data_records:
            content = data["content"]

            chunks = self.get_chunks(content)
            number = 0
            for chunk in chunks:

                chunk_id = str(doc_id) + "-" + hashlib.sha256((chunk).encode()).hexdigest()
                number += 1
                meta_data = {}
                meta_data.update(data["meta_data"])
                url = meta_data["url"]
                # add data type to meta data to allow query using data type
                meta_data["app_id"] = app_id
                meta_data["doc_id"] = doc_id
                meta_data["knowledge_id"] = knowledge_id
                meta_data["hash"] = hash_data
                meta_data["data_type"] = self.data_type.value

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
    
    def create_chunks(self, loader, src, app_id=None, config: Optional[ChunkerConfig] = None):
        """
        Loads data and chunks it.

        :param loader: The loader whose `load_data` method is used to create
        the raw data.
        :param src: The data to be handled by the loader. Can be a URL for
        remote sources or local content for local loaders.
        :param app_id: App id used to generate the doc_id.
        """
        documents = []
        chunk_ids = []
        id_map = {}
        min_chunk_size = config.min_chunk_size if config is not None else 1
        logging.info(f"Skipping chunks smaller than {min_chunk_size} characters")
        data_result = loader.load_data(src)
        data_records = data_result["data"]
        doc_id = data_result["doc_id"]
        # Prefix app_id in the document id if app_id is not None to
        # distinguish between different documents stored in the same
        # elasticsearch or opensearch index
        doc_id = f"{app_id}--{doc_id}" if app_id is not None else doc_id
        metadatas = []
        for data in data_records:
            content = data["content"]

            metadata = data["meta_data"]
            # add data type to meta data to allow query using data type
            metadata["data_type"] = self.data_type.value
            metadata["doc_id"] = doc_id

            # TODO: Currently defaulting to the src as the url. This is done intentianally since some
            # of the data types like 'gmail' loader doesn't have the url in the meta data.
            url = metadata.get("url", src)

            chunks = self.get_chunks(content)
            for chunk in chunks:
                chunk_id = hashlib.sha256((chunk + url).encode()).hexdigest()
                chunk_id = f"{app_id}--{chunk_id}" if app_id is not None else chunk_id
                if id_map.get(chunk_id) is None and len(chunk) >= min_chunk_size:
                    id_map[chunk_id] = True
                    chunk_ids.append(chunk_id)
                    documents.append(chunk)
                    metadatas.append(metadata)
        return {
            "documents": documents,
            "ids": chunk_ids,
            "metadatas": metadatas,
            "doc_id": doc_id,
        }

    def get_chunks(self, content):
        """
        Returns chunks using text splitter instance.

        Override in child class if custom logic.
        """
        return self.text_splitter.split_text(content)

    def set_data_type(self, data_type: DataType):
        """
        set the data type of chunker
        """
        self.data_type = data_type

        # TODO: This should be done during initialization. This means it has to be done in the child classes.

    @staticmethod
    def get_word_count(documents) -> int:
        return sum([len(document.split(" ")) for document in documents])

    def request_ocr_with_error_handling(self, src, _type, timeout=60*2):
        assert _type in ["pdf", "jpg", "jpeg", "png", "bmp", "tif", "tiff"], f"Invalid ocr file type {_type}"
        ocr_url = os.getenv("OCR_URL", "")
        if not ocr_url:
            raise EnvironmentError("OCR_URL is not set, please set OCR_URL environment variable")
        mime_types = {
            "pdf": "application/pdf",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "bmp": "image/bmp",
            "tif": "image/tiff",
            "tiff": "image/tiff"
        }
        mime_type = mime_types[_type]
        try:
            with open(src, "rb") as file:
                files = {
                    "file_type": (None, "pdf"),
                    "file": (src, file.read(), mime_type)
                }
            response = requests.post(ocr_url, files=files, timeout=timeout)
        except requests.exceptions.ConnectionError:
            logging.error(f"OCR_URL {ocr_url} connect failed")
            logging.error(traceback.format_exc())
            return {}
        except requests.exceptions.Timeout:
            logging.error(f"OCR_URL {ocr_url} request timeout over {timeout} seconds")
            logging.error(traceback.format_exc())
            return {}
        except Exception:
            logging.exception(traceback.format_exc())
            return {}
        if response.status_code != 200:
            logging.exception(f"OCR_URL {ocr_url} request failed with status code {response.status_code}")
            return {}
        return response.json()
