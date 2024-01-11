from typing import Optional
import hashlib
import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter

from embedchain.chunkers.base_chunker import BaseChunker
from embedchain.config.add_config import ChunkerConfig
from embedchain.helpers.json_serializable import register_deserializable


@register_deserializable
class MySQLChunker(BaseChunker):
    """Chunker for json."""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        if config is None:
            config = ChunkerConfig(chunk_size=1000, chunk_overlap=0, length_function=len)
    
        super().__init__(None)

    def chunks(self, loader, params, config: Optional[ChunkerConfig] = None):
        documents = []
        chunk_ids = []

        schema = params["source"]
        app_id = params.get("app_id",'0') #应用ID
        knowledge_id = params.get("knowledge_id",uuid.uuid4()) #知识库ID
        link = params.get("link",None) #链接
        subject = params.get("subject",None) #主题
        labels = params.get("labels",None) #标签
        is_public = params.get("is_public",0) #是否开放

        data_records = loader.load_data(schema)
        hash_data = hashlib.sha256((schema['database'] + '.' + schema['table_name']).encode()).hexdigest()
        doc_id = app_id + "-" + hash_data

        metadatas = []
        for data in data_records:
          
            meta_data = data["meta_data"]
            # add data type to meta data to allow query using data type
            meta_data["app_id"] = app_id
            meta_data["doc_id"] = doc_id
            meta_data["knowledge_id"] = knowledge_id
            meta_data["hash"] = hash_data
            meta_data["data_type"] = self.data_type.value
            
            meta_data["link"] = link
            if subject is None and "table_description" in schema:
                meta_data['subject'] = schema['table_description']
            else:
                meta_data["subject"] = subject
            meta_data["labels"] = labels
            meta_data["is_public"] = is_public
                   
            chunk_ids.append(doc_id+'-'+str(data['primary_key']))
            documents.append(data["content"])
            metadatas.append(meta_data)
        return {
            "documents": documents,
            "ids": chunk_ids,
            "metadatas": metadatas,
            "hash_file": hash_data
        }

    
