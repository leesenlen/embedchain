from typing import Optional
import hashlib
from typing import Optional,Any
from embedchain.models.base import BaseModel

from embedchain.chunkers.base_chunker import BaseChunker
from embedchain.config.add_config import ChunkerConfig
from embedchain.helpers.json_serializable import register_deserializable


@register_deserializable
class StructureChunker(BaseChunker):

    def __init__(self, config: Optional[ChunkerConfig] = None):
        if config is None:
            config = ChunkerConfig(chunk_size=1000, chunk_overlap=0, length_function=len)
    
        super().__init__(None)

    def chunks(self, loader, data: Any, metadata: Optional[dict[str, Any]] = None,config: Optional[ChunkerConfig] = None):
        documents = []
        chunk_ids = []
        table = data['model']
        if not isinstance(table, BaseModel):
            raise Exception("Model is not a BaseModel")
        if metadata is None:
            metadata = {}
        app_id =  metadata.get("app_id",1) #应用ID
        knowledge_id = metadata.get("knowledge_id",0) #知识库ID   
        subject = metadata.get("subject",table.description) #主题

        data_records = loader.load_data(data)
        hash_data = hashlib.sha256((table.database + '.' + table.table_name).encode()).hexdigest()
        doc_id = str(app_id) + "-" + hash_data

        metadatas = []
        for data in data_records:
            content = data["content"]

            meta_data = data["meta_data"]
            meta_data["app_id"] = app_id
            meta_data["doc_id"] = doc_id
            meta_data["knowledge_id"] = knowledge_id
            meta_data["hash"] = hash_data
            meta_data["data_type"] = self.data_type.value

            meta_data["subject"] = subject
            meta_data["status"] = 1

            chunk_ids.append(doc_id+'-'+str(data['primary_key']))
            documents.append(f"主题：{meta_data['subject']},段落内容：{content}")
            metadatas.append(meta_data)
        
        return {
            "documents": documents,
            "ids": chunk_ids,
            "metadatas": metadatas,
            "doc_id": doc_id
        }

    
