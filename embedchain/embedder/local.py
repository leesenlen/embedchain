import os
from typing import Optional
import requests
from embedchain.config.log_conf import logger

from embedchain.config import BaseEmbedderConfig
from embedchain.embedder.base import BaseEmbedder
from embedchain.utils.api_manager import retry


class LocalEmbedder(BaseEmbedder):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config=config)

        if self.config.model is None:
            self.config.model = "bge-m3"
        model_kwargs = {'dimensions':self.config.vector_dimension}

        # 這裡实现自己的embedding_fn
        self.set_embedding_fn(embedding_fn=self.to_embeddings)
        vector_dimension = self.config.vector_dimension or 1024
        self.set_vector_dimension(vector_dimension=vector_dimension)

    @retry(retries=3, retry_delay=2)
    def requests_local_embedding(self, inputs: list):
        embedding_url = os.getenv("LOCAL_EMBEDDING_URL", "")
        if not embedding_url:
            raise EnvironmentError("LOCAL_EMBEDDING_URL not set, please set LOCAL_EMBEDDING_URL environment variable")
        logger.info(f"start to request embedding_url: {embedding_url}")
        response = requests.post(embedding_url, json={"input": inputs, "model": self.config.model}, timeout=10)
        return response

    def to_embeddings(self, inputs: list):
        logger.info(f"embedding docs: {inputs}")
        response = self.requests_local_embedding(inputs).json()
        result = response["data"]
        logger.info(f"embedding successfully: {str(result)[:100]}...")
        return result
