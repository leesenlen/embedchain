import logging
from typing import Any, Optional, Union

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
except ImportError:
    raise ImportError(
        "Elasticsearch requires extra dependencies. Install with `pip install --upgrade embedchain[elasticsearch]`"
    ) from None

from embedchain.config import ElasticsearchDBConfig
from embedchain.helpers.json_serializable import register_deserializable
from embedchain.utils.misc import chunks
from embedchain.vectordb.base import BaseVectorDB


@register_deserializable
class ElasticsearchDB(BaseVectorDB):
    """
    Elasticsearch as vector database
    """

    BATCH_SIZE = 100

    def __init__(
        self,
        config: Optional[ElasticsearchDBConfig] = None,
        es_config: Optional[ElasticsearchDBConfig] = None,  # Backwards compatibility
    ):
        """Elasticsearch as vector database.

        :param config: Elasticsearch database config, defaults to None
        :type config: ElasticsearchDBConfig, optional
        :param es_config: `es_config` is supported as an alias for `config` (for backwards compatibility),
        defaults to None
        :type es_config: ElasticsearchDBConfig, optional
        :raises ValueError: No config provided
        """
        if config is None and es_config is None:
            self.config = ElasticsearchDBConfig()
        else:
            if not isinstance(config, ElasticsearchDBConfig):
                raise TypeError(
                    "config is not a `ElasticsearchDBConfig` instance. "
                    "Please make sure the type is right and that you are passing an instance."
                )
            self.config = config or es_config
        if self.config.ES_URL:
            self.client = Elasticsearch(self.config.ES_URL, **self.config.ES_EXTRA_PARAMS)
        elif self.config.CLOUD_ID:
            self.client = Elasticsearch(cloud_id=self.config.CLOUD_ID, **self.config.ES_EXTRA_PARAMS)
        else:
            raise ValueError(
                "Something is wrong with your config. Please check again - `https://docs.embedchain.ai/components/vector-databases#elasticsearch`"  # noqa: E501
            )

        # Call parent init here because embedder is needed
        super().__init__(config=self.config)

    def _initialize(self):
        """
        This method is needed because `embedder` attribute needs to be set externally before it can be initialized.
        """
        logging.info(self.client.info())
        index_settings = {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embeddings": {"type": "dense_vector", "index": False, "dims": self.embedder.vector_dimension},
                }
            }
        }
        es_index = self._get_index()
        if not self.client.indices.exists(index=es_index):
            # create index if not exist
            print("Creating index", es_index, index_settings)
            self.client.indices.create(index=es_index, body=index_settings)

    def _get_or_create_db(self):
        """Called during initialization"""
        return self.client

    def _get_or_create_collection(self, name):
        """Note: nothing to return here. Discuss later"""

    def get(self, ids: Optional[list[str]] = None, where: Optional[dict[str, any]] = None, limit: Optional[int] = None):
        """
        Get existing doc ids present in vector database

        :param ids: _list of doc ids to check for existence
        :type ids: list[str]
        :param where: to filter data
        :type where: dict[str, any]
        :return: ids
        :rtype: Set[str]
        """
        if ids:
            query = {"bool": {"must": [{"ids": {"values": ids}}]}}
        else:
            query = {"bool": {"must": []}}

        if where:
            for key, value in where.items():
                query["bool"]["must"].append({"term": {f"metadata.{key}.keyword": value}})

        response = self.client.search(index=self._get_index(), query=query, _source=True, size=limit)
        docs = response["hits"]["hits"]
        ids = [doc["_id"] for doc in docs]
        doc_ids = [doc["_source"]["metadata"]["doc_id"] for doc in docs]

        # Result is modified for compatibility with other vector databases
        # TODO: Add method in vector database to return result in a standard format
        result = {"ids": ids, "metadatas": []}

        for doc_id in doc_ids:
            result["metadatas"].append({"doc_id": doc_id})

        return result

    def add(
        self,
        documents: list[str],
        metadatas: list[object],
        ids: list[str],
        **kwargs: Optional[dict[str, any]],
    ) -> Any:
        """
        add data in vector database
        :param documents: list of texts to add
        :type documents: list[str]
        :param metadatas: list of metadata associated with docs
        :type metadatas: list[object]
        :param ids: ids of docs
        :type ids: list[str]
        """

        embeddings = self.embedder.embedding_fn(documents)

        for chunk in chunks(
            list(zip(ids, documents, metadatas, embeddings)), self.BATCH_SIZE, desc="Inserting batches in elasticsearch"
        ):  # noqa: E501
            ids, docs, metadatas, embeddings = [], [], [], []
            for id, text, metadata, embedding in chunk:
                ids.append(id)
                docs.append(text)
                metadatas.append(metadata)
                embeddings.append(embedding)

            batch_docs = []
            for id, text, metadata, embedding in zip(ids, docs, metadatas, embeddings):
                batch_docs.append(
                    {
                        "_index": self._get_index(),
                        "_id": id,
                        "_source": {"text": text, "metadata": metadata, "embeddings": embedding},
                    }
                )
            bulk(self.client, batch_docs, **kwargs)
        self.client.indices.refresh(index=self._get_index())

    def split_list(self, input_list, chunk_size):
        """
        将列表按照指定大小切分为二维数组
        """
        return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
    
    def upsert(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[object],
        ids: list[str],
        **kwargs: Optional[dict[str, any]],
    ) -> Any:
        
        documents_list = self.split_list(documents, self.BATCH_SIZE)
        embeddings_list = []
        for documents_chunk in documents_list:
            embeddings_chunk = self.embedder.embedding_fn(documents_chunk)
            embeddings_list.append(embeddings_chunk)
        
        embeddings = [item for sublist in embeddings_list for item in sublist]

        for chunk in chunks(
            list(zip(ids, documents, metadatas, embeddings)), self.BATCH_SIZE, desc="Inserting batches in elasticsearch"
        ):  # noqa: E501
            ids, docs, metadatas, embeddings = [], [], [], []
            for id, text, metadata, embedding in chunk:
                ids.append(id)
                docs.append(text)
                metadatas.append(metadata)
                embeddings.append(embedding)

            batch_docs = []
            for id, text, metadata, embedding in zip(ids, docs, metadatas, embeddings):
                batch_docs.append(
                    {
                        "_index": self._get_index(),
                        "_id": id,
                        "_op_type": "update",
                        "doc": {"text": text, "metadata": metadata, "embeddings": embedding},
                        "doc_as_upsert": True
                    }
                )
            bulk(self.client, batch_docs)
        self.client.indices.refresh(index=self._get_index())

    def upsert_structure(
        self,
        embedding: list[float],
        document: str,
        metadata: object,
        id: str,
        is_deleted: bool,
        **kwargs: Optional[dict[str, any]],
    ) -> Any:
        
        embedding = self.embedder.embedding_fn(document)
        result = self.check_if_exist(id)
        if result and is_deleted:
            self.client.delete(index=self._get_index(), id=id)
        else:
            if not is_deleted:  
                batch_docs = []
                batch_docs.append(
                    {
                        "_index": self._get_index(),
                        "_id": id,
                        "_op_type": "update",
                        "doc": {"text": document, "metadata": metadata, "embeddings": embedding},
                        "doc_as_upsert": True
                    }
                )
                bulk(self.client, batch_docs)
                self.client.indices.refresh(index=self._get_index())

    def check_if_exist(self, id: str) -> bool:
        """
        check if a document with the given id exists in the database
        :param id: id of document
        :type id: str
        """
        try:
            return self.client.exists(index=self._get_index(), id=id)
        except Exception as e:
            return False
        

    def publish_knowledge(self, conditions: dict):
        query = {
            "query": {
                "bool": {
                    "must": [{"match": {key: value}} for key, value in conditions.items()]
                }
            },
            "script": {
                "source": """
                    if (ctx._source.containsKey('is_public')) {
                        ctx._source.is_public = 1;
                    }
                """,
                "lang": "painless"
            }
        }
        self.client.update_by_query(index=self._get_index(), body=query)

    def publish_knowledge(self, conditions: dict):
        query = {
            "query": {
                "bool": {
                    "must": [{"match": {key: value}} for key, value in conditions.items()]
                }
            },
            "script": {
                "source": """
                    if (ctx._source.containsKey('is_public')) {
                        ctx._source.is_public = 1;
                    }
                """,
                "lang": "painless"
            }
        }
        self.client.update_by_query(index=self._get_index(), body=query)

    def enable_docs(self, doc_ids: list,status: int=1):
        query = {
            "query": {
                "terms": {
                    "metadata.doc_id": doc_ids
                }
            },
            "script": {
                "source": "ctx._source.metadata.status = params.status",
                "lang": "painless",
                "params": {
                    "status": status
                }
            }
        }
        self.client.update_by_query(index=self._get_index(), body=query)

    def delete(
        self,
        conditionsList: list
    ):
        actions = []
        for conditions in conditionsList:
            actions.append(
                {
                    "_index": self._get_index(),
                    "_op_type": "delete",
                    "_query": {
                        "bool": {
                            "must": [
                                {"match": {field: value} for field, value in conditions.items()}
                            ]
                        }
                    }
                }
            )
        bulk(self.client, actions)

    def _delete_by_query(
        self,
        conditions: dict    
    ):
        query = {
            "query": {
                "bool": {
                    "must": [
                         {"match": {field: value}} for field, value in conditions.items()
                    ]
                }
            }
        }

        # 执行删除操作
        self.client.delete_by_query(index=self._get_index(), body=query)
            
    def multi_field_match_query(
        self,
        input_query: list[str],
        and_conditions: dict[str, any],
        or_conditions: dict[str, any],
        size: int = 5
    ) -> Union[list[tuple[str, dict]], list[str]]:
        # _source = ["text", "metadata"]

        # query = {
        #     "min_score": 1,
        #     "query": {"bool": {"should": [{"match": {"text": input_query}}]}}
        # }
        # if and_conditions is not None:
        #     for field, value in and_conditions.items():
        #         if isinstance(value, list):
        #             query["query"]["bool"].setdefault("must", []).append({"terms": {field: value}})
        #         else:
        #             query["query"]["bool"].setdefault("must", []).append({"match": {field: value}})

        # if or_conditions is not None:
        #     for field, value in or_conditions.items():
        #         if isinstance(value, list):
        #             query["query"]["bool"].setdefault("should", []).append({"terms": {field: value}})
        #         else:
        #             query["query"]["bool"].setdefault("should", []).append({"match": {field: value}})
        # response = self.client.search(index=self._get_index(), body=query, _source=_source, size=22)
        
        # ids = []
        # contexts = []
        # docs = response["hits"]["hits"]
        # i = 1
        # for doc in docs:
        #     if i <= 2:
        #         context = doc["_source"]["text"]
        #         contexts.append(context)
        #         i += 1
        #     else:
        #         ids.append(doc["_id"])
        
        # input_query_vector = self.embedder.embedding_fn(input_query)
        # query_vector = input_query_vector[0]

        # knn_query = {
        #     "min_score": 1,
        #     "query": {"ids":{"values":ids}},
        #     "knn": {
        #     "field": "embeddings",
        #     "query_vector": query_vector,
        #     "k": 5,
        #     "num_candidates": 20
        #     }
        # }  

        # response = self.client.search(index=self._get_index(), body=knn_query, _source=_source, size=size-2)


        input_query_vector = self.embedder.embedding_fn(input_query)
        query_vector = input_query_vector[0]

        _source = ["text", "metadata"]
        
        query = {
            "min_score": 1,
            "query": {"bool": {"should": [{"match": {"text": input_query}}]}},
            "knn": {
            "field": "embeddings",
            "query_vector": query_vector,
            "k": 5,
            "num_candidates": 20
            }
        }
        if and_conditions is not None:
            for field, value in and_conditions.items():
                if isinstance(value, list):
                    query["query"]["bool"].setdefault("must", []).append({"terms": {field: value}})
                else:
                    query["query"]["bool"].setdefault("must", []).append({"match": {field: value}})
        
        if or_conditions is not None:
            for field, value in or_conditions.items():
                if isinstance(value, list):
                    query["query"]["bool"].setdefault("should", []).append({"terms": {field: value}})
                else:
                    query["query"]["bool"].setdefault("should", []).append({"match": {field: value}})
        response = self.client.search(index=self._get_index(), body=query, _source=_source, size=size)
        
        # query = {
        #     "script_score": {
        #         "query": {"bool": {"must": [{"exists": {"field": "text"}}]}},
        #         "script": {
        #             "source": "cosineSimilarity(params.input_query_vector, 'embeddings') + 1.0",  ##es不允许分数为负数
        #             "params": {"input_query_vector": query_vector},
        #         },
        #     }
        # }
        # if and_conditions is not None:
        #     for field, value in and_conditions.items():
        #         if isinstance(value, list):
        #             query["script_score"]["query"]["bool"].setdefault("must", []).append({"terms": {field: value}})
        #         else:
        #             query["script_score"]["query"]["bool"].setdefault("must", []).append({"match": {field: value}})
        # if or_conditions is not None:
        #     for field, value in or_conditions.items():
        #         if isinstance(value, list):
        #             query["script_score"]["query"]["bool"].setdefault("should", []).append({"terms": {field: value}})
        #         else:
        #             query["script_score"]["query"]["bool"].setdefault("should", []).append({"match": {field: value}})

        # response = self.client.search(index=self._get_index(), query=query, _source=_source, size=size)
        
        docs = response["hits"]["hits"]
        contexts = []
        for doc in docs:
            context = doc["_source"]["text"]
            contexts.append(context)
            print(doc["_score"])
            print(context)
        return contexts
        

    def query(
        self,
        input_query: list[str],
        n_results: int,
        where: dict[str, any],
        citations: bool = False,
        **kwargs: Optional[dict[str, Any]],
    ) -> Union[list[tuple[str, dict]], list[str]]:
        """
        query contents from vector database based on vector similarity

        :param input_query: list of query string
        :type input_query: list[str]
        :param n_results: no of similar documents to fetch from database
        :type n_results: int
        :param where: Optional. to filter data
        :type where: dict[str, any]
        :return: The context of the document that matched your query, url of the source, doc_id
        :param citations: we use citations boolean param to return context along with the answer.
        :type citations: bool, default is False.
        :return: The content of the document that matched your query,
        along with url of the source and doc_id (if citations flag is true)
        :rtype: list[str], if citations=False, otherwise list[tuple[str, str, str]]
        """
        input_query_vector = self.embedder.embedding_fn(input_query)
        query_vector = input_query_vector[0]

        # `https://www.elastic.co/guide/en/elasticsearch/reference/7.17/query-dsl-script-score-query.html`
        query = {
            "script_score": {
                "query": {"bool": {"must": [{"exists": {"field": "text"}}]}},
                "script": {
                    "source": "cosineSimilarity(params.input_query_vector, 'embeddings') + 1.0",
                    "params": {"input_query_vector": query_vector},
                },
            }
        }
        if where is not None:
            query["script_score"]["query"] = {"bool": {"must": [{"match": {field: value} for field, value in where.items()}]}}
        # if "app_id" in where:
        #     app_id = where["app_id"]
        #     query["script_score"]["query"] = {"match": {"metadata.app_id": app_id}}
        _source = ["text", "metadata"]
        response = self.client.search(index=self._get_index(), query=query, _source=_source, size=n_results)
        docs = response["hits"]["hits"]
        contexts = []
        for doc in docs:
            context = doc["_source"]["text"]
            if citations:
                metadata = doc["_source"]["metadata"]
                metadata["score"] = doc["_score"]
                contexts.append(tuple((context, metadata)))
            else:
                contexts.append(context)
                
        return contexts

    def set_collection_name(self, name: str):
        """
        Set the name of the collection. A collection is an isolated space for vectors.

        :param name: Name of the collection.
        :type name: str
        """
        if not isinstance(name, str):
            raise TypeError("Collection name must be a string")
        self.config.collection_name = name

    def count(self) -> int:
        """
        Count number of documents/chunks embedded in the database.

        :return: number of documents
        :rtype: int
        """
        query = {"match_all": {}}
        response = self.client.count(index=self._get_index(), query=query)
        doc_count = response["count"]
        return doc_count

    def reset(self):
        """
        Resets the database. Deletes all embeddings irreversibly.
        """
        # Delete all data from the database
        if self.client.indices.exists(index=self._get_index()):
            # delete index in Es
            self.client.indices.delete(index=self._get_index())

    def _get_index(self) -> str:
        """Get the Elasticsearch index for a collection

        :return: Elasticsearch index
        :rtype: str
        """
        # NOTE: The method is preferred to an attribute, because if collection name changes,
        # it's always up-to-date.
        return f"{self.config.collection_name}_{self.embedder.vector_dimension}".lower()

    def delete(self, where):
        """Delete documents from the database."""
        query = {"query": {"bool": {"must": []}}}
        for key, value in where.items():
            query["query"]["bool"]["must"].append({"term": {f"metadata.{key}.keyword": value}})
        self.client.delete_by_query(index=self._get_index(), body=query)
        self.client.indices.refresh(index=self._get_index())
