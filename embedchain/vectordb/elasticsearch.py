import logging
import requests
import os
from typing import Any, Optional, Union
from embedchain.rag.nlp.search import ESQueryBuilder

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
from embedchain.rag.nlp.query import EsQueryer
import tiktoken
from datetime import datetime
import hashlib


@register_deserializable
class ElasticsearchDB(BaseVectorDB):
    """
    Elasticsearch as vector database
    """

    BATCH_SIZE = 50

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
                "Something is wrong with your config. Please check again - `https://docs.embedchain.ai/components/vector-databases#elasticsearch`"
                # noqa: E501
            )
        self.es_query_engine = ESQueryBuilder(self.client)
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
                list(zip(ids, documents, metadatas, embeddings)), self.BATCH_SIZE,
                desc="Inserting batches in elasticsearch"
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
            extra_data=None,
            **kwargs: Optional[dict[str, any]],
    ) -> Any:
        if not extra_data:
            extra_data = [] * len(ids)
        documents_list = self.split_list(documents, self.BATCH_SIZE)
        embeddings_list = []
        for documents_chunk in documents_list:
            embeddings_chunk = self.embedder.embedding_fn(documents_chunk)
            embeddings_list.append(embeddings_chunk)

        embeddings = [item for sublist in embeddings_list for item in sublist]

        for chunk in chunks(
                list(zip(ids, documents, metadatas, embeddings, extra_data)), self.BATCH_SIZE,
                desc="Inserting batches in elasticsearch"
        ):  # noqa: E501
            ids, docs, metadatas, embeddings, extra_datas = [], [], [], [], []
            for id, text, metadata, embedding, extra in chunk:
                ids.append(id)
                docs.append(text)
                metadatas.append(metadata)
                embeddings.append(embedding)
                extra_datas.append(extra)

            batch_docs = []
            for id, text, metadata, embedding, extra in zip(ids, docs, metadatas, embeddings, extra_datas):
                metadata['tokens_num'] = self.num_tokens_from_string(text, "cl100k_base")
                batch_docs.append(
                    {
                        "_index": self._get_index(),
                        "_id": id,
                        "_op_type": "update",
                        "doc": {"text": text, "metadata": metadata, "embeddings": embedding, **extra},
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
        result = self.check_if_exist(id)
        if result and is_deleted:
            self.client.delete(index=self._get_index(), id=id)
        else:
            if not is_deleted:
                embedding = self.embedder.embedding_fn([document])
                metadata['tokens_num'] = self.num_tokens_from_string(document, "cl100k_base")
                batch_docs = []
                batch_docs.append(
                    {
                        "_index": self._get_index(),
                        "_id": id,
                        "_op_type": "update",
                        "doc": {"text": document, "metadata": metadata, "embeddings": embedding[0]},
                        "doc_as_upsert": True
                    }
                )
                bulk(self.client, batch_docs)
        self.client.indices.refresh(index=self._get_index())

    def upsert_document(self, document: str, id: str = None,
                        metadata: dict = {}):
        hash = hashlib.sha256(document.encode()).hexdigest()
        embedding = self.embedder.embedding_fn([document])
        if id is None:
            if "doc_id" in metadata:
                id = metadata['doc_id'] + "-" + hash
            else:
                id = str(metadata['knowledge_id']) + "-" + hash
                metadata['doc_id'] = id
                metadata['hash'] = hash
        else:
            response = self.client.get(index=self._get_index(), id=id)
            metadata = response["_source"]["metadata"]

        metadata['tokens_num'] = self.num_tokens_from_string(document, "cl100k_base")
        update_body = {
            "doc": {"text": document, "metadata": metadata, "embeddings": embedding[0]},
            "doc_as_upsert": True  # 如果文档不存在则插入
        }

        # 执行 upsert 操作
        self.client.update(index=self._get_index(), id=id, body=update_body)
        self.client.indices.refresh(index=self._get_index())
        return {
            'segment_id': id,
            'app_id': metadata['app_id'],
            'doc_id': metadata['system_doc_id'] if 'system_doc_id' in metadata else 0,
            'knowledge_id': metadata['knowledge_id'],
            'vector_doc_id': metadata['doc_id'],
            'hash': metadata['hash'],
            'status': 1,
            'url': metadata['url'] if 'url' in metadata else '',
            'tokens_num': metadata['tokens_num'],
            'data_type': metadata['data_type'] if 'data_type' in metadata else '',
            'subject': metadata['subject'] if 'subject' in metadata else '',
            'link': metadata['link'] if 'link' in metadata else '',
            'text': document
        }

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
        self.client.indices.refresh(index=self._get_index())

    def enable_docs(self, doc_ids: list, status: int = 1):
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
        self.client.indices.refresh(index=self._get_index())

    def enable_segment(self, segment_id: str, status: int = 1):
        update_body = {
            "doc": {
                "metadata": {
                    "status": status
                }
            }
        }

        # 执行更新操作
        self.client.update(index=self._get_index(), id=segment_id, body=update_body)
        self.client.indices.refresh(index=self._get_index())

    def _delete_by_query(
            self,
            conditions: dict
    ):
        query = {"query": {"bool": {"must": []}}}
        for key, value in conditions.items():
            query["query"]["bool"]["must"].append({"term": {key: value}})
        self.client.delete_by_query(index=self._get_index(), body=query)
        self.client.indices.refresh(index=self._get_index())

    def _delete_by_id(self, id: str):
        self.client.delete(index=self._get_index(), id=id)
        self.client.indices.refresh(index=self._get_index())

    def paged_query(self,
                    and_conditions: dict[str, any],
                    page_number: int = 1,
                    page_size: int = 10):
        """
        分页查询
        """
        from_index = (page_number - 1) * page_size

        _source = ["text", "metadata"]
        query = {
            "query": {"bool": {"must": []}},
            "sort": [
                {"metadata.segment_number": {"order": "asc"}}
            ],
            "from": from_index,
            "size": page_size
        }
        if and_conditions is not None:
            for field, value in and_conditions.items():
                if isinstance(value, list):
                    query["query"]["bool"].setdefault("must", []).append({"terms": {field: value}})
                else:
                    query["query"]["bool"].setdefault("must", []).append({"match": {field: value}})

        response = self.client.search(index=self._get_index(), body=query, source=_source)
        docs = response["hits"]["hits"]
        total_count = response['hits']['total']['value']
        contexts = []
        for doc in docs:
            context = doc["_source"]["text"]
            segment_id = doc["_id"]
            knowledge_id = doc["_source"]["metadata"]["knowledge_id"] if "knowledge_id" in doc["_source"][
                "metadata"] else 0
            app_id = doc["_source"]["metadata"]["app_id"] if "app_id" in doc["_source"]["metadata"] else 0
            vecotr_doc_id = doc["_source"]["metadata"]["doc_id"] if "doc_id" in doc["_source"]["metadata"] else ''
            status = doc["_source"]["metadata"]["status"] if "status" in doc["_source"]["metadata"] else 0
            doc_id = doc["_source"]["metadata"]["system_doc_id"] if "system_doc_id" in doc["_source"]["metadata"] else 0
            hash = doc["_source"]["metadata"]["hash"] if "hash" in doc["_source"]["metadata"] else ''
            url = doc["_source"]["metadata"]["url"] if "url" in doc["_source"]["metadata"] else ''
            tokens_num = doc["_source"]["metadata"]["tokens_num"] if "tokens_num" in doc["_source"]["metadata"] else 0
            data_type = doc["_source"]["metadata"]["data_type"] if "data_type" in doc["_source"]["metadata"] else ''
            subject = doc["_source"]["metadata"]["subject"] if "subject" in doc["_source"]["metadata"] else ''
            link = doc["_source"]["metadata"]["link"] if "link" in doc["_source"]["metadata"] else ''

            map = {
                'segment_id': segment_id,
                'app_id': app_id,
                'doc_id': doc_id,
                'knowledge_id': knowledge_id,
                'vector_doc_id': vecotr_doc_id,
                'hash': hash,
                'status': status,
                'url': url,
                'tokens_num': tokens_num,
                'data_type': data_type,
                'subject': subject,
                'link': link,
                'text': context
            }
            contexts.append(map)
        return {'items': contexts, 'page': page_number, 'page_size': page_size, 'total': total_count}

    def multi_field_match_query(
            self,
            input_query: list[str],
            and_conditions: dict[str, any],
            match_weight: float = 0.05,
            knn_weight: float = 0.5,
            knowledge_tokens: int = 6000,
            model: str = 'gpt-3.5-turbo',
            knn_threshold: float = 0.3,
            match_threshold: float = 1,
            rerank=True
    ) -> Union[list[tuple[str, dict]], list[str]]:
        # 起始时间
        start_time = datetime.now()
        # knn与关键字一起时加过滤条件需要都加上，只加在query里knn并不会生效
        input_query_vector = self.embedder.embedding_fn(input_query)
        logging.info(f"查询作向量化耗时：{(datetime.now() - start_time).total_seconds()}")
        result = self.es_query_engine.search(input_query, and_conditions, self._get_index(), input_query_vector[0],
                                             **{"knn_threshold": knn_threshold, "match_threshold": match_threshold,
                                                "top_k": 16, "match_weight": match_weight, "knn_weight": knn_weight})
        # TODO: 这里result返回了很多额外信息, 后续可以用来做rerank，以及坐标展示等等
        contexts = []
        sum_tokens = 0
        # 默认使用rerank
        if rerank and os.getenv("RERANK_URL", ""):
            self.rerank(input_query[0], result, discard_threshold=0.01, top_k=10)
        for i, _id in enumerate(result.ids):
            context = result.field[_id]
            tokens_num = self.num_tokens_from_messages(context["content_with_weight"], model)
            sum_tokens += tokens_num
            context["context"] = context["content_with_weight"]
            context["id"] = _id
            context["tokens_num"] = tokens_num
            context["knowledge_id"] = context["metadata"]["knowledge_id"]
            context["score"] = result.scores[i]
            if rerank and os.getenv("RERANK_URL", ""):
                context["rerank_score"] = result.rerank_scores[i]
            del context["content_with_weight"]
            del context["content_ltks"]
            contexts.append(context)
            if sum_tokens > knowledge_tokens:
                break
        return contexts

        # 旧版rag
        # query_vector = input_query_vector[0]
        # _source = ["text", "metadata"]
        # if match_weight == 1 and knn_weight == 0:
        #     contexts = self.match_query(input_query[0], _source, and_conditions, match_weight, model)
        #     logging.info(f"关键字搜索耗时：{(datetime.now() - start_time).total_seconds()}")
        # elif match_weight == 0 and knn_weight == 1:
        #     contexts = self.knn_query(query_vector, _source, and_conditions, knn_weight, model)
        #     logging.info(f"knn语义搜索耗时：{(datetime.now() - start_time).total_seconds()}")
        # else:
        #     match_contexts = self.match_query(input_query[0], _source, and_conditions, match_weight, model)
        #     logging.info(f"关键字搜索耗时：{(datetime.now() - start_time).total_seconds()}")
        #     knn_contexts = self.knn_query(query_vector, _source, and_conditions, knn_weight, model)
        #     logging.info(f"knn语义搜索耗时：{(datetime.now() - start_time).total_seconds()}")
        #     contexts = self.reciprocal_rank_fusion(match_contexts, knn_contexts)
        #     logging.info(f"混合搜索耗时：{(datetime.now() - start_time).total_seconds()}")
        #
        # # token计数不能超过knowledge_tokens，默认6000
        # # es获取的文档个数不能超过10
        # max_size = 10
        # sum_tokens = 0
        # size = 0
        # for context in contexts:
        #     sum_tokens += context["tokens_num"]
        #     size += 1
        #     if size > max_size:
        #         size -= 1
        #         break
        #     if sum_tokens > knowledge_tokens:
        #         size -= 1
        #         break
        #
        # return contexts[:size]

    def reciprocal_rank_fusion(self, match_contexts, knn_contexts):
        """
        rrf算法输出混合最佳搜索
        """
        # 初始化结果字典
        fused_ranking = {}
        # 统计每个文档的倒数排名之和
        for rank_list in [match_contexts, knn_contexts]:
            for rank, map in enumerate(rank_list):
                if map['id'] not in fused_ranking:
                    map['rank'] = 0
                    fused_ranking[map['id']] = map
                fused_ranking[map['id']]['rank'] += 1 / (rank + 1)

        sorted_list = sorted(fused_ranking.values(), key=lambda x: x["rank"], reverse=True)
        return sorted_list

    def match_query(self, input_query: str,
                    _source: list[str],
                    and_conditions: dict[str, any],
                    match_weight: float,
                    model: str):
        """
        关键字搜索
        """
        match_query = {
            "query": {"bool": {"must": [{"match": {"text": {"query": input_query, "boost": match_weight}}}]}},
        }
        if and_conditions is not None:
            for field, value in and_conditions.items():
                if isinstance(value, list):
                    match_query["query"]["bool"].setdefault("must", []).append({"terms": {field: value}})
                else:
                    match_query["query"]["bool"].setdefault("must", []).append({"match": {field: value}})

        response = self.client.search(index=self._get_index(), body=match_query, _source=_source, size=50)
        docs = response["hits"]["hits"]
        contexts = []
        for doc in docs:
            context = doc["_source"]["text"]
            id = doc["_id"]
            knowledge_id = doc["_source"]["metadata"]["knowledge_id"] if "knowledge_id" in doc["_source"][
                "metadata"] else 0
            tokens_num = self.num_tokens_from_messages(context, model)
            # link = ""
            # if "link" in doc["_source"]["metadata"] and doc["_source"]["metadata"]["link"] != "":
            #     link = "\n\n引用来源链接地址：" + doc["_source"]["metadata"]["link"]
            map = {"context": context, "id": id, "knowledge_id": knowledge_id, "tokens_num": tokens_num}
            contexts.append(map)
        return contexts

    def knn_query(self, query_vector,
                  _source: list[str],
                  and_conditions: dict[str, any],
                  knn_weight: float,
                  model: str):
        """
        knn搜索
        """
        knn_query = {
            "knn": {
                "field": "embeddings",
                "query_vector": query_vector,
                "k": 50,
                "num_candidates": 100,
                "boost": knn_weight
            }
        }
        if and_conditions is not None:
            knn_query["knn"]["filter"] = []
            for field, value in and_conditions.items():
                if isinstance(value, list):
                    knn_query["knn"]["filter"].append({"terms": {field: value}})
                else:
                    knn_query["knn"]["filter"].append({"term": {field: value}})

        response = self.client.search(index=self._get_index(), body=knn_query, _source=_source, size=50)
        docs = response["hits"]["hits"]
        contexts = []
        for doc in docs:
            context = doc["_source"]["text"]
            id = doc["_id"]
            knowledge_id = doc["_source"]["metadata"]["knowledge_id"] if "knowledge_id" in doc["_source"][
                "metadata"] else 0
            tokens_num = self.num_tokens_from_messages(context, model)
            # link = ""
            # if "link" in doc["_source"]["metadata"] and doc["_source"]["metadata"]["link"] != "":
            #     link = "\n\n引用来源链接地址：" + doc["_source"]["metadata"]["link"]
            map = {"context": context, "id": id, "knowledge_id": knowledge_id, "tokens_num": tokens_num}
            contexts.append(map)
        return contexts

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        tokens_num = len(encoding.encode(string))
        return tokens_num

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(messages))

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
            query["script_score"]["query"] = {
                "bool": {"must": [{"match": {field: value} for field, value in where.items()}]}}
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
        return self.config.collection_name

    def delete(self, where):
        """Delete documents from the database."""
        query = {"query": {"bool": {"must": []}}}
        for key, value in where.items():
            query["query"]["bool"]["must"].append({"term": {f"metadata.{key}.keyword": value}})
        self.client.delete_by_query(index=self._get_index(), body=query)
        self.client.indices.refresh(index=self._get_index())

    def rerank(self, query, docs, discard_threshold=0.01, top_k=10) -> None:
        """
        对搜索到的结果，进行rerank重排序，剔除置信度较低的结果
        :param docs: 文档
        :param discard_threshold: 丢弃阈值，置信度低于该值的结果将被丢弃
        :param top_k: 保留的结果数量
        """
        rerank_url = os.getenv("RERANK_URL")
        contents = []
        for i, _id in enumerate(docs.ids):
            contents.append(docs.field[_id]["content_with_weight"])
        rerank_scores = requests.post(rerank_url, json={"question": query, "docs": contents}).json()["scores"]
        combined_list = [(rerank_score, _id, score) for rerank_score, _id, score in zip(rerank_scores, docs.ids, docs.scores) if
                         rerank_score >= discard_threshold]
        combined_list.sort(key=lambda x: x[0], reverse=True)
        if len(combined_list) > top_k:
            combined_list = combined_list[:top_k]
        rerank_scores, docs.ids, docs.scores = zip(*combined_list)
        docs.rerank_scores = list(rerank_scores)


    def rerank_with_model(self):
        ...
