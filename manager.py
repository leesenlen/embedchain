import os
from embedchain.app import App

es_host = os.getenv("ES_HOST")
es_key = os.getenv("ES_KEY")

config = {
    'embedder':
        {
            'provider': 'azure_openai',
            'config':
                {
                    'model': 'text-embedding-3-large',
                    'deployment_name': 'text-embedding-3-large',
                    'vector_dimension': 1024
                }
        },
    'vectordb': {
        'config': {
            'collection_name': 'aiagent_vectordb_v2', 'es_url': [es_host],
            'basic_auth': ('elastic', es_key), 'verify_certs': False}, 'provider': 'elasticsearch'},
    'chunker': {
        'chunk_size': 1024, 'chunk_overlap': 30, 'length_function': 'len', 'min_chunk_size': 30
    }
}

embedder = App.from_config(config=config)

if __name__ == "__main__":
    # file_path = "C:\\Users\\admin\\Documents\\docs\\外派人员管理制.pdf"
    # file_path = "E:\\project\\eip-docs\\doc\\version.md"
    # metadata = {'system_doc_id': 945,
    #             'app_id': 2,
    #             'knowledge_id': 438,
    #             'subject': '版本changelog',
    #             'link': ''}
    # doc_id = embedder.upsert(source=file_path, metadata=metadata)

    import requests
    import pandas as pd
    test_queryes = ["绩效等级分为哪几个", "绩效等级有哪几个", "工牌丢了怎么办", "工牌补办要多少钱", "寻佑兰是哪个部门的", "当前是哪个版本，有什么新功能",
                    "员工离职流程是怎样的", "驻外人员有什么福利", "外派人员怎么打卡", "境外外派补贴标准是什么", "我昨天加班到9：20，今天可以几点打卡上班？"
                    ,"周五加班到10点，下周一可以弹性打卡上班么？"]
    # test_queryes = ["我昨天加班到9：20，今天可以几点打卡上班？"]
    df = pd.DataFrame(columns=['question', 'subject', 'origin_score', 'rerank_score', 'doc'])
    for query in test_queryes:
        result = embedder.multi_field_match_query(query, and_conditions={'metadata.knowledge_id': [438]})
        docs = [each["context"] for each in result]
        origin_scores = [each["score"] for each in result]
        subjects = [each["metadata"]["subject"] for each in result]
        scores = [each["rerank_score"] for each in result]
        for doc, origin_score, score, subject in zip(docs, origin_scores, scores, subjects):
            df = pd.concat([df, pd.DataFrame(
                {'question': [query], 'doc': [doc], 'origin_score': [origin_score], 'rerank_score': [score], "subject": [subject]})])
            # df = df.append({"question": query, "doc": doc["context"], "origin_score": doc["score"], "rerank_score": score}, ignore_index=True)
    df.to_excel('data.xlsx', index=False)
