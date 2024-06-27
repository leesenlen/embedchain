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


    test_queryes = ["绩效等级分为哪几个", "绩效等级有哪几个", "工牌丢了怎么办", "工牌补办要多少钱", "寻佑兰是哪个部门的", "当前是哪个版本，有什么新功能"]
    for query in test_queryes:
        print(query)
        result = embedder.multi_field_match_query(query, and_conditions={'metadata.knowledge_id': [438]})
        for each in result:
            print(each)
        print("=" * 100)
