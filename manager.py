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
            'collection_name': 'aiagent_vectordb', 'es_url': [es_host],
            'basic_auth': ('elastic', es_key), 'verify_certs': False}, 'provider': 'elasticsearch'},
    'chunker': {
        'chunk_size': 1024, 'chunk_overlap': 30, 'length_function': 'len', 'min_chunk_size': 30
    }
}

embedder = App.from_config(config=config)

if __name__ == "__main__":
    file_path = "C:\\Users\\admin\\Documents\\docs\\30_19441887.docx"
    metadata = {'system_doc_id': 942,
                'app_id': 2,
                'knowledge_id': 403,
                'subject': '14_11245820',
                'link': 'www.google.com'}
    doc_id = embedder.upsert(source=file_path, metadata=metadata)
