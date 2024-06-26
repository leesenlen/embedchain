import os
from embedchain import App

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://openai-sailvan-canadaeast-proxy.valsun.cn/openai"
os.environ["OPENAI_API_KEY"] = "fe55676aedad4ee090f6cf3830643cda"
# os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"

app = App.from_config(config_path="./data/example_config.yaml")
import os

# def list_files(directory):
#     files = []
#     for root, _, filenames in os.walk(directory):
#         for filename in filenames:
#             files.append(os.path.join(root, filename))
#     return files
#
# directory = 'C:/Users/sw/Desktop/人力资源/人力资源规章制度'
# files = list_files(directory)
# for file in files:
#     doc_id = app.upsert(
#         source=file,
#         data_type="pdf_file")
#     print(doc_id)


a = app.multi_field_match_query("工牌丢了应该怎么办")
for i in a:
    print(i)
