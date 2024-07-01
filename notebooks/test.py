import os
from embedchain import App

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://saiwei-lab-gpt.valsun.cn/openai"
os.environ["OPENAI_API_KEY"] = "4d2eacfd380746a4af23ed23c410c950"
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"

app = App.from_config(config_path="./data/example_config.yaml")
# import os

# def list_files(directory):
#     files = []
#     for root, _, filenames in os.walk(directory):
#         for filename in filenames:
#             files.append(os.path.join(root, filename))
#     return files

# directory = 'C:/Users/sw/Desktop/人力资源/人力资源规章制度'
# files = list_files(directory)
# for file in files:
#     doc_id = app.upsert(
#         source=file,
#         data_type="pdf_file")
#     print(doc_id)

doc_id = app.upsert(source="C:/Users/sw/Desktop/人力资源/人力资源规章制度/SD-BF-HR-003劳动合同管理规定.pdf",data_type="pdf_file")
print(doc_id)
# a = app.multi_field_match_query("去年8月份， 销售192部的销售额是多少",and_conditions={"metadata.app_id":1,"metadata.status":1})
# for i in a:
#     print(i)

# import requests  
# import os  
# from mimetypes import guess_type  
# import pprint
  
# # 设置请求的URL  
# base_url = 'https://aiagent-test.sailvan.com/v1'  
# url = base_url + '/document/create/file'
# def list_files(directory):
#     files = []
#     for root, _, filenames in os.walk(directory):
#         for filename in filenames:
#             files.append(os.path.join(root, filename))
#     return files

# directory = 'C:/Users/sw/Desktop/人力资源/晋江人力资源'
# file_paths = list_files(directory)

# files = [
#   ("files[]" , (os.path.basename(file_path), 
#                 open(file_path, "rb"))) for file_path in file_paths  
# ]
# params = {
#     "knowledge_id":2
# }    
# headers = {
#     "Auth-AppKey":"RNWRXwFbkiMMYyMYCUsdgpSHTswYJyHA"
# }
# # 发送POST请求，包含文件  

# response = requests.post(url, params,files=files,headers=headers)  
# # 检查请求是否成功  
# pprint.pprint(response.json())
# def auto_assign_value(value):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             MyClass._value = value  # 修改 MyClass() 的 _value 属性
#             return func(*args, **kwargs)
#         return wrapper
#     return decorator

# class MyClass:
#     _value = None  # 类属性，初始化默认值为 None

#     @classmethod
#     @auto_assign_value(100)  # 使用装饰器，并指定初始值为 100
#     def method1(cls):
#         print("Method 1 called, value:", cls._value)

#     @classmethod
#     @auto_assign_value(200)  # 使用装饰器，并指定初始值为 200
#     def method2(cls):
#         print("Method 2 called, value:", cls._value)

# class AnotherClass:
#     @classmethod
#     @auto_assign_value(300)  # 使用装饰器，并指定初始值为 300
#     def some_method(cls):
#         print("Some method called, value:", MyClass._value)

# # 调用方法
# MyClass.method1()  # 输出：Method 1 called, value: 100
# MyClass.method2()  # 输出：Method 2 called, value: 200
# AnotherClass.some_method()  # 输出：Some method called, value: 300







