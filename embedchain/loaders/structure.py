import json

from embedchain.loaders.base_loader import BaseLoader
from embedchain.utils.misc import clean_string
from embedchain.models.base import BaseModel


class StructureLoader(BaseLoader):
   
    def load_data(self, data: dict):
        json_data = {}
        if isinstance(data['json'], str):
            json_data = json.loads(data['json'], ensure_ascii=False)
        json_data = data['json']
        table = data['model']
        result = []
        if isinstance(table, BaseModel):
            doc_content = clean_string(self.parse_row(json_data,table))
            result.append({"primary_key": json_data[table.primary_key],"content": doc_content, "meta_data": {"fields":json_data}})
            return result
        else:
            raise Exception("Model is not a BaseModel")
        
    def parse_row(self,row:dict,table: BaseModel):        
        content = {}
        if table.table_schema is not None:
            for column,value in table.table_schema.items():
                if table.ignore_fields is not None:
                    split = table.ignore_fields.split(',')
                    if column in split:
                        continue
                if value is not None and "field_name" in value:
                    if "enum_translate" in value:
                        enum_translate = self.convert_to_dict(value['enum_translate'])
                        content[value['field_name']] = enum_translate[row[column]]
                    else:
                        content[value['field_name']] = row[column]
        else:
            content = row
        
        return ",".join([f"{key}为{value}" for key, value in content.items()])
    
    def convert_to_dict(self, enum_translate:str):
        pairs = enum_translate.split(',')

        my_dict = {}

        # 遍历键值对列表，将每对键值对解析并添加到字典中
        for pair in pairs:
            key, value = pair.split(':')
            my_dict[key] = value
        return my_dict
 



        
        




