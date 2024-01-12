from typing import Optional,Any,Union

class BaseModel:
    """
    数据库表结构
    database : 数据库名
    table_name : 表名
    table_description : 表注释
    primary_key : 主键
    columns :{'id':{'description':'主键','is_writable':False}, 'name':{'description':'名字'}, 'age':{'description':'年龄'}, 'address':{'description':None}}
    """
    database: str
    table_name: str
    table_description: str = None
    primary_key: str = 'id' #唯一键，多个字段可用,分隔 
    columns: Optional[dict[str,dict[str,Any]]] = None

    def _to_schema(self):
        schema = {}
        schema["database"] = self.database
        schema["table_name"] = self.table_name
        schema['primary_key'] = self.primary_key


        if self.table_description is not None:
            schema['table_description'] = self.table_description
    
        if self.columns is not None:
            self.check_columns()
            schema["columns"] = self.columns
    
        return schema
    
    def check_columns(self):
        for column,value in self.columns.items():
            if value is None:
                raise ValueError(f'column {column} is None,the dict type is required.Key value could be set description,is_writable')
            else:
                if 'description' not in value:
                    raise ValueError(f'column {column} is required a key value named [description]')