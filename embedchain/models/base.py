from typing import Optional,Any,Union

class BaseModel:
    """
    数据库表结构
    {
	"database":"aiagent",
	"table_name": "用户信息表",
	"description": "存储用户信息",
	"primary_key":"user_id",
	"table_schema": {
		"user_id": {
			"field_name": "用户表主键",
			"field_type": "integer",
			"description": "用户表主键，唯一键，关联查询时经常使用到"
		},
		"user_name": {
			"field_name": "用户姓名",
			"field_type": "string",
			"description": "公司用户的真实姓名，重名的情况下会加后缀"
		},
        "status": {
			"field_name": "用户状态",
			"field_type": "string",
			"description": "",
            "enum_translate":"0:离职,1:在职"
		}
	},
	"format_type": "json /  concat",
	"ignore_fields": "username, mobile_phone"
    }   
    """

    ALLOWED_SCHEMA_FIELDS = {'field_name', 'field_type', 'description'}
    def __init__(
        self, database: str, 
        table_name: str, 
        description: str = None, 
        primary_key: str = 'id', 
        table_schema: Optional[dict[str,dict[str,str]]] = None, 
        format_type: str = 'json', 
        ignore_fields: str = None):
        
        self.database = database
        self.table_name = table_name
        self.description = description
        self.primary_key = primary_key
        self._validate_table_schema(table_schema)
        self.table_schema = table_schema
        self._validate_format_type(format_type)
        self.format_type = format_type
        self.ignore_fields = ignore_fields

    
    def __repr__(self):
        return f"BaseModel(database={self.database!r}, table_name={self.table_name!r}, " \
               f"description={self.description!r}, primary_key={self.primary_key!r}, " \
               f"table_schema={self.table_schema!r}, format_type={self.format_type!r}, " \
               f"ignore_fields={self.ignore_fields!r})"
    
    def _validate_table_schema(self, table_schema: dict):
        for field_name, field_info in table_schema.items():
            if not set(field_info.keys()).issubset(self.ALLOWED_SCHEMA_FIELDS):
                raise ValueError(f"Invalid field(s) in table_schema for field {field_name}. "
                                 f"Only {', '.join(self.ALLOWED_SCHEMA_FIELDS)} are allowed.")
            
    def _validate_format_type(self, format_type: str):
        if format_type not in {'json', 'concat'}:
            raise ValueError(f"Invalid format_type {format_type}. "
                             f"Only 'json' and 'concat' are allowed.")