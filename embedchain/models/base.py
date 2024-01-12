class BaseModel:
    database: str
    table_name: str
    table_description: str = None
    primary_key: str = 'id' #唯一键，多个字段可用,分隔 
    query_columns: list[str] = None
    query_columns_description: list[str] = None

    def _to_schema(self):
        schema = {}
        columns = {}
        schema["database"] = self.database
        schema["table_name"] = self.table_name
        schema['primary_key'] = self.primary_key

        if self.table_description is not None:
            schema['table_description'] = self.table_description
        if self.query_columns is not None:
            if self.query_columns_description is None:
                raise ValueError("query_columns_description must be set when query_columns is set")
            else:
                if len(self.query_columns) != len(self.query_columns_description):
                    raise ValueError("query_columns_description must be the same length as query_columns")
                else:
                    for column in self.query_columns:
                        columns[column] = {
                            "description": self.query_columns_description[self.query_columns.index(column)]
                        }
                    schema["columns"] = columns
        
        return schema