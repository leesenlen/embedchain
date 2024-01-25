import hashlib
import logging
from typing import Any, Optional
import json

from embedchain.loaders.base_loader import BaseLoader
from embedchain.utils.misc import clean_string
from embedchain.models.base import BaseModel


class MySQLLoader(BaseLoader):
    def __init__(self, config: Optional[dict[str, Any]]):
        super().__init__()
        if not config:
            raise ValueError(
                f"Invalid sql config: {config}.",
                "Provide the correct config, refer `https://docs.embedchain.ai/data-sources/mysql`.",
            )

        self.config = config
        self.connection = None
        self.cursor = None
        self._setup_loader(config=config)

    def _setup_loader(self, config: dict[str, Any]):
        try:
            import mysql.connector as sqlconnector
        except ImportError as e:
            raise ImportError(
                "Unable to import required packages for MySQL loader. Run `pip install --upgrade 'embedchain[mysql]'`."  # noqa: E501
            ) from e

        try:
            self.connection = sqlconnector.connection.MySQLConnection(**config)
            self.cursor = self.connection.cursor(dictionary=True)
        except (sqlconnector.Error, IOError) as err:
            logging.info(f"Connection failed: {err}")
            raise ValueError(
                f"Unable to connect with the given config: {config}.",
                "Please provide the correct configuration to load data from you MySQL DB. \
                    Refer `https://docs.embedchain.ai/data-sources/mysql`.",
            )


    def load_data(self, table: BaseModel):
        
        return self.fetch_all_id_range(table)
  
        

    # def fetch_all_paginated(self, database, table_name, page_size=50):
    #     """
    #     普通分页方式导入数据
    #     :param database: 数据库名
    #     :param table_name: 表名
    #     :param page_size: 每页数据量
    #     :return:
    #     """
    #     page_num = 1
    #     has_more_data = True
    #     while has_more_data:
    #         offset = (page_num - 1) * page_size
    #         query = f"SELECT * FROM {database}.{table_name} LIMIT {page_size} OFFSET {offset}"
    #         self.cursor.execute(query)
    #         result = self.cursor.fetchall()

    #         # 如果返回结果为空，则没有更多数据
    #         if not result:
    #             has_more_data = False
    #         else:
    #             # 处理当前页的数据（示例）
    #             for row in result:
    #                 print(row)
    #             page_num += 1

    #     self.cursor.close()
    #     self.connection.close()

    def fetch_all_id_range(self, table: BaseModel, id_range=1000):
        """
        默认以primary_key进行分块导入
        """
        columns_string = '*'
        
        if table.table_schema is not None:
            column_info = []
            for column,value in table.table_schema.items():
                column_info.append(column)
            columns_string = ','.join(column_info)
   
        query = f"SELECT {table.primary_key} FROM {table.database}.{table.table_name} order by {table.primary_key} desc LIMIT 1"
        self.cursor.execute(query)
        result = self.cursor.fetchone()

        # 获取最大 ID
        max_id = result[table.primary_key] if result else 0
        data = []
        start_id = 1
        end_id = start_id + id_range -1
        while start_id <= max_id:
            query = f"SELECT {columns_string} FROM {table.database}.{table.table_name} where {table.primary_key} between {start_id} and {end_id}"
            self.cursor.execute(query)
            result = self.cursor.fetchall()

            if result:
                for row in result:
                    doc_content = clean_string(self.parse_row(row,table))
                    data.append({"primary_key": row[table.primary_key],"content": doc_content, "meta_data": {"url": query,"fields":row}})
        
            start_id = end_id + 1
            end_id = start_id + id_range -1

        self.cursor.close()
        self.connection.close()
        return data
        
        

    def parse_row(self,row:dict,table: BaseModel):        
        content = {}
        if table.table_schema is not None:
            for column,value in table.table_schema.items():
                if table.ignore_fields is not None and column in table.ignore_fields:
                    continue
                if value is not None and "field_name" in value:
                    content[value['field_name']] = row[column]
        else:
            content = row
        match table.format_type:
            case "json":
                return str(content)
            case "concat":
                return ",".join([f"{key}:{value}" for key, value in content.items()])
            case _:
                raise ValueError(f"Invalid format type: {table.format_type}")   



        
        




