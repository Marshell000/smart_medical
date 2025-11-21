import pymysql
from neo4j import GraphDatabase
from pymysql.cursors import DictCursor
from configuration.config import *


# 写入Neo4j
class Neo4jWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(**NEO4J_CONFIG)

    def write_node(self, properties: list[dict], label: str, unique_key: str):
        cypher = f"""
            UNWIND $batch AS item
            MERGE (n:{label} {{{unique_key}: item.{unique_key}}})
            ON CREATE SET n = item
            ON MATCH SET n = item
        """
        result = self.driver.execute_query(cypher, batch=properties)
        summary = result.summary
        print(f"成功创建 {summary.counters.nodes_created} 个节点")

    def write_relations(self,label_type:str, start_label,end_label,relations:list[dict]):
        cypher = f"""
            UNWIND $batch AS item
            MATCH (start:{start_label}{{id:item.start_id}}),(end:{end_label}{{id:item.end_id}})
            MERGE (start)-[:{label_type}]->(end)
        """
        result = self.driver.execute_query(cypher, batch=relations)
        summary = result.summary
        print(f"成功创建 {summary.counters.relationships_created} 条关系")


if __name__ == '__main__':
    writer = Neo4jWriter()




