from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType

from configuration.config import *

class IndexUntil:
    def __init__(self):

        self.graph = Neo4jGraph(
            url = NEO4J_CONFIG['url'],
            username = NEO4J_CONFIG['auth'][0],
            password = NEO4J_CONFIG['auth'][1]

        )
        #
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=" BAAI/bge-base-zh-v1.5",
            encode_kwage = {"normalize_embeddings": True},
        )

    # 创建全文索引,传入索引名称，节点标签，属性
    def create_fulltext_index(self,index_name,label,property):
        cypher = f"""CREATE FULLTEXT INDEX {index_name} IF NOY EXISTS
         FOR (n:{label}) ON EACH [n.{property}]
        """

        self.graph.query(cypher)

    # 创建向量索引
    def create_vector_index(self,index_name,label,sourcec_property,embedding_property):
        # 生成嵌入向量，并添加到节点属性中
        embedding_dim = self._add_embedding(label,sourcec_property,embedding_property)

        # 生成向量索引
        cypher = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (m:{label})
        ON m.{embedding_property}
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {embedding_dim},
            `vector.similarity_function`: 'cosine'
        }}}}"""

        self.graph.query(cypher)

    # 生成嵌入向量，并添加到节点属性中
    def _add_embedding(self,label,sourcec_property,embedding_property):
        # 1.查询所有节点对应的属性值，作为模型的输入，还需要查出节点ID
        cypher = f"""
            MATCH (n:{label})
            RETURN n.{sourcec_property} AS text, id(n) as id
        """
        results = self.graph.query(cypher)
        # 2.获取查询结果中的文本内容和ID
        docs = [result['text'] for result in results]

        # 3.调用嵌入模型，输入给模型,生成嵌入向量
        embeddings = self.embedding_model.embed_documents(docs)

        # 4.将ID和嵌入向量组合成字典形式
        batch = []
        for id, embedding in zip(results, embeddings):
            item = {"id":results["id"],"embedding":embedding}
            batch.append(item)

        # 5.执行cypher,按照ID查询节点，写入新的向量属性
        cypher = f"""
            UNWIND $batch AS item
            MATCH (n:{label})
            WHERE id(n) = item.id
            SET n.{embedding_property} = item.embedding 
        """
        self.graph.query(cypher,params={'batch':batch})

        return len(embeddings[0])

if __name__ == "__main__":
    index = IndexUntil()
    #
    # index_name = "vector"  # vector - 向量检索的索引名称
    # keyword_index_name = "keyword"  # keywords - 全文检索的索引名称
    #
    # store = Neo4jVector.from_existing_index(
    #     index.embedding_model,
    #     url=NEO4J_CONFIG['url'],
    #     username=NEO4J_CONFIG['auth'][0],
    #     password=NEO4J_CONFIG['auth'][1],
    #     index_name=index_name,
    #     keyword_index_name=keyword_index_name,
    #     search_type=SearchType.HYBRID,
    # )
    #
    #

    index.create_fulltext_index("disease_name_fulltext_index","Disease","name")
    index.create_vector_index("disease_name_vector_index","Disease","name","embedding")

    index.create_fulltext_index("department_fulltext_index","Department","name")
    index.create_vector_index("department_vector_index","Department","name","embedding")

    index.create_fulltext_index("Symptom_fulltext_index","Symptom","name")
    index.create_vector_index("Symptom_vector_index","Symptom","name","embedding")

    index.create_fulltext_index("cause_fulltext_index","Cause","decs")
    index.create_vector_index("cause_desc_vector_index","Cause","desc","embedding")

    index.create_fulltext_index("drug_fulltext_index","Drug","name")
    index.create_vector_index("drug_vector_index","Drug","name","embedding")

    index.create_fulltext_index("food_fulltext_index","Food","name")
    index.create_vector_index("food_vector_index","Food","name","embedding")

    index.create_fulltext_index("way_fulltext_index","Way","name")
    index.create_vector_index("way_vector_index","Way","name","embedding")

    index.create_fulltext_index("prevent_fulltext_index","Prevent","desc")
    index.create_vector_index("prevent_vector_index","Prevent","desc","embedding")

    index.create_fulltext_index("check_fulltext_index","Check","name")
    index.create_vector_index("check_vector_index","Check","name","embedding")

    index.create_fulltext_index("treat_fulltext_index","Treat","name")
    index.create_vector_index("treat_vector_index","Treat","name","embedding")

    index.create_fulltext_index("people_fulltext_index","People","name")
    index.create_vector_index("people_vector_index","People","name","embedding")

    index.create_fulltext_index("duration_fulltext_index","Duration","name")
    index.create_vector_index("duration_vector_index","Duration","name","embedding")
