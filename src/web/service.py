from typing import Dict, List, Any

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType

from configuration import config

class ChatService:
    def __init__(self):
        # LLM
        self.llm = ChatDeepSeek(model="deepseek-chat", temperature=0.4, api_key=config.API_KEY)

        # Neo4j connections
        self.graph = Neo4jGraph(
            url=config.NEO4J_CONFIG['uri'],
            username=config.NEO4J_CONFIG['user'],
            password=config.NEO4J_CONFIG['password']
        )

        # Embeddings + Vector store for hybrid retrieval
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            encode_kwargs={"normalize_embeddings": True}
        )

        # 混合检索 hybird search
        self.vector_stores = {
            'Disease': Neo4jVector.from_existing_index(
                self.embeddings,
                url=config.NEO4J_CONFIG['uri'],
                username=config.NEO4J_CONFIG['auth'][0],
                password=config.NEO4J_CONFIG['auth'][1],
                embedding=self.embeddings,
                index_name='disease_name_vector_index',
                keyword_index_name='disease_name_fulltext_index',
                search_type=SearchType.HYBRID
            ),
            'Department': Neo4jVector.from_existing_index(
                self.embeddings,
                url=config.NEO4J_CONFIG['uri'],
                username=config.NEO4J_CONFIG['auth'][0],
                password=config.NEO4J_CONFIG['auth'][1],
                embedding=self.embeddings,
                keyword_index_name='department_fulltext_index',
                index_name='department_vector_index',
                search_type=SearchType.HYBRID
            ),
            'Symptom': Neo4jVector.from_existing_index(
                self.embeddings,
                url=config.NEO4J_CONFIG['uri'],
                username=config.NEO4J_CONFIG['auth'][0],
                password=config.NEO4J_CONFIG['auth'][1],
                embedding=self.embeddings,
                keyword_index_name='Symptom_fulltext_index',
                index_name='Symptom_vector_index',
                search_type=SearchType.HYBRID
            ),
            'Cause': Neo4jVector.from_existing_index(
                self.embeddings,
                url=config.NEO4J_CONFIG['uri'],
                username=config.NEO4J_CONFIG['auth'][0],
                password=config.NEO4J_CONFIG['auth'][1],
                embedding=self.embeddings,
                keyword_index_name='cause_fulltext_index',
                index_name='cause_desc_vector_index',
                search_type=SearchType.HYBRID
            ),
            'Drug': Neo4jVector.from_existing_index(
                self.embeddings,
                url=config.NEO4J_CONFIG['uri'],
                username=config.NEO4J_CONFIG['auth'][0],
                password=config.NEO4J_CONFIG['auth'][1],
                embedding=self.embeddings,
                keyword_index_name='drug_fulltext_index',
                index_name='drug_vector_index',
                search_type=SearchType.HYBRID
            ),
            'Food': Neo4jVector.from_existing_index(
                self.embeddings,
                url=config.NEO4J_CONFIG['uri'],
                username=config.NEO4J_CONFIG['auth'][0],
                password=config.NEO4J_CONFIG['auth'][1],
                embedding=self.embeddings,
                keyword_index_name='food_fulltext_index',
                index_name='food_vector_index',
                search_type=SearchType.HYBRID
            ),
            'Way': Neo4jVector.from_existing_index(
                self.embeddings,
                url=config.NEO4J_CONFIG['uri'],
                username=config.NEO4J_CONFIG['auth'][0],
                password=config.NEO4J_CONFIG['auth'][1],
                embedding=self.embeddings,
                keyword_index_name='way_fulltext_index',
                index_name='way_vector_index',
                search_type=SearchType.HYBRID
            ),
            'Prevent': Neo4jVector.from_existing_index(
                self.embeddings,
                url=config.NEO4J_CONFIG['uri'],
                username=config.NEO4J_CONFIG['auth'][0],
                password=config.NEO4J_CONFIG['auth'][1],
                embedding=self.embeddings,
                keyword_index_name='prevent_fulltext_index',
                index_name='prevent_vector_index',
                search_type=SearchType.HYBRID
            ),
            'Check': Neo4jVector.from_existing_index(
                self.embeddings,
                url=config.NEO4J_CONFIG['uri'],
                username=config.NEO4J_CONFIG['auth'][0],
                password=config.NEO4J_CONFIG['auth'][1],
                embedding=self.embeddings,
                keyword_index_name='check_fulltext_index',
                index_name='check_vector_index',
                search_type=SearchType.HYBRID
            ),
            'Treat': Neo4jVector.from_existing_index(
                self.embeddings,
                url=config.NEO4J_CONFIG['uri'],
                username=config.NEO4J_CONFIG['auth'][0],
                password=config.NEO4J_CONFIG['auth'][1],
                embedding=self.embeddings,
                keyword_index_name='treat_fulltext_index',
                index_name='treat_vector_index',
                search_type=SearchType.HYBRID
            ),
            'People': Neo4jVector.from_existing_index(
                self.embeddings,
                url=config.NEO4J_CONFIG['uri'],
                username=config.NEO4J_CONFIG['auth'][0],
                password=config.NEO4J_CONFIG['auth'][1],
                embedding=self.embeddings,
                keyword_index_name='people_fulltext_index',
                index_name='people_vector_index',
                search_type=SearchType.HYBRID
            ),
            'Duration': Neo4jVector.from_existing_index(
                self.embeddings,
                url=config.NEO4J_CONFIG['uri'],
                username=config.NEO4J_CONFIG['auth'][0],
                password=config.NEO4J_CONFIG['auth'][1],
                embedding=self.embeddings,
                keyword_index_name='duration_fulltext_index',
                index_name='duration_vector_index',
                search_type=SearchType.HYBRID
            )
        }

        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()

    def _generate_cypher(self, question: str, schema_info: str):
        generate_cypher_prompt = PromptTemplate(
            input_variables=["question", "schema_info"],
            template="""
                你是一个专业的Neo4j Cypher查询生成器。你的任务是根据用户问题生成一条Cypher查询语句，用于从知识图谱中获取回答用户问题所需的信息。

                用户问题：{question}

                知识图谱结构信息：{schema_info}

                要求：
                1. 生成参数化Cypher查询语句，用param_0, param_1等代替具体值
                2. 识别需要对齐的实体
                3. 必须严格使用以下JSON格式输出结果
                {{
                  "cypher_query": "生成的Cypher语句",
                  "entities_to_align": [
                    {{
                      "param_name": "param_0",
                      "entity": "原始实体名称",
                      "label": "节点类型"
                    }}
                  ]
                }}"""
        ).format(schema_info=schema_info, question=question)
        cypher = self.llm.invoke(generate_cypher_prompt)
        cypher = self.json_parser.invoke(cypher)
        return cypher

    def _entity_align(self, entities_to_align: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """使用向量+关键词检索修正实体名称"""
        for node in entities_to_align:
            if node['label'] in self.vector_stores:
                results = self.vector_stores[node['label']].similarity_search(node['entity'], k=1)
                if results:
                    node['entity'] = results[0].page_content
        return entities_to_align

    def _execute_cypher(self, cypher: str, prams: Dict[str, str]) -> List[Dict[str, Any]]:
        """执行 Cypher 查询并返回结果"""
        results = self.graph.query(cypher, params=prams)
        return results

    def _generate_final_answer(self, question: str, query_result: List[Dict[str, Any]]) -> str:
        """
        将 Cypher 查询结果生成自然语言答案
        """
        prompt = PromptTemplate(
            input_variables=["question", "query_result"],
            template="""
                你是一个电商智能客服，根据用户问题，以及数据库查询结果生成一段简洁、准确的自然语言回答。
                用户问题: {question}
                数据库返回结果: {query_result}
            """).format(question=question, query_result=query_result)
        result = self.llm.invoke(prompt)
        return self.str_parser.invoke(result)
    # 根据用户问题抽取实体，生成cypher
    def chat(self, question: str):
        cypher = self._generate_cypher(question, self.graph.schema)
        cypher_query = cypher['cypher_query']
        entities_to_align = cypher['entities_to_align']
        entities = self._entity_align(entities_to_align)
        params = {entity['param_name']: entity['entity'] for entity in entities}
        print(cypher_query)
        print(params)
        query_result = self._execute_cypher(cypher_query, params)
        return self._generate_final_answer(question, query_result)
