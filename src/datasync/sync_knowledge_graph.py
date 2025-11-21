import json
from neo4j import GraphDatabase
import os
from configuration.config import *

# ================== 配置区 =================
JSONL_FILE_PATH = KN_GRAPH_DATA_DIR / "medical_kg.jsonl"
# ==========================================

driver = GraphDatabase.driver(NEO4J_CONFIG["uri"], auth=(NEO4J_CONFIG["auth"][0], NEO4J_CONFIG["auth"][1]))


def create_constraints(tx):
    """创建所有唯一约束"""
    constraints = [
        "CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT department_name_unique IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT cause_desc_unique IF NOT EXISTS FOR (c:Cause) REQUIRE c.desc IS UNIQUE",
        "CREATE CONSTRAINT drug_name_unique IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT food_name_unique IF NOT EXISTS FOR (f:Food) REQUIRE f.name IS UNIQUE",
        "CREATE CONSTRAINT way_name_unique IF NOT EXISTS FOR (w:Way) REQUIRE w.name IS UNIQUE",
        "CREATE CONSTRAINT prevent_desc_unique IF NOT EXISTS FOR (p:Prevent) REQUIRE p.desc IS UNIQUE",
        "CREATE CONSTRAINT check_name_unique IF NOT EXISTS FOR (c:Check) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT treat_name_unique IF NOT EXISTS FOR (t:Treat) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT people_name_unique IF NOT EXISTS FOR (p:People) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT duration_name_unique IF NOT EXISTS FOR (d:Duration) REQUIRE d.name IS UNIQUE",
    ]
    for cypher in constraints:
        tx.run(cypher)


def import_disease_record(tx, record):
    name = record["name"]

    # 1. 创建疾病节点
    tx.run("""
        MERGE (d:Disease {name: $name})
        SET d.desc = $desc
    """, name=name, desc=record.get("desc", ""))

    # 2. 并发症 (Disease -> Disease)
    for comp in record.get("acompany", []):
        if comp.strip():
            tx.run("""
                MATCH (d:Disease {name: $disease})
                MERGE (c:Disease {name: $comp})
                MERGE (d)-[:ACOMPANY]->(c)
            """, disease=name, comp=comp)

    # 3. 所属科室 (Disease -> Department)
    for dept in record.get("department", []):
        if dept.strip():
            tx.run("""
                MATCH (d:Disease {name: $disease})
                MERGE (dept:Department {name: $dept})
                MERGE (d)-[:BELONG]->(dept)
            """, disease=name, dept=dept)

    # 4. 症状 (Disease -> Symptom)
    for sym in record.get("symptom", []):
        if sym.strip():
            tx.run("""
                MATCH (d:Disease {name: $disease})
                MERGE (s:Symptom {name: $sym})
                MERGE (d)-[:HAVE]->(s)
            """, disease=name, sym=sym)

    # 5. 诱因 (Cause -> Disease)  ← 修正：用desc
    cause = record.get("cause")
    if cause and cause.strip():
        tx.run("""
            MATCH (d:Disease {name: $disease})
            MERGE (c:Cause {desc: $cause})
            MERGE (c)-[:CAUSE]->(d)
        """, disease=name, cause=cause)

    # 6. 药物 (Disease -> Drug)
    for drug in record.get("drug", []):
        if drug.strip():
            tx.run("""
                MATCH (d:Disease {name: $disease})
                MERGE (m:Drug {name: $drug})
                MERGE (d)-[:COMMON_USE]->(m)
            """, disease=name, drug=drug)

    # 7. 宜食 (Disease -> Food)
    for food in record.get("eat", []):
        if food.strip():
            tx.run("""
                MATCH (d:Disease {name: $disease})
                MERGE (f:Food {name: $food})
                MERGE (d)-[:EAT]->(f)
            """, disease=name, food=food)

    # 8. 忌食 (Disease -> Food)
    for food in record.get("not_eat", []):
        if food.strip():
            tx.run("""
                MATCH (d:Disease {name: $disease})
                MERGE (f:Food {name: $food})
                MERGE (d)-[:NO_EAT]->(f)
            """, disease=name, food=food)

    # 9. 传播途径 (Disease -> Way)  ← 修正：用name
    way = record.get("way")
    if way and way.strip():
        tx.run("""
            MATCH (d:Disease {name: $disease})
            MERGE (w:Way {name: $way})
            MERGE (d)-[:TRANSMIT]->(w)
        """, disease=name, way=way)

    # 10. 预防措施 (Prevent -> Disease)  ← 修正：用desc
    prevent = record.get("prevent")
    if prevent and prevent.strip():
        tx.run("""
            MATCH (d:Disease {name: $disease})
            MERGE (p:Prevent {desc: $prevent})
            MERGE (p)-[:PREVENT]->(d)
        """, disease=name, prevent=prevent)

    # 11. 医学检查 (MedicalCheck -> Disease)
    for check in record.get("check", []):
        if check.strip():
            tx.run("""
                MATCH (d:Disease {name: $disease})
                MERGE (c:Check {name: $check})
                MERGE (c)-[:CHECK]->(d)
            """, disease=name, check=check)

    # 12. 治疗方式 (Treatment -> Disease)
    for treat in record.get("treat", []):
        if treat.strip():
            tx.run("""
                MATCH (d:Disease {name: $disease})
                MERGE (t:Treat {name: $treat})
                MERGE (t)-[:TREAT]->(d)
            """, disease=name, treat=treat)

    # 13. 易感人群 (Disease -> People)
    people = record.get("people")
    if people and people.strip():
        tx.run("""
            MATCH (d:Disease {name: $disease})
            MERGE (p:People {name: $people})
            MERGE (d)-[:COMMON_ON]->(p)
        """, disease=name, people=people)

    # 14. 治疗周期 (Disease -> Duration)
    duration = record.get("duration")
    if duration and duration.strip():
        tx.run("""
            MATCH (d:Disease {name: $disease})
            MERGE (dur:Duration {name: $duration})
            MERGE (d)-[:TREAT_DURATION]->(dur)
        """, disease=name, duration=duration)


def main():
    if not os.path.exists(JSONL_FILE_PATH):
        print(f"文件不存在: {JSONL_FILE_PATH}")
        return

    with driver.session() as session:
        print("正在创建唯一约束...")
        session.execute_write(create_constraints)
        print("唯一约束创建完成")

    print(f"开始导入文件{JSONL_FILE_PATH} ...")
    count = 0
    with open(JSONL_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                with driver.session() as session:
                    session.execute_write(import_disease_record, record)
                count += 1
                if count % 500 == 0:
                    print(f"  已导入 {count} 条记录...")
            except Exception as e:
                print(f"第 {count + 1} 行解析失败: {e}")
                continue

    print(f"\n最终 {count} 个疾病成功导入 Neo4j！")
    driver.close()


if __name__ == "__main__":
    main()