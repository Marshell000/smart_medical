from pathlib import Path


# 1.目录路径
ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = ROOT_DIR /"smart_medical" / "data"

KN_GRAPH_DATA_DIR = DATA_DIR / "knowledge_graph"
AN_DATA_DIR = DATA_DIR / "annotated_data"

LOG_DIR = ROOT_DIR / "log"
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"

NER__RAW_DATA = DATA_DIR / "NER"/""

# web 静态资源目录
WEB_STATIC_DIR = ROOT_DIR /"smart_medical" /  "src" /"web" / "templates"


# 2. 数据库连接


NEO4J_CONFIG = {
    "uri": "neo4j://localhost:7687",
    "auth": ("neo4j","196900gu")
}

# 5.DeepSeek_API_key
API_KEY = "sk-7166e3c858cc4e6380efe9e23bf879e6"
BASE_URL = "https://api.deepseek.com"
