"""
Project Icarus - 向量数据库上膛脚本 v2
====================================
将 icarus_cleansed_db.json 清洗数据写入本地 ChromaDB 向量数据库。
Embedding 引擎: Ollama nomic-embed-text (本地推理，数据不出站)

# NOTE: 架构预留 - RAG 检索生成节点参数红线
# -------------------------------------------------------
# 下一步 RAG 检索生成节点调用本地 qwen2.5:14b，通过 Ollama API。
# 锁死参数:
#   ctx_size    = 4096   (上下文窗口，禁止擅自放大)
#   top_k       = 3-5    (检索返回条数，3条为默认，最大不超过5)
#   调用模式    = 串行   (先检索后生成，严禁并行调用)
#   Ollama URL  = http://localhost:11434/api/generate
#   Model       = qwen2.5:14b
# -------------------------------------------------------
"""

import json
import logging
import sys

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from tqdm import tqdm

# ─── 日志配置 ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── 常量定义 ───────────────────────────────────────────
DATA_PATH = "./icarus_cleansed_db_v3.json"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "icarus_failures_final"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL_NAME = "bge-m3"


def buildDocument(record: dict) -> str:
    """
    构建向量化主文本。
    拼接格式严格为:
      "行业: {industry}。失败原因: {failure_reasons}。认知偏差原型: {archetype}。"

    NOTE: company_name 绝对禁止出现在向量化文本中。
    v3 数据结构中 industry / failure_reasons / archetype 均为顶层直接字段。
    """
    return (
        f"行业: {record['industry']}。"
        f"失败原因: {record['failure_reasons']}。"
        f"认知偏差原型: {record['archetype']}。"
    )


def buildMetadata(record: dict) -> dict:
    """
    构建元数据，存入 company_name、funding_amount、archetype、industry。
    """
    return {
        "company_name": record["company_name"],
        "funding_amount": record["funding_amount"],
        "archetype": record["archetype"],
        "industry": record["industry"],
    }


def main() -> None:
    # ─── 1. 加载清洗数据 ────────────────────────────────
    logger.info("正在加载清洗数据: %s", DATA_PATH)
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            records: list[dict] = json.load(f)
    except FileNotFoundError:
        logger.error("数据文件不存在: %s", DATA_PATH)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error("JSON 解析失败: %s", e)
        sys.exit(1)

    logger.info("成功加载 %d 条记录", len(records))

    # ─── 2. 初始化 ChromaDB (清理并重建) ────────────────
    logger.info("初始化 ChromaDB PersistentClient: %s", CHROMA_PERSIST_DIR)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # 删除已有同名 Collection，确保数据干净
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        logger.info("检测到已有 Collection '%s'，正在删除...", COLLECTION_NAME)
        client.delete_collection(name=COLLECTION_NAME)
        logger.info("已删除旧 Collection")

    # ─── 3. 配置本地 Embedding 引擎 ────────────────────
    logger.info(
        "配置 Ollama Embedding 引擎: model=%s, url=%s",
        EMBED_MODEL_NAME,
        OLLAMA_EMBED_URL,
    )
    embedding_fn = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL_NAME,
        url=OLLAMA_EMBED_URL,
    )

    # 创建新 Collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    logger.info("已创建新 Collection: '%s'", COLLECTION_NAME)

    # ─── 4. 数据拆解与写入 ──────────────────────────────
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for index, record in enumerate(
        tqdm(records, desc="[INGEST] Building vectors", unit="rec", ncols=80)
    ):
        doc = buildDocument(record)
        meta = buildMetadata(record)
        record_id = f"icarus_{index}"

        documents.append(doc)
        metadatas.append(meta)
        ids.append(record_id)

    # 批量写入 ChromaDB
    logger.info("正在将 %d 条数据写入 ChromaDB...", len(documents))
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    # ─── 5. 工程化终端展示 ──────────────────────────────
    final_count = collection.count()
    logger.info("=" * 60)
    logger.info("[OK] 向量数据库上膛完毕")
    logger.info("   Collection 名称 : %s", COLLECTION_NAME)
    logger.info("   写入总条数       : %d", final_count)
    logger.info("   持久化路径       : %s", CHROMA_PERSIST_DIR)
    logger.info("   Embedding 模型   : %s", EMBED_MODEL_NAME)
    logger.info("=" * 60)

    # 数据完整性校验
    if final_count != len(records):
        logger.warning(
            "[WARN] 数据条数不一致! 源数据 %d 条, 写入 %d 条",
            len(records),
            final_count,
        )
    else:
        logger.info("[OK] 数据完整性校验通过: %d / %d", final_count, len(records))


if __name__ == "__main__":
    main()
