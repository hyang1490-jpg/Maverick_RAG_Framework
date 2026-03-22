"""
Project Icarus 核心引擎 (v2)
适配新 JSON 架构：archetype / core_cognitive_bias / fatal_action / trigger_condition
将 grok_data.json 中的人物档案向量化并持久化到 ChromaDB。
"""

import json
import logging
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer

# NOTE: 日志配置 —— 统一使用 logging，禁止 print
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("icarus_core")

# ─── 常量 ───────────────────────────────────────────────
DATA_FILE = Path(__file__).parent / "grok_data.json"
DB_PATH = "./maverick_icarus_db"
COLLECTION_NAME = "fatal_decisions_archive"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"


def detectDevice() -> str:
    """
    安全检测 CUDA 可用性。
    RTX 5080 (Blackwell) 与某些 CUDA toolkit 版本存在兼容性问题，
    因此需要实际创建 tensor 进行端到端验证。
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _test = torch.zeros(1, device="cuda")
        del _test
        return "cuda"
    except Exception as e:
        logger.warning("CUDA 可用性检测失败 (%s)，回退至 CPU 模式", e)
        return "cpu"


DEVICE = detectDevice()


def buildDocument(record: dict) -> str:
    """
    将单条人物档案的关键字段拼接为一段连续长文本，
    供 SentenceTransformer 生成语义向量。

    新架构拼接逻辑：core_cognitive_bias + fatal_action + trigger_condition
    这三个字段涵盖了认知偏差、致命行为和触发条件，
    是语义检索匹配危险思维模式的核心特征。
    """
    return (
        f"认知偏差：{record['core_cognitive_bias']}。"
        f"致命行为：{record['fatal_action']}。"
        f"触发条件：{record['trigger_condition']}。"
    )


def main() -> None:
    # ── 1. 读取数据弹药 ──────────────────────────────────
    logger.info("正在读取数据文件: %s", DATA_FILE)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        records: list[dict] = json.load(f)
    logger.info("成功加载 %d 条人物档案", len(records))

    # ── 2. 初始化本地向量数据库 ──────────────────────────
    logger.info("初始化 ChromaDB 持久化客户端 → %s", DB_PATH)
    client = chromadb.PersistentClient(path=DB_PATH)
    # NOTE: 先删除旧 collection 再重建，确保新架构字段完全一致
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info("已清除旧 Collection [%s]", COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)
    logger.info("Collection [%s] 已重建", COLLECTION_NAME)

    # ── 3. 加载降维引擎 ──────────────────────────────────
    logger.info("加载 SentenceTransformer 模型: %s (device=%s)", EMBEDDING_MODEL, DEVICE)
    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    if DEVICE == "cuda":
        logger.info("模型加载完成，CUDA 加速已启用 🚀")
    else:
        logger.info("模型加载完成，当前运行于 CPU 模式")

    # ── 4. 构建 Document / Embedding / Metadata 并批量入库
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for record in records:
        doc = buildDocument(record)
        ids.append(record["target_id"])
        documents.append(doc)
        # NOTE: metadata 只存短/中等长度字段，长文本由 radar 从 JSON 直接查表
        metadatas.append({
            "target_id": record["target_id"],
            "archetype": record["archetype"],
            "ultimate_outcome": record["ultimate_outcome"],
        })

    # NOTE: 先统一编码再 upsert，避免逐条调用的性能损耗
    logger.info("正在为 %d 条文档生成特征向量...", len(documents))
    embeddings = model.encode(documents, show_progress_bar=True, normalize_embeddings=True)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    logger.info("✅ 全部 %d 条人物档案已成功 upsert 进 ChromaDB [%s]", len(ids), COLLECTION_NAME)
    logger.info("向量数据库持久化路径: %s", DB_PATH)
    logger.info("🔥 Project Icarus 数据管道点火完成")


if __name__ == "__main__":
    main()
