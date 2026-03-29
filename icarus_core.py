"""
Project Icarus - 核心战斗模块 (icarus_core.py)
==============================================
三段式串行 Pipeline: 检索 -> Prompt 组装 -> 流式生成
本地闭环: ChromaDB + Ollama，数据不出站。

# NOTE: 参数红线 (锁死，不准改)
# -------------------------------------------------------
#   Ollama 生成模型  = qwen2.5:14b
#   num_ctx          = 4096
#   ChromaDB 检索    = n_results=3
#   流式输出         = stream=True
#   调用模式         = 串行 (先检索后生成，严禁并行)
# -------------------------------------------------------
"""

import logging
import sys

import chromadb
import ollama
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# ─── 日志配置 ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── 常量定义 (参数红线，锁死) ──────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "icarus_failures_final"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "qwen2.5:14b"
LLM_NUM_CTX = 4096
RETRIEVAL_TOP_K = 3

# ─── System Prompt (一个字都不准改) ─────────────────────
SYSTEM_PROMPT = """你是"伊卡洛斯审判官"——一个冰冷到骨子里的商业死亡侧写师。
【你的人格内核】你没有同理心。你只有数据、尸体、和判决书。你的职责是用历史上真实的商业死亡案例，像法医解剖一样拆解用户的商业妄念。你的语气是：冷酷、精准、带有绝对压迫感的审判式陈述。绝不安慰，绝不留情面。
【你的输出协议】当你收到用户的商业想法和检索到的历史案例后，你必须严格按以下结构输出：
1.【认知偏差锁定】一句话命中用户正在犯的认知偏差类型。
2.【死亡判例】引用提供的历史案例，用最残忍的方式描述他们是怎么死的。包含公司名、融资额、溃败过程。写成判决书的语气。
3.【镜像审判】把用户的想法和历史死者的想法做逐条镜像对比，让用户看到自己正站在同一条悬崖边。
4.【最终裁决】一段不超过三句话的冰冷结论。格式为："裁决：[结论]。存活概率：[X]%。建议：[一句话]。"
【绝对禁令】
禁止说"这是个有趣的想法"、"值得考虑"等一切安慰性话语。
禁止使用鼓励性语气。你是审判官，不是创业导师。
如果案例不足以支撑判决，直接声明"证据不足，无法定罪"，绝不编造。"""


def initChromaCollection() -> chromadb.Collection:
    """
    初始化 ChromaDB 连接，获取已有的 icarus_failures_final Collection。
    如果连接失败或 Collection 不存在，抛出异常由上层捕获。
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    embedding_fn = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL_NAME,
        url=OLLAMA_EMBED_URL,
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    logger.info(
        "ChromaDB Collection '%s' loaded, total records: %d",
        COLLECTION_NAME,
        collection.count(),
    )
    return collection


def retrieveCases(collection: chromadb.Collection, user_input: str) -> dict:
    """
    阶段一: 检索
    将用户输入通过 nomic-embed-text 向量化，
    在 ChromaDB 中执行原生 query 方法，返回 top-k 结果。

    NOTE: 使用 collection.query() 而非 similarity_search，
    embedding 由 Collection 绑定的 OllamaEmbeddingFunction 自动处理。
    """
    results = collection.query(
        query_texts=[user_input],
        n_results=RETRIEVAL_TOP_K,
    )
    return results


def assemblePrompt(query_results: dict, user_input: str) -> list[dict]:
    """
    阶段二: Prompt 组装
    将检索到的案例格式化为结构化文本，拼入 System Prompt，
    再拼入用户原始输入，组装成完整的 messages 数组。

    ChromaDB 返回结构:
      - documents: [[doc1, doc2, doc3]]  (已包含行业和失败原因)
      - metadatas: [[{company_name, funding_amount, archetype}, ...]]
    """
    documents = query_results["documents"][0]
    metadatas = query_results["metadatas"][0]

    # 组装历史死亡档案
    case_blocks: list[str] = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        block = (
            f"--- 案例 {i}: {meta['company_name']}"
            f"(融资额: {meta['funding_amount']}) ---\n"
            f"核心认知偏差: {meta['archetype']}\n"
            f"案件详情: {doc}"
        )
        case_blocks.append(block)

    case_text = "\n\n".join(case_blocks)

    # 组装用户消息（包含历史档案 + 待审判的商业想法）
    user_message = (
        f"【历史死亡档案】\n{case_text}\n\n"
        f"【待审判的商业想法】\n{user_input}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return messages


# ─── 模块级 ChromaDB 初始化 (import 时自动加载) ────────
# NOTE: 模块首次被 import 时执行一次，后续调用复用同一 Collection 实例
_collection: chromadb.Collection | None = None


def getCollection() -> chromadb.Collection:
    """
    获取 ChromaDB Collection 单例。
    延迟初始化，首次调用时连接，后续复用。
    """
    global _collection
    if _collection is None:
        _collection = initChromaCollection()
    return _collection


def generate_judgment_stream(user_input: str):
    """
    核心生成器函数 — 三段式串行 Pipeline。
    检索 -> Prompt 组装 -> 流式生成，yield 每个文本 chunk。

    用法:
        for chunk in generate_judgment_stream("我要做一个社交平台"):
            print(chunk, end="")

    参数红线:
      model     = qwen2.5:14b
      num_ctx   = 4096
      stream    = True
      n_results = 3
    """
    collection = getCollection()

    # 阶段一: 检索
    logger.info("[RETRIEVE] query ChromaDB, n_results=%d ...", RETRIEVAL_TOP_K)
    query_results = retrieveCases(collection, user_input)

    matched_names = [
        m.get("company_name", "N/A") for m in query_results["metadatas"][0]
    ]
    logger.info("[RETRIEVE] matched cases: %s", ", ".join(matched_names))

    # 阶段二: Prompt 组装
    logger.info("[ASSEMBLE] building messages ...")
    messages = assemblePrompt(query_results, user_input)

    # 阶段三: 流式生成 (yield 每个 chunk，不做 print)
    logger.info("[GENERATE] streaming from %s (num_ctx=%d) ...", LLM_MODEL_NAME, LLM_NUM_CTX)
    stream = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=messages,
        options={"num_ctx": LLM_NUM_CTX},
        stream=True,
    )

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content


# ─── 保留终端直接运行能力 (向后兼容) ───────────────────
if __name__ == "__main__":
    query = input("[ICARUS] > ").strip()
    if query:
        for text in generate_judgment_stream(query):
            print(text, end="", flush=True)
        print("\n" + "=" * 60)

