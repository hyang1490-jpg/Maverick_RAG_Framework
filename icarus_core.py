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

import json
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
EMBED_MODEL_NAME = "bge-m3"
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


_VALID_ARCHETYPES = (
    "Cargo Cult, Dunning-Kruger, Echo Chamber, God Complex, "
    "Groupthink, Icarus Syndrome, Sunk Cost Fallacy, Survivorship Bias"
)
_VALID_INDUSTRIES = (
    "二手家具交易平台, 二手服装交易平台, 云计算基础设施, 互联网浏览器, 企业SaaS, "
    "共享居住, 出行服务, 在线住宿预订, 在线拍卖, 在线教育, 在线旅游, 在线视频, "
    "在线零售, 增强现实, 建筑科技, 旅游服务, 智能硬件, 汽车交易市场平台, 汽车租赁, "
    "玩具零售, 电子商务, 直播平台, 社交书签管理, 社交网络, 社交网络营销工具, "
    "移动应用开发, 金融科技, 音乐共享平台, 食品科技"
)


def classifyInput(user_input: str) -> dict | None:
    """
    轻量分类：调用 LLM 从用户输入中提取 industry 和 archetype。
    industry 和 archetype 均限定为库中实际存在的枚举值。
    max_tokens=50，只输出 JSON，解析失败或调用异常时返回 None。
    """
    prompt = (
        f'请从以下输入中判断最可能的行业标签和认知偏差类型。\n'
        f'industry 必须从以下列表中选一个：{_VALID_INDUSTRIES}\n'
        f'archetype 必须从以下列表中选一个：{_VALID_ARCHETYPES}\n'
        f'只输出JSON格式 {{"industry": "...", "archetype": "..."}}，不要输出其他内容。\n'
        f'输入：{user_input}'
    )
    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 50},
        )
        text = response["message"]["content"].strip()
        return json.loads(text)
    except Exception as e:
        logger.warning("[CLASSIFY] failed: %s", e)
        return None


def _queryWithFilter(
    collection: chromadb.Collection, user_input: str, where: dict
) -> dict | None:
    """
    带 where 过滤的单路检索，取 top-1。
    若过滤后无匹配文档或查询出错，返回 None。
    """
    try:
        return collection.query(
            query_texts=[user_input],
            n_results=1,
            where=where,
        )
    except Exception as e:
        logger.warning("[RETRIEVE] filtered query failed (where=%s): %s", where, e)
        return None


def retrieveCases(collection: chromadb.Collection, user_input: str) -> dict:
    """
    阶段一: 三路并行召回 + 合并去重
    - 先用 LLM 轻量分类，提取 industry / archetype
    - 路线一: where industry 过滤 + 语义检索 top-1
    - 路线二: where archetype 过滤 + 语义检索 top-1
    - 路线三: 全局语义检索 top-1（保底）
    三路结果按 company_name 去重后按 distance 升序排列。
    分类失败则 fallback 到原始全局 top-3。

    NOTE: 使用 collection.query() 而非 similarity_search，
    embedding 由 Collection 绑定的 OllamaEmbeddingFunction 自动处理。
    """
    # 轻量分类
    classification = classifyInput(user_input)

    if classification is None:
        logger.info("[RETRIEVE] classification failed, fallback to global top-%d", RETRIEVAL_TOP_K)
        return collection.query(query_texts=[user_input], n_results=RETRIEVAL_TOP_K)

    logger.info(
        "[CLASSIFY] industry=%s, archetype=%s",
        classification.get("industry"),
        classification.get("archetype"),
    )

    # 三路检索
    route_results = []

    r1 = _queryWithFilter(
        collection, user_input,
        {"industry": {"$eq": classification.get("industry", "")}},
    )
    if r1:
        route_results.append(r1)

    r2 = _queryWithFilter(
        collection, user_input,
        {"archetype": {"$eq": classification.get("archetype", "")}},
    )
    if r2:
        route_results.append(r2)

    r3 = collection.query(query_texts=[user_input], n_results=1)
    route_results.append(r3)

    # 合并去重（按 company_name），保留所有唯一案例
    seen: set[str] = set()
    merged_docs: list[str] = []
    merged_metas: list[dict] = []
    merged_distances: list[float] = []

    for r in route_results:
        if not r or not r.get("documents") or not r["documents"][0]:
            continue
        for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
            name = meta.get("company_name", "")
            if name not in seen:
                seen.add(name)
                merged_docs.append(doc)
                merged_metas.append(meta)
                merged_distances.append(dist)

    if not merged_docs:
        logger.warning("[RETRIEVE] all routes empty, fallback to global top-%d", RETRIEVAL_TOP_K)
        return collection.query(query_texts=[user_input], n_results=RETRIEVAL_TOP_K)

    # 按 distance 升序排列，最相关的排最前
    sorted_triples = sorted(
        zip(merged_distances, merged_docs, merged_metas), key=lambda x: x[0]
    )
    s_distances, s_docs, s_metas = zip(*sorted_triples)

    return {
        "documents": [list(s_docs)],
        "metadatas": [list(s_metas)],
        "distances": [list(s_distances)],
    }


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

    # 检索置信度门控：top-1 distance > 0.65 时知识库覆盖不足，跳过生成阶段
    top1_distance = query_results["distances"][0][0]
    logger.info("[RETRIEVE] top-1 distance: %.4f", top1_distance)
    if top1_distance > 0.65:
        logger.warning("[RETRIEVE] distance exceeds threshold (0.65), aborting generation.")
        yield "当前知识库无法覆盖此领域的风险分析，请换一个更具体的创业场景描述"
        return

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

