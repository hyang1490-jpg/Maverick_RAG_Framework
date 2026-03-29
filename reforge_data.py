"""
Project Icarus - 数据修复脚本 (reforge_data.py)
================================================
读取 raw_failures.json + funding_lookup.json，
调用本地 Ollama qwen2.5:14b 修复 industry / failure_reasons，
合并真实融资额，输出 icarus_cleansed_db_v3.json。
"""

import json
import logging
import sys

import ollama
from tqdm import tqdm

# ─── 日志配置 ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── 常量定义 ───────────────────────────────────────────
RAW_DATA_PATH = "./raw_failures.json"
FUNDING_LOOKUP_PATH = "./funding_lookup.json"
OUTPUT_PATH = "./icarus_cleansed_db_v3.json"
LLM_MODEL = "qwen2.5:14b"
LLM_OPTIONS = {"num_ctx": 4096, "temperature": 0.1}


def buildCleansingPrompt(company_name: str, failure_reasons: str, archetype: str) -> str:
    """
    构建数据清洗 Prompt。
    要求模型输出严格 JSON，包含 industry / failure_reasons / archetype 三个字段。
    """
    return f"""你是一个数据清洗专家。根据以下创业公司的信息，输出一个严格的JSON对象。

公司名：{company_name}
原始描述：{failure_reasons}

要求：
1. "industry"：根据公司名和描述，判断其所属行业，用精简的中文短语表示（如：外卖平台、社交网络、企业SaaS、智能硬件、在线教育、金融科技等）
2. "failure_reasons"：将原始英文描述翻译并浓缩为2-3句冰冷、精准的中文陈述，描述其死因
3. "archetype"：保留原值不变，原值为：{archetype}

只输出JSON，不要任何其他文字。格式：
{{"industry": "...", "failure_reasons": "...", "archetype": "..."}}"""


def callLlm(prompt: str) -> str:
    """
    调用本地 Ollama qwen2.5:14b，返回模型原始文本输出。
    低温 (0.1) 保证输出稳定。
    """
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=LLM_OPTIONS,
        stream=False,
    )
    return response["message"]["content"].strip()


def extractJson(raw_text: str) -> dict | None:
    """
    从模型输出中提取 JSON 对象。
    兼容模型在 JSON 外层包裹 markdown 代码块的情况。
    """
    text = raw_text.strip()

    # 剥离 markdown 代码块包裹
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉首行 (```json) 和末行 (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def main() -> None:
    # ─── 1. 加载数据源 ──────────────────────────────────
    logger.info("Loading raw data: %s", RAW_DATA_PATH)
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_records: list[dict] = json.load(f)
    logger.info("Loaded %d raw records", len(raw_records))

    logger.info("Loading funding lookup: %s", FUNDING_LOOKUP_PATH)
    with open(FUNDING_LOOKUP_PATH, "r", encoding="utf-8") as f:
        funding_lookup: dict[str, str] = json.load(f)
    logger.info("Loaded %d funding entries", len(funding_lookup))

    # ─── 2. 加载 icarus_cleansed_db.json 作为 archetype 兜底源 ──
    archetype_fallback: dict[str, str] = {}
    try:
        with open("./icarus_cleansed_db.json", "r", encoding="utf-8") as f:
            cleansed_records: list[dict] = json.load(f)
        for rec in cleansed_records:
            name = rec.get("company_name", "")
            arch = rec.get("archetype", "")
            if name and arch:
                archetype_fallback[name] = arch
        logger.info("Loaded archetype fallback from icarus_cleansed_db.json (%d entries)", len(archetype_fallback))
    except FileNotFoundError:
        logger.warning("icarus_cleansed_db.json not found, archetype fallback unavailable")

    # ─── 3. 逐条调用 LLM 修复 ──────────────────────────
    results: list[dict] = []
    success_count = 0
    fail_count = 0

    for record in tqdm(raw_records, desc="[REFORGE] Processing", unit="rec", ncols=90):
        company_name = record.get("company_name", "Unknown")
        # NOTE: raw_failures.json 中 failure_reasons 实际存在 fatal_action 或其他字段中
        # 优先取 fatal_action（英文失败描述），兜底取 failure_reasons
        raw_description = record.get("fatal_action", record.get("failure_reasons", ""))
        original_archetype = record.get("archetype", "Unknown")

        prompt = buildCleansingPrompt(company_name, raw_description, original_archetype)

        try:
            raw_output = callLlm(prompt)
            parsed = extractJson(raw_output)

            if parsed is None:
                logger.warning(
                    "[PARSE FAIL] %s | raw output: %s",
                    company_name,
                    raw_output[:200],
                )
                fail_count += 1
                continue

            # 字段合并
            reforged = {
                "company_name": company_name,
                "industry": parsed.get("industry", "Unknown"),
                "failure_reasons": parsed.get("failure_reasons", raw_description),
                "funding_amount": funding_lookup.get(company_name, "Unknown"),
                "archetype": parsed.get("archetype", archetype_fallback.get(company_name, original_archetype)),
            }
            results.append(reforged)
            success_count += 1

        except Exception as e:
            logger.warning("[LLM ERROR] %s | %s", company_name, e)
            fail_count += 1
            continue

    # ─── 4. 保存结果 ────────────────────────────────────
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ─── 5. 统计报告 ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[DONE] Data reforging complete")
    logger.info("  Output        : %s", OUTPUT_PATH)
    logger.info("  Total records : %d", len(raw_records))
    logger.info("  Success       : %d", success_count)
    logger.info("  Failed/Skipped: %d", fail_count)
    logger.info("  Saved         : %d", len(results))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
