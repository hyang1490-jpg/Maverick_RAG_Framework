#!/usr/bin/env python3
"""
Project Icarus - LLM 数据提纯流水线 v1.0
==========================================
功能: 读取 raw_failures.json → 逐条送入本地 Ollama 大模型 → 犯罪心理侧写式分析
     → 输出标准化的 icarus_cleansed_db.json

核心侧写字段:
  - archetype:         认知偏差原型（上帝情结 / 信息茧房 / 偏执等）
  - fatal_action:      导致崩盘的极其具体的致命动作
  - trigger_condition:  诱发崩盘的客观或主观触发条件
  - ultimate_outcome:   极其惨痛的最终结局

依赖: requests (已在 requirements.txt)
运行: python llm_processor.py
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

# =============================================================================
# 日志配置
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llm_processor")

# =============================================================================
# 可配置参数 —— 全部集中在这里，方便一键调参
# =============================================================================

# Ollama 服务地址
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"

# 模型名称（灵活配置入口）
MODEL_NAME = "qwen2.5:14b"

# 输入 / 输出文件
INPUT_FILE = Path("raw_failures.json")
OUTPUT_FILE = Path("icarus_cleansed_db.json")

# 重试配置
MAX_RETRIES = 3          # 单条案例最大重试次数
RETRY_DELAY = 2.0        # 重试间隔（秒）

# LLM 推理参数
LLM_TEMPERATURE = 0.3    # 低温 = 更确定性、更冰冷客观
LLM_NUM_PREDICT = 1024   # 最大输出 token 数

# =============================================================================
# 犯罪心理侧写师 System Prompt —— 冰冷、客观、无情
# =============================================================================

SYSTEM_PROMPT = """你是一名顶级犯罪心理侧写师，专门分析商业领域的"犯罪现场"——失败的初创公司。
你的分析必须像尸检报告一样冰冷、客观、精准。不要有任何同情心，不要使用任何安慰性的语言。
你只关心事实和因果链条。

你的任务：根据提供的失败初创公司原始资料，提取并输出以下 4 个侧写字段。

输出要求（严格遵守）：
1. 只输出一个 JSON 对象，不要有任何前缀、后缀、解释、markdown 格式
2. JSON 必须包含且仅包含以下 4 个字段
3. 所有值必须使用英文
4. 不要输出 ```json 代码块标记

字段定义：
- "archetype": 认知偏差原型。从以下类型中选择最匹配的一个（可自创）：
  God Complex（上帝情结）/ Echo Chamber（信息茧房）/ Paranoid Delusion（偏执妄想）/
  Icarus Syndrome（伊卡洛斯综合症，飞得太高）/ Ostrich Effect（鸵鸟效应，逃避现实）/
  Cargo Cult（货物崇拜，模仿表面不理解本质）/ Dunning-Kruger（达克效应，能力不足却自信过度）/
  Sunk Cost Fallacy（沉没成本谬误）/ Groupthink（群体思维）/ Survivorship Bias（幸存者偏差）

- "fatal_action": 导致公司崩盘的最具体、最致命的那一个动作或决策。必须是可操作级别的描述，不能是笼统的总结。

- "trigger_condition": 从外部环境或内部管理中，精准定位那个引爆崩盘的触发条件。可以是市场变化、竞争对手动作、资金链断裂时间点、关键人物离开等。

- "ultimate_outcome": 用一句话描述这家公司最惨痛的结局。要冰冷、直白、带有讽刺性。"""


def build_user_prompt(case: dict[str, Any]) -> str:
    """根据单条原始案例数据，构建发送给 LLM 的用户提示词"""
    return f"""目标公司: {case.get('company_name', 'Unknown')}
行业: {case.get('industry', 'N/A')}
融资金额: {case.get('funding_amount', 'N/A')}
失败原因原始记录: {case.get('failure_reasons', 'N/A')}
结局描述: {case.get('outcome', 'N/A')}
来源: {case.get('source_url', 'N/A')}

请立即输出侧写 JSON。"""


# =============================================================================
# Ollama 交互层
# =============================================================================

def check_ollama_health() -> bool:
    """检查 Ollama 服务是否在线"""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        logger.info(f"Ollama 在线，可用模型: {models}")

        # 检查目标模型是否已加载
        # NOTE: Ollama 的 model name 可能带 :latest 后缀，做模糊匹配
        model_base = MODEL_NAME.split(":")[0]
        if not any(model_base in m for m in models):
            logger.warning(f"目标模型 '{MODEL_NAME}' 未在本地找到，Ollama 会尝试自动拉取。")
        return True
    except requests.exceptions.ConnectionError:
        logger.error(f"无法连接 Ollama ({OLLAMA_BASE_URL})。请确认 Ollama 服务已启动。")
        return False
    except Exception as e:
        logger.error(f"Ollama 健康检查异常: {e}")
        return False


def call_ollama(system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    调用 Ollama generate API，返回模型原始文本输出。
    使用 stream=False 模式，一次性获取完整响应。
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_predict": LLM_NUM_PREDICT,
        },
    }

    try:
        resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except requests.exceptions.Timeout:
        logger.error("Ollama 推理超时（120s），可能显存不足或模型过大。")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Ollama 连接断开，服务可能已崩溃。")
        return None
    except Exception as e:
        logger.error(f"Ollama 调用异常: {e}")
        return None


# =============================================================================
# JSON 解析与修复层 —— 对 LLM 常见的格式错误做容错处理
# =============================================================================

REQUIRED_FIELDS = {"archetype", "fatal_action", "trigger_condition", "ultimate_outcome"}


def extract_json_from_response(raw_text: str) -> Optional[dict[str, str]]:
    """
    从 LLM 原始输出中提取 JSON 对象。
    容错策略:
    1. 先尝试直接 json.loads
    2. 再尝试正则提取 {...} 块
    3. 最后尝试修复常见格式问题（如尾部逗号）
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()

    # 策略 1: 直接解析
    result = _try_parse(text)
    if result:
        return result

    # 策略 2: 正则提取第一个 JSON 对象（处理 LLM 输出带前后缀的情况）
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        result = _try_parse(json_match.group(0))
        if result:
            return result

    # 策略 3: 处理嵌套 JSON（更宽松的匹配）
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        result = _try_parse(json_match.group(0))
        if result:
            return result

    logger.warning(f"JSON 解析全部失败，原始输出: {text[:200]}...")
    return None


def _try_parse(text: str) -> Optional[dict[str, str]]:
    """尝试解析 JSON 文本，含尾部逗号修复"""
    # 去除可能的 markdown 代码块标记
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # 修复尾部逗号 (trailing comma) —— LLM 常见毛病
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and REQUIRED_FIELDS.issubset(obj.keys()):
            return {k: str(v) for k, v in obj.items() if k in REQUIRED_FIELDS}
        elif isinstance(obj, dict):
            logger.warning(f"JSON 字段不完整，缺少: {REQUIRED_FIELDS - obj.keys()}")
            return None
    except json.JSONDecodeError:
        pass
    return None


# =============================================================================
# 主控流水线
# =============================================================================

def process_single_case(case: dict[str, Any], index: int, total: int) -> Optional[dict[str, Any]]:
    """
    处理单条案例: 构建提示词 → 调用 LLM → 解析输出 → 组装最终记录。
    内置重试机制，最多 MAX_RETRIES 次。
    """
    company = case.get("company_name", "Unknown")
    logger.info(f"[{index}/{total}] 正在侧写: {company}")

    user_prompt = build_user_prompt(case)

    for attempt in range(1, MAX_RETRIES + 1):
        raw_output = call_ollama(SYSTEM_PROMPT, user_prompt)

        if raw_output is None:
            logger.warning(f"  第 {attempt}/{MAX_RETRIES} 次: LLM 无响应，{RETRY_DELAY}s 后重试...")
            time.sleep(RETRY_DELAY)
            continue

        profile = extract_json_from_response(raw_output)

        if profile is not None:
            # 组装最终记录: 原始字段 + 侧写字段
            cleansed = {
                "company_name": company,
                "funding_amount": case.get("funding_amount", "N/A"),
                "archetype": profile["archetype"],
                "fatal_action": profile["fatal_action"],
                "trigger_condition": profile["trigger_condition"],
                "ultimate_outcome": profile["ultimate_outcome"],
            }
            logger.info(f"  ✓ 侧写完成 | 原型: {profile['archetype']}")
            return cleansed

        logger.warning(f"  第 {attempt}/{MAX_RETRIES} 次: JSON 解析失败，重试中...")
        time.sleep(RETRY_DELAY)

    # 全部重试失败 —— 用兜底数据，不让流水线崩溃
    logger.error(f"  ✗ {company}: {MAX_RETRIES} 次重试全部失败，使用兜底数据")
    return {
        "company_name": company,
        "funding_amount": case.get("funding_amount", "N/A"),
        "archetype": "Analysis Failed",
        "fatal_action": "LLM processing failed after retries",
        "trigger_condition": "N/A",
        "ultimate_outcome": case.get("outcome", "N/A"),
    }


def main():
    """主入口"""
    logger.info("=" * 60)
    logger.info("  Project Icarus - LLM 数据提纯流水线 v1.0")
    logger.info(f"  模型: {MODEL_NAME}")
    logger.info(f"  输入: {INPUT_FILE}")
    logger.info(f"  输出: {OUTPUT_FILE}")
    logger.info("=" * 60)

    # 1. 检查 Ollama 服务
    logger.info("[1/4] 检查 Ollama 服务状态...")
    if not check_ollama_health():
        logger.error("Ollama 服务不可用，流水线中止。")
        sys.exit(1)

    # 2. 加载原始数据
    logger.info("[2/4] 加载原始数据...")
    if not INPUT_FILE.exists():
        logger.error(f"输入文件不存在: {INPUT_FILE.resolve()}")
        sys.exit(1)

    raw_data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    total = len(raw_data)
    logger.info(f"  → 加载 {total} 条原始案例")

    # 3. 逐条送入 LLM 提纯
    logger.info(f"[3/4] 开始 LLM 侧写分析（预计耗时较长，每条约 10-30s）...")
    cleansed_results: list[dict[str, Any]] = []
    success_count = 0
    fail_count = 0

    for i, case in enumerate(raw_data, 1):
        result = process_single_case(case, i, total)
        if result:
            cleansed_results.append(result)
            if result["archetype"] != "Analysis Failed":
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1

        # 每处理 10 条输出一次进度
        if i % 10 == 0:
            logger.info(f"  --- 进度: {i}/{total} | 成功: {success_count} | 失败: {fail_count} ---")

    # 4. 写入输出文件
    logger.info(f"[4/4] 写入提纯数据 → {OUTPUT_FILE}")
    OUTPUT_FILE.write_text(
        json.dumps(cleansed_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    logger.info(f"  提纯完成！")
    logger.info(f"  总计: {total} 条 | 成功: {success_count} 条 | 失败: {fail_count} 条")
    logger.info(f"  输出: {OUTPUT_FILE.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
