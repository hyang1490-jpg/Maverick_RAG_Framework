"""
Project Icarus - 数据补丁脚本 (patch_v3.py)
============================================
修复 icarus_cleansed_db_v3.json 中的两个数据污染问题:
1. funding_amount 被 PowerShell 截断 $ 符号 → 用硬编码字典覆写
2. archetype 全变 Unknown → 从 icarus_cleansed_db.json 回填
"""

import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── 硬编码融资额字典 (真实数据，不经过 PowerShell) ─────
FUNDING: dict[str, str] = {
    "Dazo": "$1.8M", "Yik Yak": "$73.5M", "PepperTap": "$51.2M", "Zoomo": "$7M",
    "Quibi": "$1.75B", "Pixate": "$3.8M", "Musical.ly": "$16.4M", "Justin.tv": "$8M",
    "Toys R Us": "未公开", "Houseparty": "$70M", "Xinja": "$71M", "Quirky": "$185M",
    "Atrium": "$75.5M", "Glitch": "未公开", "Yogome": "$36M", "Tink Labs": "$160M",
    "Daqri": "$275M", "Katerra": "$2.2B", "HubHaus": "$11M", "ScaleFactor": "$104M",
    "Desti": "$2M", "HotelsAroundYou": "未公开", "HiGear": "$1.3M", "Zirtual": "$5.5M",
    "RoomsTonite": "$1.5M", "Secret": "$35M", "Rafter": "$58M", "Netscape": "未公开",
    "Sharingear": "未公开", "Zulily": "$138M", "EventVue": "$400K", "Totsy": "$34M",
    "Skully": "$14.5M", "Verelo": "未公开", "Wantful": "$5.5M", "Gowalla": "$10.4M",
    "PostRocket": "未公开", "Wesabe": "$4.7M", "99dresses": "$3M", "Lookery": "$2M",
    "Argyle Social": "$1.7M", "Karhoo": "$250M", "ArsDigita": "$40M", "PoliMobile": "未公开",
    "Tilt": "$62M", "Delicious": "未公开", "Move Loot": "$22M", "QBotix": "$12.5M",
    "Fuhu": "$85M", "SellanApp": "未公开", "Formspring": "$14.3M", "Reach.ly": "未公开",
    "FoundationDB": "$22.7M", "Stayzilla": "$34M", "Auctionata": "$95M",
}

V3_PATH = "./icarus_cleansed_db_v3.json"
OLD_CLEANSED_PATH = "./icarus_cleansed_db.json"
FUNDING_LOOKUP_PATH = "./funding_lookup.json"


def main() -> None:
    # ─── 1. 重建 funding_lookup.json ────────────────────
    with open(FUNDING_LOOKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(FUNDING, f, ensure_ascii=False, indent=2)
    logger.info("funding_lookup.json rebuilt (%d entries)", len(FUNDING))

    # ─── 2. 构建 archetype 查找表 ──────────────────────
    with open(OLD_CLEANSED_PATH, "r", encoding="utf-8") as f:
        old_records: list[dict] = json.load(f)

    archetype_lookup: dict[str, str] = {}
    for rec in old_records:
        name = rec.get("company_name", "")
        arch = rec.get("archetype", "")
        if name and arch:
            archetype_lookup[name] = arch
    logger.info("Archetype lookup built from icarus_cleansed_db.json (%d entries)", len(archetype_lookup))

    # ─── 3. 加载 v3 数据 ───────────────────────────────
    with open(V3_PATH, "r", encoding="utf-8") as f:
        v3_records: list[dict] = json.load(f)
    logger.info("Loaded %d records from %s", len(v3_records), V3_PATH)

    # ─── 4. 逐条修复 ──────────────────────────────────
    archetype_fixed = 0
    funding_fixed = 0

    for rec in v3_records:
        name = rec.get("company_name", "")

        # 修复 archetype
        if rec.get("archetype") == "Unknown" and name in archetype_lookup:
            rec["archetype"] = archetype_lookup[name]
            archetype_fixed += 1

        # 修复 funding_amount (全量覆写)
        old_funding = rec.get("funding_amount", "")
        new_funding = FUNDING.get(name, old_funding)
        if new_funding != old_funding:
            funding_fixed += 1
        rec["funding_amount"] = new_funding

    # ─── 5. 覆盖保存 ──────────────────────────────────
    with open(V3_PATH, "w", encoding="utf-8") as f:
        json.dump(v3_records, f, ensure_ascii=False, indent=2)

    # ─── 6. 统计报告 ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("[DONE] Patch applied to %s", V3_PATH)
    logger.info("  Archetype fixed : %d", archetype_fixed)
    logger.info("  Funding fixed   : %d", funding_fixed)
    logger.info("  Total records   : %d", len(v3_records))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
