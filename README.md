# Project Icarus — 创业认知偏差检测引擎

> 每一个失败的创业背后都有可识别的认知偏差模式。Icarus 通过检索历史失败案例，帮助创业者在决策前识别自己可能踩入的认知陷阱。

![Status](https://img.shields.io/badge/Status-v2.1-ff5540?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-100%25_Local_RAG-131313?style=flat-square&labelColor=930100)
![LLM](https://img.shields.io/badge/LLM-Qwen2.5_14B-blue?style=flat-square)
![Embedding](https://img.shields.io/badge/Embedding-BGE--M3-green?style=flat-square)
![Database](https://img.shields.io/badge/Database-ChromaDB-orange?style=flat-square)

---

## Why This Project?

创业失败率高达90%，但失败模式是有限且可识别的。大量创业者反复踩入相同的认知陷阱——沉没成本谬误、回音室效应、货物崇拜、伊卡洛斯综合征。

Icarus 将55个真实商业失败案例结构化入库，通过语义检索匹配用户的创业想法与历史失败模式，输出认知偏差分析报告。

**核心设计原则：100% 本地运行。** 创业想法属于高度敏感信息，不应经过任何第三方服务器。全部推理在用户本地 GPU 上完成，零远程 API 调用，零数据泄露。

---

## How It Works

用户输入一段创业想法，系统执行三阶段串行 Pipeline：

**阶段一：语义检索** — BGE-M3 多语言模型将输入向量化，在 ChromaDB 中检索最相似的 Top-3 历史失败案例。内置 distance 门控机制，当检索置信度不足时主动拒绝生成，避免低质量输出。

**阶段二：Prompt 组装** — 将匹配到的失败案例（行业、死因、认知偏差类型）拼装为结构化上下文，注入本地 LLM。

**阶段三：流式生成** — Qwen2.5:14b 基于历史案例生成认知偏差分析报告，包含偏差锁定、历史镜像、风险评估和应对建议。前端通过 SSE 实现打字机效果实时渲染。

---

## Tech Stack

| Layer | Technology | Why This Choice |
|-------|-----------|----------------|
| Backend | FastAPI + Uvicorn | 轻量异步框架，支持 SSE 流式输出 |
| Vector DB | ChromaDB (Local SQLite) | 本地持久化，无需云服务 |
| Embedding | BGE-M3 (Ollama) | 多语言优化，中文语义理解显著优于 nomic-embed-text |
| LLM | Qwen2.5:14b (Ollama) | 中文推理能力强，14B 参数适配 16GB VRAM |
| Frontend | Vanilla JS + Tailwind CSS | 零依赖，SSE 打字机渲染 |
| Hardware | RTX 5080 16GB VRAM | 支持 14B 模型全本地推理 |

---

## Technical Decisions

**为什么用 ChromaDB 而不是 Pinecone？** 产品定位要求 100% 本地隐私，云端向量数据库不符合需求。

**为什么 Embedding 文本排除 company_name？** 防止公司名称的字面相似度污染语义检索结果。只用 industry / failure_reasons / archetype 三个字段做向量化。

**为什么从 nomic-embed-text 迁移到 BGE-M3？** 数据集已全部中文化，BGE-M3 的多语言训练使其对中文语义的表征质量显著优于英文为主的 nomic 模型。实测同一 query 的召回相关性从不相关提升到精准命中。

**为什么加 distance 门控？** 当知识库中没有相关案例时，强行生成会导致 LLM 编造内容。distance > 0.65 时系统主动拒绝，保证输出质量下限。

---

## Data

55 个精选商业失败案例，经多轮清洗重铸：

- **industry**: 中文行业标签（社交网络、食品科技、电商等 8+ 行业）
- **failure_reasons**: 中文深度死因分析（由 Qwen2.5:14b 本地生成）
- **funding_amount**: 真实融资金额（交叉验证）
- **archetype**: 认知偏差原型（Sunk Cost Fallacy, Echo Chamber, Cargo Cult, Icarus Syndrome 等）

数据来源：Failory 创业失败数据库，通过 Playwright 自动化爬取 + 本地 LLM 清洗重铸。

---

## Quick Start

```bash
git clone https://github.com/hyang1490-jpg/Maverick_RAG_Framework.git
cd Maverick_RAG_Framework
pip install -r requirements.txt

ollama pull qwen2.5:14b
ollama pull bge-m3

python ingest_v2.py
python api_server.py
```

Open `index.html` in browser → Input a startup idea → Get analysis.

---

## Roadmap

- 扩库至 200+ 案例，覆盖 8+ 行业
- 多维度检索（按行业 / 认知偏差类型 / 融资阶段分别召回）
- 认知偏差雷达图可视化
- 存活率评分仪表盘

---

## Author

**Yang Hao** — AI Product & Risk Intelligence

MIT License
