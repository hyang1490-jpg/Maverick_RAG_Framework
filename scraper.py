#!/usr/bin/env python3
"""
Project Icarus - 失败初创公司数据收割机 v2.0 (Playwright 重装版)
================================================================
目标靶场: Failory Cemetery (https://www.failory.com/cemetery)
升级内容: Playwright 无头浏览器 → 自动滚动 + Load More 点击 → 炸出全部懒加载卡片
不变内容: 详情页解析逻辑 / 反爬延迟策略 / JSON 输出结构
 
作者: 手哥 x Claude
日期: 2026-03-22
"""
 
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
 
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, Browser
 
# =============================================================================
# 配置区
# =============================================================================
 
BASE_URL = "https://www.failory.com"
LIST_URL = f"{BASE_URL}/cemetery"
 
# 抓取上限（设 0 = 不限制，全部拿下）
MAX_ITEMS = 0
 
OUTPUT_FILE = Path("raw_failures.json")
 
# 详情页请求间隔（秒）—— 做人留一线
MIN_DELAY = 1.5
MAX_DELAY = 3.5
 
# Playwright 滚动配置
SCROLL_PAUSE = 2.0        # 每次滚动后等待加载的秒数
SCROLL_STEP = 800          # 每次滚动的像素步长
MAX_SCROLL_ATTEMPTS = 50   # 最大滚动次数（防止死循环）
LOAD_MORE_TIMEOUT = 3000   # Load More 按钮点击后等待毫秒数
 
# 随机 User-Agent 池
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]
 
 
# =============================================================================
# 数据模型（与 v1 完全一致）
# =============================================================================
 
@dataclass
class FailureCase:
    """单个失败案例的标准化数据结构"""
    company_name: str
    industry: str
    funding_amount: Optional[str]
    failure_reasons: str
    outcome: str
    source_url: str
 
 
# =============================================================================
# Playwright 列表页引擎 —— 重型装甲核心
# =============================================================================
 
class PlaywrightListEngine:
    """
    用 Playwright 无头浏览器加载列表页。
    核心能力: 自动滚动触底 + 自动点击 Load More 按钮 → 炸出所有懒加载卡片。
    """
 
    def __init__(self):
        self.browser: Optional[Browser] = None
 
    async def launch(self):
        """启动无头浏览器"""
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",  # 隐藏自动化特征
                "--no-sandbox",
            ]
        )
        print("  [✓] Playwright Chromium 已启动（无头模式）")
 
    async def shutdown(self):
        """关闭浏览器，释放资源"""
        if self.browser:
            await self.browser.close()
        await self.pw.stop()
        print("  [✓] Playwright 已关闭")
 
    async def fetch_full_list_html(self, url: str) -> str:
        """
        核心方法: 打开列表页 → 滚动到底 + 点击 Load More → 返回完全展开的 HTML。
 
        策略:
        1. 先检查有没有 "Load More" / "Show More" 按钮，有就反复点
        2. 同时配合 scrollTo 滚动，触发 Intersection Observer 懒加载
        3. 双重终止条件: 页面高度不再增加 + 按钮消失
        """
        context = await self.browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        page = await context.new_page()
 
        # 注入反检测脚本 —— 隐藏 webdriver 标记
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
        """)
 
        print(f"  [→] 正在加载: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(2000)  # 等首屏渲染稳定
 
        # ── 滚动 + Load More 循环 ──
        prev_height = 0
        stale_count = 0  # 连续高度不变的次数
        scroll_count = 0
 
        print("  [↓] 开始自动滚动，炸出所有隐藏卡片...")
 
        while scroll_count < MAX_SCROLL_ATTEMPTS:
            scroll_count += 1
 
            # 1) 尝试点击 Load More 按钮
            clicked = await self._try_click_load_more(page)
            if clicked:
                stale_count = 0  # 点击成功，重置停滞计数
 
            # 2) 滚动一步
            await page.evaluate(f"window.scrollBy(0, {SCROLL_STEP})")
            await page.wait_for_timeout(int(SCROLL_PAUSE * 1000))
 
            # 3) 检查页面高度是否还在增长
            current_height = await page.evaluate("document.body.scrollHeight")
 
            if current_height == prev_height:
                stale_count += 1
                # 连续 3 次高度不变 = 彻底触底
                if stale_count >= 3:
                    print(f"  [■] 触底确认！连续 {stale_count} 次高度无变化 (scrollHeight={current_height})")
                    break
            else:
                stale_count = 0
                prev_height = current_height
 
            # 进度反馈（每 5 次汇报一次）
            if scroll_count % 5 == 0:
                card_count = await page.evaluate(
                    "document.querySelectorAll('a[href*=\"/cemetery/\"]').length"
                )
                print(f"  [↓] 已滚动 {scroll_count} 次 | 页面高度: {current_height}px | 已发现卡片: {card_count}")
 
        # 最后再来一次完全触底滚动，确保万无一失
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)
 
        # 拿到完全展开的 HTML
        html = await page.content()
        final_cards = await page.evaluate(
            "document.querySelectorAll('a[href*=\"/cemetery/\"]').length"
        )
        print(f"  [✓] 页面完全展开！总滚动 {scroll_count} 次 | 最终卡片数: {final_cards}")
 
        await context.close()
        return html
 
    async def _try_click_load_more(self, page: Page) -> bool:
        """
        尝试查找并点击 Load More / Show More 类型的按钮。
        Failory (Webflow) 常见的按钮选择器全覆盖。
        返回: 是否成功点击。
        """
        selectors = [
            # 文本匹配
            "button:has-text('Load More')",
            "button:has-text('Show More')",
            "button:has-text('Load more')",
            "button:has-text('Show more')",
            "a:has-text('Load More')",
            "a:has-text('Show More')",
            # Webflow 分页按钮
            ".w-pagination-next",
            "[class*='load-more']",
            "[class*='show-more']",
            "[class*='pagination'] a:last-child",
        ]
 
        for selector in selectors:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=500):
                    await btn.scroll_into_view_if_needed()
                    await btn.click()
                    await page.wait_for_timeout(LOAD_MORE_TIMEOUT)
                    print(f"  [⊕] 成功点击: {selector}")
                    return True
            except Exception:
                continue
 
        return False
 
 
# =============================================================================
# 详情页网络层 —— 保持 requests 轻装（详情页不需要 JS 渲染）
# =============================================================================
 
class HttpClient:
    """带反爬意识的 HTTP 客户端（仅用于详情页）"""
 
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
 
    def get(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """GET 请求 → BeautifulSoup，内置重试 + 限流处理"""
        for attempt in range(1, retries + 1):
            self.session.headers["User-Agent"] = random.choice(USER_AGENTS)
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                return BeautifulSoup(resp.text, "html.parser")
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "N/A"
                print(f"  [!] HTTP {status} @ {url} (第 {attempt}/{retries} 次)")
                if status == 429:
                    cooldown = 10 * attempt
                    print(f"  [!] 触发限流，冷却 {cooldown}s...")
                    time.sleep(cooldown)
                elif status in (403, 404):
                    return None
            except requests.exceptions.RequestException as e:
                print(f"  [!] 网络异常: {e} (第 {attempt}/{retries} 次)")
            time.sleep(random.uniform(2, 5))
        print(f"  [✗] 彻底失败: {url}")
        return None
 
    def polite_sleep(self):
        """礼貌延迟"""
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
 
 
# =============================================================================
# 解析层（与 v1 完全一致，一字未改）
# =============================================================================
 
class FailoryParser:
    """Failory 网站专用解析器"""
 
    @staticmethod
    def parse_list_page(soup: BeautifulSoup) -> list[dict]:
        """解析列表页 HTML，提取每个案例的链接和基础信息。"""
        items = []
 
        cards = soup.select(
            ".collection-list .collection-item a, "
            ".w-dyn-list .w-dyn-item a, "
            "[class*='cemetery'] a[href*='/cemetery/'], "
            "a[href*='/cemetery/']"
        )
 
        if not cards:
            cards = soup.find_all("a", href=re.compile(r"/(cemetery|failures)/[a-z0-9-]+"))
 
        seen_urls = set()
        for card in cards:
            href = card.get("href", "")
            if not href or href in seen_urls:
                continue
 
            full_url = href if href.startswith("http") else f"{BASE_URL}{href}"
            if full_url.count("/") < 4:
                continue
 
            seen_urls.add(href)
 
            name = ""
            industry = ""
            snippet = ""
 
            heading = card.select_one("h2, h3, h4, [class*='heading'], [class*='name'], [class*='title']")
            if heading:
                name = heading.get_text(strip=True)
 
            tag_el = card.select_one("[class*='tag'], [class*='category'], [class*='industry']")
            if tag_el:
                industry = tag_el.get_text(strip=True)
 
            desc_el = card.select_one("p, [class*='description'], [class*='excerpt']")
            if desc_el:
                snippet = desc_el.get_text(strip=True)
 
            if not name:
                name = card.get_text(strip=True)[:80]
 
            items.append({
                "name": name,
                "url": full_url,
                "industry": industry,
                "snippet": snippet,
            })
 
        return items
 
    @staticmethod
    def parse_detail_page(soup: BeautifulSoup, fallback: dict) -> FailureCase:
        """解析单个案例详情页，提取完整结构化信息。"""
        # --- 公司名 ---
        name = ""
        h1 = soup.find("h1")
        if h1:
            name = h1.get_text(strip=True)
        name = name or fallback.get("name", "Unknown")
 
        # --- 行业 ---
        industry = fallback.get("industry", "")
        if not industry:
            for label_text in ["Industry", "Sector", "Category", "行业"]:
                label = soup.find(string=re.compile(label_text, re.I))
                if label:
                    parent = label.find_parent()
                    if parent:
                        sibling = parent.find_next_sibling()
                        if sibling:
                            industry = sibling.get_text(strip=True)
                            break
                        industry = parent.get_text(strip=True).replace(label_text, "").strip(": ")
                        if industry:
                            break
 
        # --- 融资金额 ---
        funding = ""
        funding_patterns = [
            re.compile(r"\$[\d,.]+ ?[MBKmillion|billion|thousand]*", re.I),
            re.compile(r"raised .*?\$[\d,.]+[MBK]?", re.I),
            re.compile(r"funding.*?\$[\d,.]+", re.I),
        ]
        for label_text in ["Funding", "Raised", "Total Raised", "Investment"]:
            label = soup.find(string=re.compile(label_text, re.I))
            if label:
                context = label.find_parent()
                if context:
                    text = context.get_text()
                    for pat in funding_patterns:
                        match = pat.search(text)
                        if match:
                            funding = match.group(0).strip()
                            break
                if funding:
                    break
        if not funding:
            full_text = soup.get_text()
            for pat in funding_patterns:
                match = pat.search(full_text)
                if match:
                    funding = match.group(0).strip()
                    break
 
        # --- 失败原因 ---
        failure_reasons = ""
        reason_keywords = [
            "why .* fail", "reason", "cause", "what went wrong",
            "lesson", "mistake", "postmortem", "post-mortem",
        ]
        for kw in reason_keywords:
            header = soup.find(string=re.compile(kw, re.I))
            if header:
                parent = header.find_parent(["h1", "h2", "h3", "h4", "strong", "b"])
                if parent:
                    paragraphs = []
                    for sib in parent.find_next_siblings():
                        if sib.name in ("h1", "h2", "h3", "h4"):
                            break
                        if sib.name == "p":
                            paragraphs.append(sib.get_text(strip=True))
                        elif sib.name in ("ul", "ol"):
                            paragraphs.append(sib.get_text(separator="; ", strip=True))
                    if paragraphs:
                        failure_reasons = " ".join(paragraphs)
                        break
        if not failure_reasons:
            all_p = soup.select("article p, .rich-text p, .w-richtext p, main p")
            texts = [p.get_text(strip=True) for p in all_p if len(p.get_text(strip=True)) > 50]
            failure_reasons = " ".join(texts[:3])
        if not failure_reasons:
            failure_reasons = fallback.get("snippet", "N/A")
        if len(failure_reasons) > 800:
            failure_reasons = failure_reasons[:800] + "..."
 
        # --- 最终结局 ---
        outcome = ""
        for kw in ["Outcome", "Status", "Result", "What happened", "Shut down", "Acquired"]:
            label = soup.find(string=re.compile(kw, re.I))
            if label:
                parent = label.find_parent()
                if parent:
                    outcome = parent.get_text(strip=True)
                    outcome = re.sub(rf"^{kw}\s*:?\s*", "", outcome, flags=re.I).strip()
                    if outcome:
                        break
        if not outcome:
            full_text = soup.get_text()
            shutdown_match = re.search(
                r"(shut ?down|closed|ceased|bankrupt|acquired by|pivoted|dissolved).*?\.",
                full_text, re.I
            )
            if shutdown_match:
                outcome = shutdown_match.group(0).strip()
        outcome = outcome or "Shut down (details unavailable)"
 
        return FailureCase(
            company_name=name,
            industry=industry or "N/A",
            funding_amount=funding or "N/A",
            failure_reasons=failure_reasons or "N/A",
            outcome=outcome,
            source_url=fallback.get("url", ""),
        )
 
 
# =============================================================================
# 主控流程（async 重构）
# =============================================================================
 
async def run_scraper():
    """主流程: Playwright 列表页 → requests 详情页 → JSON"""
    print("=" * 60)
    print("  Project Icarus v2.0 - Playwright 重装版")
    print("  目标: Failory Cemetery (全量懒加载穿透)")
    limit_str = f"{MAX_ITEMS} 条" if MAX_ITEMS > 0 else "无上限（全部拿下）"
    print(f"  抓取上限: {limit_str}")
    print("=" * 60)
 
    # ── 第一步: Playwright 炸开列表页 ──
    print(f"\n[1/3] 启动 Playwright，加载列表页...")
    engine = PlaywrightListEngine()
    await engine.launch()
 
    try:
        full_html = await engine.fetch_full_list_html(LIST_URL)
    finally:
        await engine.shutdown()
 
    # 把完全展开的 HTML 交给 BeautifulSoup 解析
    list_soup = BeautifulSoup(full_html, "html.parser")
    parser = FailoryParser()
    items = parser.parse_list_page(list_soup)
    print(f"  → 列表解析完成，共发现 {len(items)} 个案例链接")
 
    if not items:
        # 备用路径
        print("[!] 主列表为空，尝试备用路径 /failures ...")
        engine2 = PlaywrightListEngine()
        await engine2.launch()
        try:
            backup_html = await engine2.fetch_full_list_html(f"{BASE_URL}/failures")
        finally:
            await engine2.shutdown()
        backup_soup = BeautifulSoup(backup_html, "html.parser")
        items = parser.parse_list_page(backup_soup)
        print(f"  → 备用路径发现 {len(items)} 个链接")
 
    if not items:
        print("[✗] 未发现任何案例链接。网站结构可能已变更。")
        sys.exit(1)
 
    # 应用抓取上限
    if MAX_ITEMS > 0:
        items = items[:MAX_ITEMS]
    total = len(items)
    print(f"  → 本次抓取: {total} 条")
 
    # ── 第二步: requests 逐个抓取详情页（保持轻装） ──
    print(f"\n[2/3] 开始抓取详情页（requests 轻装模式）...")
    client = HttpClient()
    results: list[dict] = []
 
    for i, item in enumerate(items, 1):
        url = item["url"]
        name = item.get("name", "?")
        print(f"  [{i}/{total}] {name} → {url}")
 
        detail_soup = client.get(url)
        if detail_soup:
            case = parser.parse_detail_page(detail_soup, fallback=item)
            results.append(asdict(case))
            print(f"         ✓ {case.company_name} | {case.industry} | {case.funding_amount}")
        else:
            results.append(asdict(FailureCase(
                company_name=name,
                industry=item.get("industry", "N/A"),
                funding_amount="N/A",
                failure_reasons=item.get("snippet", "N/A"),
                outcome="N/A",
                source_url=url,
            )))
            print(f"         ✗ 详情页失败，已用预览数据兜底")
 
        if i < total:
            client.polite_sleep()
 
    # ── 第三步: 输出 JSON ──
    print(f"\n[3/3] 写入 {OUTPUT_FILE}...")
    OUTPUT_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"  → 成功写入 {len(results)} 条记录")
    print(f"  → 文件路径: {OUTPUT_FILE.resolve()}")
    print(f"\n{'=' * 60}")
    print("  收割完成。全量数据已就绪，供 AirSense 侧写引擎消费。")
    print(f"{'=' * 60}")
 
 
# =============================================================================
# 入口
# =============================================================================
 
if __name__ == "__main__":
    asyncio.run(run_scraper())
 