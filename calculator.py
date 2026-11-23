"""每日耗费计算核心（不依赖 AstrBot，方便单测与离线核对）。

支持两种数据源：
1. AstrBot 本地 SQLite ``data_v4.db`` 的 ``provider_stats`` 表（按 DeepSeek 官方两档价实时估算）。
2. DeepSeek 控制台导出的用量 CSV / ZIP（``amount-YYYY-MM-DD_YYYY-MM-DD.csv``）。
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

CST = timezone(timedelta(hours=8))

# DeepSeek 官方两档定价（元 / 百万 token，deepseek-flash / deepseek-chat）
# 高峰：北京时间工作日 09:00-12:00、14:00-18:00
DEFAULT_PRICING = {
    "peak": {"hit": 0.04, "miss": 2.0, "output": 8.0},
    "off": {"hit": 0.02, "miss": 1.0, "output": 4.0},
}

# CSV 中的单价阈值（元 / token），用于区分高峰与空闲行
_PEAK_UNIT_PRICES = (0.000002, 0.00000004, 0.000008)


@dataclass
class DayCost:
    """单日耗费汇总。"""

    date: str
    requests: int = 0
    peak_requests: int = 0
    miss_tokens: int = 0
    hit_tokens: int = 0
    output_tokens: int = 0
    peak_cost: float = 0.0
    off_cost: float = 0.0
    extra_requests: int = 0
    extra_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.peak_cost + self.off_cost

    @property
    def hit_rate(self) -> float:
        total_in = self.miss_tokens + self.hit_tokens
        return (self.hit_tokens / total_in * 100.0) if total_in > 0 else 0.0


def is_peak_cst(dt_cst: datetime) -> bool:
    """判断北京时间是否处于 DeepSeek 高峰时段。"""
    if dt_cst.weekday() >= 5:
        return False
    h = dt_cst.hour
    return (9 <= h < 12) or (14 <= h < 18)


def parse_utc_to_cst(ts: str) -> datetime:
    """将 provider_stats.created_at（UTC ISO 字符串）转为北京时间。"""
    raw = str(ts).strip()
    if raw.endswith("Z"):
        raw = raw[:-1]
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(CST)


def price_for_dt(dt_cst: datetime, pricing: dict | None = None) -> tuple[float, float, float]:
    """返回 (cache_hit, cache_miss, output)，单位：元/百万 token。"""
    p = pricing or DEFAULT_PRICING
    tier = p["peak"] if is_peak_cst(dt_cst) else p["off"]
    return float(tier["hit"]), float(tier["miss"]), float(tier["output"])


def calc_from_db(
    db_path: Path | str,
    days: int = 7,
    target_date: str | None = None,
    pricing: dict | None = None,
    now_cst: datetime | None = None,
    extra_jsonl: Path | str | None = None,
) -> dict[str, DayCost]:
    """从 AstrBot data_v4.db 的 provider_stats 表计算每日耗费，可选合并插件直连补记账本。"""
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"数据库不存在: {db}")

    ref_now = now_cst or datetime.now(CST)
    if target_date:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=CST)
        start_cst = target_dt
        end_cst = target_dt + timedelta(days=1)
    else:
        days = max(1, int(days))
        today_start = ref_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_cst = today_start - timedelta(days=days - 1)
        end_cst = today_start + timedelta(days=1)

    since_utc = (start_cst - timedelta(hours=1)).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT created_at, token_input_other, token_input_cached, token_output "
            "FROM provider_stats WHERE created_at >= ?",
            (since_utc,),
        ).fetchall()
    finally:
        conn.close()

    result: dict[str, DayCost] = {}
    for created_at, miss, hit, out in rows:
        try:
            dt_cst = parse_utc_to_cst(created_at)
        except Exception:  # noqa: BLE001
            continue
        if not (start_cst <= dt_cst < end_cst):
            continue

        day_key = dt_cst.strftime("%Y-%m-%d")
        d = result.setdefault(day_key, DayCost(date=day_key))
        miss_i = int(miss or 0)
        hit_i = int(hit or 0)
        out_i = int(out or 0)

        p_hit, p_miss, p_out = price_for_dt(dt_cst, pricing)
        cost = miss_i / 1e6 * p_miss + hit_i / 1e6 * p_hit + out_i / 1e6 * p_out

        d.requests += 1
        d.miss_tokens += miss_i
        d.hit_tokens += hit_i
        d.output_tokens += out_i
        if is_peak_cst(dt_cst):
            d.peak_requests += 1
            d.peak_cost += cost
        else:
            d.off_cost += cost

    if extra_jsonl:
        _merge_extra_usage(Path(extra_jsonl), result, start_cst, end_cst, pricing)

    return dict(sorted(result.items()))


def _merge_extra_usage(
    jsonl_path: Path,
    result: dict[str, DayCost],
    start_cst: datetime,
    end_cst: datetime,
    pricing: dict | None = None,
) -> None:
    """将插件直连调用（未进 provider_stats）的用量合并到每日汇总中。"""
    if not jsonl_path.exists():
        return
    for line in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            dt_cst = datetime.fromisoformat(str(rec["ts"]))
            if dt_cst.tzinfo is None:
                dt_cst = dt_cst.replace(tzinfo=CST)
            else:
                dt_cst = dt_cst.astimezone(CST)
        except Exception:  # noqa: BLE001
            continue
        if not (start_cst <= dt_cst < end_cst):
            continue

        day_key = dt_cst.strftime("%Y-%m-%d")
        d = result.setdefault(day_key, DayCost(date=day_key))
        miss_i = int(rec.get("miss") or 0)
        hit_i = int(rec.get("hit") or 0)
        out_i = int(rec.get("out") or 0)

        p_hit, p_miss, p_out = price_for_dt(dt_cst, pricing)
        cost = miss_i / 1e6 * p_miss + hit_i / 1e6 * p_hit + out_i / 1e6 * p_out

        d.requests += 1
        d.extra_requests += 1
        d.extra_cost += cost
        d.miss_tokens += miss_i
        d.hit_tokens += hit_i
        d.output_tokens += out_i
        if is_peak_cst(dt_cst):
            d.peak_requests += 1
            d.peak_cost += cost
        else:
            d.off_cost += cost


def fetch_deepseek_balance(api_key: str, timeout: int = 8) -> dict[str, str]:
    """查询 DeepSeek 账户当前余额（返回 CNY 余额字典，失败抛异常）。"""
    req = urllib.request.Request(
        "https://api.deepseek.com/user/balance",
        headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    infos = data.get("balance_infos") or []
    for item in infos:
        if item.get("currency") == "CNY":
            return {
                "currency": "CNY",
                "total": str(item.get("total_balance", "0.00")),
                "topped_up": str(item.get("topped_up_balance", "0.00")),
                "granted": str(item.get("granted_balance", "0.00")),
            }
    if infos:
        item = infos[0]
        return {
            "currency": str(item.get("currency", "")),
            "total": str(item.get("total_balance", "0.00")),
            "topped_up": str(item.get("topped_up_balance", "0.00")),
            "granted": str(item.get("granted_balance", "0.00")),
        }
    return {"currency": "CNY", "total": "0.00", "topped_up": "0.00", "granted": "0.00"}


def _iter_csv_rows(file_path: Path) -> Iterable[dict[str, str]]:
    """读取单个 .csv 或 .zip（取其中所有 .csv）中的行。"""
    if file_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(file_path, "r") as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(".csv"):
                    text = zf.read(name).decode("utf-8-sig", errors="replace")
                    yield from csv.DictReader(io.StringIO(text))
    else:
        with file_path.open("r", encoding="utf-8-sig", newline="") as fh:
            yield from csv.DictReader(fh)


def calc_from_csv(path: Path | str) -> dict[str, DayCost]:
    """解析 DeepSeek 官方导出的用量 CSV 或 ZIP，返回按天汇总字典。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"账单文件不存在: {p}")

    files = (
        sorted(f for f in p.iterdir() if f.suffix.lower() in (".csv", ".zip"))
        if p.is_dir()
        else [p]
    )
    if not files:
        raise FileNotFoundError(f"目录下没有 .csv 或 .zip 账单文件: {p}")

    result: dict[str, DayCost] = {}
    # 若目录下有多个重叠账单，按 (day, type, price, model, api_key) 去重或只取最新文件
    target_files = [max(files, key=lambda f: f.stat().st_mtime)] if p.is_dir() else files

    for f in target_files:
        for row in _iter_csv_rows(f):
            day_key = str(row.get("start_time_iso", ""))[:10]
            if not day_key:
                continue
            rtype = str(row.get("type", "")).strip()
            amount = float(row.get("amount") or 0)
            unit_price_raw = str(row.get("price") or "").strip()
            unit_price = float(unit_price_raw) if unit_price_raw else 0.0

            d = result.setdefault(day_key, DayCost(date=day_key))
            if rtype == "request_count":
                d.requests += int(amount)
            else:
                cost = amount * unit_price
                is_peak_row = any(abs(unit_price - x) < 1e-12 for x in _PEAK_UNIT_PRICES)
                if is_peak_row:
                    d.peak_cost += cost
                else:
                    d.off_cost += cost

                if rtype == "input_cache_miss_tokens":
                    d.miss_tokens += int(amount)
                elif rtype == "input_cache_hit_tokens":
                    d.hit_tokens += int(amount)
                elif rtype == "output_tokens":
                    d.output_tokens += int(amount)

    return dict(sorted(result.items()))


def format_report(
    days_map: dict[str, DayCost],
    title: str = "每日耗费统计",
    note: str = "",
) -> str:
    """生成适合 QQ 阅读的纯文本耗费报告。"""
    if not days_map:
        return f"{title}\n这段时间没有调用记录。"

    lines = [title, ""]
    total_cost = 0.0
    total_req = 0
    total_miss = 0
    total_hit = 0
    total_out = 0
    total_extra_req = 0
    total_extra_cost = 0.0

    for day_key in sorted(days_map.keys(), reverse=True):
        d = days_map[day_key]
        total_cost += d.total_cost
        total_req += d.requests
        total_miss += d.miss_tokens
        total_hit += d.hit_tokens
        total_out += d.output_tokens
        total_extra_req += d.extra_requests
        total_extra_cost += d.extra_cost

        short_day = day_key[5:] if len(day_key) == 10 else day_key
        lines.append(
            f"{short_day} | ¥{d.total_cost:.2f} "
            f"(峰¥{d.peak_cost:.2f}/闲¥{d.off_cost:.2f}) | "
            f"{d.requests}次 | 命中{d.hit_rate:.0f}%"
        )

    n_days = len(days_map)
    avg_cost = total_cost / n_days
    total_in = total_miss + total_hit
    total_hit_rate = (total_hit / total_in * 100.0) if total_in > 0 else 0.0

    lines.append("")
    lines.append(
        f"合计({n_days}天)：¥{total_cost:.2f} | 日均 ¥{avg_cost:.2f} | "
        f"{total_req}次 | 总命中 {total_hit_rate:.1f}%"
    )
    lines.append(
        f"Token：未命中 {total_miss:,} / 命中 {total_hit:,} / 输出 {total_out:,}"
    )
    if total_extra_req > 0:
        lines.append(f"其中后台插件直连补记：{total_extra_req}次 / ¥{total_extra_cost:.2f}")
    if note:
        lines.append(note)
    return "\n".join(lines)


def format_reconcile(db_days: dict[str, DayCost], csv_days: dict[str, DayCost]) -> str:
    """对比本地 provider_stats 与官方 CSV 账单。"""
    all_days = sorted(set(db_days.keys()) & set(csv_days.keys()), reverse=True)
    if not all_days:
        return "官方账单与本地数据库没有重叠的日期，无法对账。"

    lines = ["官方账单 vs 本地估算对账：", ""]
    sum_csv = 0.0
    sum_db = 0.0
    for day_key in all_days:
        c = csv_days[day_key]
        b = db_days[day_key]
        diff = c.total_cost - b.total_cost
        sum_csv += c.total_cost
        sum_db += b.total_cost
        short_day = day_key[5:]
        lines.append(
            f"{short_day} | 官方¥{c.total_cost:.2f} ({c.requests}次) "
            f"vs 本地¥{b.total_cost:.2f} ({b.requests}次) | 差额 {diff:+.2f}"
        )

    lines.append("")
    lines.append(
        f"重叠{len(all_days)}天合计：官方 ¥{sum_csv:.2f} vs 本地 ¥{sum_db:.2f} "
        f"(差额 {sum_csv - sum_db:+.2f})"
    )
    lines.append("注：差额通常来自插件直接调 provider.text_chat() 未入 provider_stats 的调用。")
    return "\n".join(lines)
