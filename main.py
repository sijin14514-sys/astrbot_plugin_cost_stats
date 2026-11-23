"""AstrBot 每日耗费统计插件（astrbot_plugin_cost_stats）。

1. 实时读取 ``data_v4.db`` 的 ``provider_stats`` 表（主对话管线）；
2. 自动拦截未进 ``provider_stats`` 的后台插件直连 ``Provider.text_chat`` 调用，
   记录到 ``extra_usage.jsonl``，合并后按 DeepSeek 官方高峰/空闲两档价精确计算；
3. 支持查询 DeepSeek 官方实时余额（``/user/balance``）及核对官方导出的 CSV/ZIP 账单。
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic.dataclasses import dataclass

from astrbot.api import AstrBotConfig, FunctionTool, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star, register

from .calculator import (
    CST,
    calc_from_csv,
    calc_from_db,
    fetch_deepseek_balance,
    format_reconcile,
    format_report,
)

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path

    _DATA_ROOT = Path(get_astrbot_data_path())
except Exception:  # noqa: BLE001
    _DATA_ROOT = Path(os.environ.get("ASTRBOT_DATA", "/root/astrbot-data/data"))

_PLUGIN_NAME = Path(__file__).parent.name
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# 防止插件重载时重复包裹 Provider.text_chat
_ORIG_TEXT_CHAT: Callable[..., Awaitable[Any]] | None = None
_ACTIVE_EXTRA_FILE: Path | None = None


def _called_from_internal_pipeline() -> bool:
    """判断当前 text_chat 是否由 AstrBot 主对话管线发起（主对话管线会自行写 provider_stats）。"""
    frame = sys._getframe(1)
    depth = 0
    while frame is not None and depth < 25:
        filename = frame.f_code.co_filename.replace("\\", "/")
        if "agent_sub_stages/internal.py" in filename:
            return True
        frame = frame.f_back
        depth += 1
    return False


def _extract_usage(resp: Any) -> tuple[int, int, int] | None:
    """从 LLMResponse 中提取 (miss, hit, output) token 数。"""
    usage = getattr(resp, "usage", None)
    if usage is not None:
        miss = int(getattr(usage, "input_other", 0) or 0)
        hit = int(getattr(usage, "input_cached", 0) or 0)
        out = int(getattr(usage, "output", 0) or 0)
        if miss or hit or out:
            return miss, hit, out

    raw = getattr(resp, "raw_completion", None)
    raw_usage = getattr(raw, "usage", None)
    if raw_usage is not None:
        prompt_tokens = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
        hit = int(getattr(raw_usage, "prompt_cache_hit_tokens", 0) or 0)
        if not hit:
            details = getattr(raw_usage, "prompt_tokens_details", None)
            hit = int(getattr(details, "cached_tokens", 0) or 0)
        miss = int(getattr(raw_usage, "prompt_cache_miss_tokens", 0) or max(0, prompt_tokens - hit))
        out = int(getattr(raw_usage, "completion_tokens", 0) or 0)
        if miss or hit or out:
            return miss, hit, out
    return None


def _install_text_chat_hook(extra_file: Path) -> None:
    """挂钩 Provider.text_chat，补记所有绕过主对话管线的直连调用。"""
    global _ORIG_TEXT_CHAT, _ACTIVE_EXTRA_FILE
    _ACTIVE_EXTRA_FILE = extra_file
    try:
        from astrbot.core.provider.provider import Provider
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[cost_stats] 无法导入 Provider，跳过直连补记钩子: {exc}")
        return

    current = getattr(Provider, "text_chat", None)
    if current is None:
        return
    if getattr(current, "_cost_stats_hooked", False):
        return

    _ORIG_TEXT_CHAT = current

    @functools.wraps(current)
    async def wrapped_text_chat(self: Any, *args: Any, **kwargs: Any) -> Any:
        is_internal = _called_from_internal_pipeline()
        resp = await current(self, *args, **kwargs)
        if not is_internal and _ACTIVE_EXTRA_FILE is not None:
            tokens = _extract_usage(resp)
            if tokens is not None:
                miss, hit, out = tokens
                rec = {
                    "ts": datetime.now(CST).isoformat(timespec="seconds"),
                    "miss": miss,
                    "hit": hit,
                    "out": out,
                }
                try:
                    with _ACTIVE_EXTRA_FILE.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[cost_stats] 写入直连补记失败: {exc}")
        return resp

    setattr(wrapped_text_chat, "_cost_stats_hooked", True)
    Provider.text_chat = wrapped_text_chat  # type: ignore[method-assign]
    logger.info("[cost_stats] 已启用后台插件直连调用补记钩子 (Provider.text_chat)")


def _uninstall_text_chat_hook() -> None:
    """卸载 Provider.text_chat 钩子。"""
    global _ORIG_TEXT_CHAT, _ACTIVE_EXTRA_FILE
    _ACTIVE_EXTRA_FILE = None
    if _ORIG_TEXT_CHAT is None:
        return
    try:
        from astrbot.core.provider.provider import Provider

        if getattr(Provider.text_chat, "_cost_stats_hooked", False):
            Provider.text_chat = _ORIG_TEXT_CHAT  # type: ignore[method-assign]
    except Exception:  # noqa: BLE001
        pass
    _ORIG_TEXT_CHAT = None


@dataclass
class CostTool(FunctionTool):
    """绑到插件协程上的 FunctionTool。"""

    impl: Callable[..., Awaitable[Any]] | None = None

    async def call(self, context: Any, **kwargs: Any) -> Any:
        event: AstrMessageEvent = context.context.event
        allowed = self.parameters.get("properties", {})
        args = {k: v for k, v in kwargs.items() if k in allowed}
        return await self.impl(event, **args)


@register(
    "astrbot_plugin_cost_stats",
    "sixiaomiao-qq",
    "每日耗费统计（DeepSeek 高峰/空闲两档计费 + 后台直连补记 + 实时余额 + 官方账单核对）",
    "v0.2.0",
)
class CostStatsPlugin(Star):
    """每日耗费统计插件。"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        self.data_root = _DATA_ROOT
        self.db_path = self.data_root / "data_v4.db"
        self.cmd_config_path = self.data_root / "cmd_config.json"
        self.plugin_data = self.data_root / "plugin_data" / _PLUGIN_NAME
        self.bills_dir = self.plugin_data / "bills"
        self.extra_file = self.plugin_data / "extra_usage.jsonl"
        self.state_file = self.plugin_data / "state.json"
        self.bills_dir.mkdir(parents=True, exist_ok=True)

        self.owner_qq = str(self.config.get("owner_qq", "") or "").strip()
        self.private_only = bool(self.config.get("private_only", True))
        self.default_days = max(1, min(90, int(self.config.get("default_days", 7) or 7)))
        self.daily_report_enable = bool(self.config.get("daily_report_enable", False))
        self.daily_report_time = str(self.config.get("daily_report_time", "23:50") or "23:50").strip()
        self.daily_report_umo = str(self.config.get("daily_report_umo", "") or "").strip()
        self.enable_llm_tool = bool(self.config.get("enable_llm_tool", True))
        self.track_extra_calls = bool(self.config.get("track_extra_calls", True))

        self.pricing = {
            "peak": {
                "hit": float(self.config.get("peak_hit_price", 0.04) or 0.04),
                "miss": float(self.config.get("peak_miss_price", 2.0) or 2.0),
                "output": float(self.config.get("peak_output_price", 8.0) or 8.0),
            },
            "off": {
                "hit": float(self.config.get("off_hit_price", 0.02) or 0.02),
                "miss": float(self.config.get("off_miss_price", 1.0) or 1.0),
                "output": float(self.config.get("off_output_price", 4.0) or 4.0),
            },
        }

        if self.track_extra_calls:
            _install_text_chat_hook(self.extra_file)

        self._report_task: asyncio.Task | None = None
        if self.daily_report_enable and self.daily_report_umo:
            self._report_task = asyncio.create_task(self._daily_report_loop())

        if self.enable_llm_tool:
            self.context.add_llm_tools(
                CostTool(
                    name="query_daily_cost",
                    description=(
                        "查询你（肆小喵）近几天或指定日期的模型API花费、调用次数、缓存命中率以及 DeepSeek 账户实时余额。"
                        "仅在主人私聊询问花费/账单/花了多少钱/还剩多少钱时调用。"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "days": {
                                "type": "integer",
                                "description": "统计最近几天（1~30，默认 7，查今天传 1）",
                            },
                            "date": {
                                "type": "string",
                                "description": "指定日期 YYYY-MM-DD（可选；填写后优先查该日）",
                            },
                        },
                    },
                    impl=self._tool_query_cost,
                )
            )

    async def terminate(self) -> None:
        _uninstall_text_chat_hook()
        if self._report_task and not self._report_task.done():
            self._report_task.cancel()
            try:
                await self._report_task
            except asyncio.CancelledError:
                pass

    # ---------- 权限、余额与状态 ----------

    def _is_owner(self, event: AstrMessageEvent) -> bool:
        if self.owner_qq:
            return str(event.get_sender_id()) == self.owner_qq
        return bool(event.is_admin())

    def _is_private(self, event: AstrMessageEvent) -> bool:
        return not bool(event.get_group_id())

    def _check_access(self, event: AstrMessageEvent) -> str | None:
        if not self._is_owner(event):
            return "这个只有主人能看喵。"
        if self.private_only and not self._is_private(event):
            return "账单在私聊里看喵，群里不方便。"
        return None

    def _find_deepseek_key(self) -> str:
        try:
            cfg = json.loads(self.cmd_config_path.read_text(encoding="utf-8"))
            for ps in cfg.get("provider_sources") or []:
                base = str(ps.get("api_base", "")).lower()
                pid = str(ps.get("id", "")).lower()
                if "deepseek" in base or "deepseek" in pid:
                    keys = ps.get("key") or []
                    if keys:
                        return str(keys[0]).strip()
        except Exception:  # noqa: BLE001
            pass
        return ""

    async def _get_balance_line(self) -> str:
        key = self._find_deepseek_key()
        if not key:
            return ""
        try:
            bal = await asyncio.to_thread(fetch_deepseek_balance, key)
            return f"当前 DeepSeek 账户余额：¥{bal['total']}（充值 ¥{bal['topped_up']} / 赠送 ¥{bal['granted']}）"
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[cost_stats] 查询余额失败: {exc}")
            return ""

    def _last_report_date(self) -> str:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return str(data.get("last_report_date", ""))
        except Exception:  # noqa: BLE001
            return ""

    def _save_report_date(self, day_str: str) -> None:
        try:
            self.state_file.write_text(
                json.dumps({"last_report_date": day_str}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[cost_stats] 保存日报状态失败: {exc}")

    # ---------- 定时日报 ----------

    async def _daily_report_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(45)
                now = datetime.now(CST)
                hhmm = now.strftime("%H:%M")
                today = now.strftime("%Y-%m-%d")
                if hhmm < self.daily_report_time:
                    continue
                if self._last_report_date() == today:
                    continue

                days_map = calc_from_db(
                    self.db_path,
                    target_date=today,
                    pricing=self.pricing,
                    now_cst=now,
                    extra_jsonl=self.extra_file,
                )
                bal_line = await self._get_balance_line()
                msg = format_report(
                    days_map,
                    title=f"肆小喵今日耗费日报（{today}）",
                    note=bal_line,
                )
                chain = MessageChain([Plain(msg)])
                await self.context.send_message(self.daily_report_umo, chain)
                self._save_report_date(today)
                logger.info(f"[cost_stats] 已发送 {today} 耗费日报")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[cost_stats] 定时日报异常: {exc}")
                await asyncio.sleep(60)

    # ---------- 命令 ----------

    @filter.command("余额")
    async def cmd_balance(self, event: AstrMessageEvent):
        """查询 DeepSeek 账户实时余额。"""
        deny = self._check_access(event)
        if deny:
            yield event.plain_result(deny)
            return
        key = self._find_deepseek_key()
        if not key:
            yield event.plain_result("没找到 DeepSeek API Key，查不到余额喵。")
            return
        try:
            bal = await asyncio.to_thread(fetch_deepseek_balance, key)
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"查余额失败：{exc}")
            return
        yield event.plain_result(
            f"DeepSeek 账户实时余额：¥{bal['total']}\n"
            f"（充值余额 ¥{bal['topped_up']} / 赠送余额 ¥{bal['granted']}）"
        )

    @filter.command("今日耗费")
    async def cmd_today_cost(self, event: AstrMessageEvent):
        """查看今天的耗费明细。"""
        deny = self._check_access(event)
        if deny:
            yield event.plain_result(deny)
            return
        now = datetime.now(CST)
        today = now.strftime("%Y-%m-%d")
        try:
            days_map = calc_from_db(
                self.db_path,
                target_date=today,
                pricing=self.pricing,
                now_cst=now,
                extra_jsonl=self.extra_file,
            )
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"统计失败：{exc}")
            return
        bal_line = await self._get_balance_line()
        yield event.plain_result(
            format_report(
                days_map,
                title=f"肆小喵今日耗费（{today}）",
                note=bal_line,
            )
        )

    @filter.command("耗费")
    async def cmd_cost(self, event: AstrMessageEvent):
        """查看每日耗费，支持 /耗费 [天数|今天|昨天|YYYY-MM-DD]。"""
        deny = self._check_access(event)
        if deny:
            yield event.plain_result(deny)
            return

        raw = str(event.message_str or "").strip()
        parts = raw.split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else ""

        now = datetime.now(CST)
        try:
            if arg in ("今天", "今日"):
                today = now.strftime("%Y-%m-%d")
                days_map = calc_from_db(
                    self.db_path,
                    target_date=today,
                    pricing=self.pricing,
                    now_cst=now,
                    extra_jsonl=self.extra_file,
                )
                title = f"肆小喵今日耗费（{today}）"
            elif arg in ("昨天", "昨日"):
                yday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
                days_map = calc_from_db(
                    self.db_path,
                    target_date=yday,
                    pricing=self.pricing,
                    now_cst=now,
                    extra_jsonl=self.extra_file,
                )
                title = f"肆小喵昨日耗费（{yday}）"
            elif _DATE_RE.match(arg):
                days_map = calc_from_db(
                    self.db_path,
                    target_date=arg,
                    pricing=self.pricing,
                    now_cst=now,
                    extra_jsonl=self.extra_file,
                )
                title = f"肆小喵耗费明细（{arg}）"
            else:
                days = self.default_days
                if arg:
                    if not arg.isdigit():
                        yield event.plain_result(
                            "用法：/耗费 [天数|今天|昨天|YYYY-MM-DD]，例如 /耗费 7"
                        )
                        return
                    days = max(1, min(90, int(arg)))
                days_map = calc_from_db(
                    self.db_path,
                    days=days,
                    pricing=self.pricing,
                    now_cst=now,
                    extra_jsonl=self.extra_file,
                )
                title = f"肆小喵近 {days} 天耗费"
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"统计失败：{exc}")
            return

        bal_line = await self._get_balance_line()
        yield event.plain_result(
            format_report(
                days_map,
                title=title,
                note=bal_line,
            )
        )

    @filter.command("账单核对")
    async def cmd_reconcile(self, event: AstrMessageEvent):
        """读取 bills/ 下最新官方 CSV/ZIP 账单并与本地统计核对。"""
        deny = self._check_access(event)
        if deny:
            yield event.plain_result(deny)
            return

        try:
            csv_days = calc_from_csv(self.bills_dir)
        except FileNotFoundError:
            yield event.plain_result(
                f"还没有官方账单文件。\n请把 DeepSeek 导出的 CSV 或 ZIP 放到：\n{self.bills_dir}"
            )
            return
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"解析官方账单失败：{exc}")
            return

        bill_text = format_report(csv_days, title="DeepSeek 官方账单汇总")

        if self.db_path.exists() and csv_days:
            min_day = min(csv_days.keys())
            max_day = max(csv_days.keys())
            span_days = (
                datetime.strptime(max_day, "%Y-%m-%d")
                - datetime.strptime(min_day, "%Y-%m-%d")
            ).days + 2
            try:
                now = datetime.now(CST)
                db_days = calc_from_db(
                    self.db_path,
                    days=max(span_days, 14),
                    pricing=self.pricing,
                    now_cst=now,
                    extra_jsonl=self.extra_file,
                )
                reconcile_text = format_reconcile(db_days, csv_days)
                yield event.plain_result(f"{bill_text}\n\n{reconcile_text}")
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[cost_stats] 对账读取 DB 失败: {exc}")

        yield event.plain_result(bill_text)

    # ---------- LLM 工具（用肆小喵自己的语气向主人汇报） ----------

    async def _tool_query_cost(
        self,
        event: AstrMessageEvent,
        days: int = 7,
        date: str = "",
    ) -> str:
        deny = self._check_access(event)
        if deny:
            return f"拒绝查询：{deny}"

        now = datetime.now(CST)
        date_str = str(date or "").strip()
        try:
            if date_str and _DATE_RE.match(date_str):
                days_map = calc_from_db(
                    self.db_path,
                    target_date=date_str,
                    pricing=self.pricing,
                    now_cst=now,
                    extra_jsonl=self.extra_file,
                )
                title = f"{date_str} 耗费"
            else:
                n = max(1, min(30, int(days or 7)))
                days_map = calc_from_db(
                    self.db_path,
                    days=n,
                    pricing=self.pricing,
                    now_cst=now,
                    extra_jsonl=self.extra_file,
                )
                title = f"近 {n} 天耗费"
        except Exception as exc:  # noqa: BLE001
            return f"查账失败：{exc}"

        bal_line = await self._get_balance_line()
        note = f"{bal_line}\n请用你自己的语气简短告诉主人，别用 markdown。" if bal_line else "请用你自己的语气简短告诉主人，别用 markdown。"
        return format_report(days_map, title=title, note=note)
