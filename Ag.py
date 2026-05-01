from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape, unescape
from io import StringIO
import re
from typing import Iterable
from urllib.parse import quote

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


APP_NAME = "Silver-Macro-Kalman-Terminal"
SILVER_TICKER = "SI=F"
DXY_TICKER = "DX-Y.NYB"
TNX_TICKER = "^TNX"
FRED_REAL_RATE = "DFII10"
FRED_DOLLAR_INDEX = "DTWEXBGS"
FRED_TREASURY_10Y = "DGS10"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TRANSLATION_URL = "https://api.mymemory.translated.net/get"
GOOGLE_TRANSLATION_URL = "https://translate.googleapis.com/translate_a/single"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
DATA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 Silver-Macro-Kalman-Terminal "
        "(local research dashboard; contact: local)"
    )
}
INTRADAY_MAX_PERIOD = "2y"
LIVE_QUOTE_REFRESH_SECONDS = 2
HOME_NEWS_PAGE_SIZE = 10
NEWS_FETCH_ITEMS_PER_SOURCE = 160
YAHOO_VALUE_MULTIPLIERS: dict[str, float] = {}

SENTIMENT_WEIGHT = 0.40
DOLLAR_WEIGHT = 0.35
RATE_WEIGHT = 0.25

NEWS_KEYWORDS = (
    "silver",
    "fed",
    "federal reserve",
    "war",
    "industrial demand",
    "solar",
    "photovoltaic",
    "electronics",
    "semiconductor",
    "electric vehicle",
    "battery",
    "supply deficit",
)
NEWS_KEYWORD_PATTERNS = {
    "silver": re.compile(r"\bsilver\b", re.IGNORECASE),
    "fed": re.compile(r"\bfed\b", re.IGNORECASE),
    "federal reserve": re.compile(r"\bfederal\s+reserve\b", re.IGNORECASE),
    "war": re.compile(r"\bwars?\b", re.IGNORECASE),
    "industrial demand": re.compile(r"\bindustrial\s+demand\b", re.IGNORECASE),
    "solar": re.compile(r"\bsolar\b", re.IGNORECASE),
    "photovoltaic": re.compile(r"\bphotovoltaics?\b|\bPV\b", re.IGNORECASE),
    "electronics": re.compile(r"\belectronics?\b|\belectrical\b", re.IGNORECASE),
    "semiconductor": re.compile(r"\bsemiconductors?\b|\bchips?\b", re.IGNORECASE),
    "electric vehicle": re.compile(r"\belectric\s+vehicles?\b|\bEVs?\b", re.IGNORECASE),
    "battery": re.compile(r"\bbatter(?:y|ies)\b", re.IGNORECASE),
    "supply deficit": re.compile(r"\bsupply\s+deficits?\b|\bsilver\s+shortage\b", re.IGNORECASE),
}
INDUSTRIAL_NEWS_KEYWORDS = {
    "industrial demand",
    "solar",
    "photovoltaic",
    "electronics",
    "semiconductor",
    "electric vehicle",
    "battery",
    "supply deficit",
}
NON_GEOPOLITICAL_WAR_TERMS = (
    "talent war",
    "price war",
    "bidding war",
    "streaming war",
    "console war",
    "culture war",
)
RSS_FEEDS = {
    "CNBC Markets": "https://www.cnbc.com/id/15839135/device/rss/rss.html",
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Google News Query": (
        "https://news.google.com/rss/search?"
        "q=silver%20OR%20Fed%20OR%20war%20when:7d&hl=en-US&gl=US&ceid=US:en"
    ),
    "Google News Silver Industry": (
        "https://news.google.com/rss/search?"
        f"q={quote('(silver industrial demand) OR (silver solar) OR (photovoltaic silver) OR (silver electronics) OR (silver supply deficit) when:14d')}"
        "&hl=en-US&gl=US&ceid=US:en"
    ),
}


@dataclass
class FetchResult:
    data: pd.DataFrame
    warnings: list[str]


@dataclass
class QuoteResult:
    price: float | None
    timestamp: datetime | None
    source: str
    warning: str | None = None


class MacroKalman2D:
    """Two-dimensional Kalman filter with price and hidden velocity states."""

    def __init__(
        self,
        initial_price: float,
        rho: float = 0.86,
        alpha: float = 0.25,
        q: float = 0.08,
        r: float = 2.5,
    ) -> None:
        self.x = np.array([[float(initial_price)], [0.0]], dtype=float)
        self.p = np.eye(2, dtype=float)
        self.rho = float(np.clip(rho, 0.0, 1.0))
        self.alpha = float(alpha)
        self.q = max(float(q), 1e-9)
        self.r = max(float(r), 1e-9)

        self.f = np.array([[1.0, 1.0], [0.0, self.rho]], dtype=float)
        self.b = np.array([[0.5 * self.alpha], [self.alpha]], dtype=float)
        self.h = np.array([[1.0, 0.0]], dtype=float)
        self.q_matrix = np.array([[self.q, 0.0], [0.0, self.q]], dtype=float)
        self.i = np.eye(2, dtype=float)

    def predict(self, u_score: float = 0.0) -> None:
        u = 0.0 if pd.isna(u_score) else float(u_score)
        self.x = self.f @ self.x + self.b * u
        self.p = self.f @ self.p @ self.f.T + self.q_matrix

    def update(self, observed_price: float | None, vol_ratio: float = 1.0) -> None:
        if observed_price is None or pd.isna(observed_price):
            return

        safe_ratio = 1.0 if pd.isna(vol_ratio) else max(float(vol_ratio), 0.05)
        r_dynamic = self.r * safe_ratio
        r_matrix = np.array([[r_dynamic]], dtype=float)
        z = np.array([[float(observed_price)]], dtype=float)
        residual = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + r_matrix
        k = self.p @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + k @ residual
        self.p = (self.i - k @ self.h) @ self.p

    def step(
        self,
        observed_price: float | None,
        u_score: float = 0.0,
        vol_ratio: float = 1.0,
    ) -> tuple[float, float]:
        self.predict(u_score)
        self.update(observed_price, vol_ratio=vol_ratio)
        return float(self.x[0, 0]), float(self.x[1, 0])


def normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = frame.copy()
    result.index = pd.to_datetime(result.index)
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    minimum_periods = max(10, min(window, 20))
    mean = values.ewm(span=window, min_periods=minimum_periods, adjust=False).mean()
    std = values.ewm(span=window, min_periods=minimum_periods, adjust=False).std(bias=False)
    z = (values - mean) / std.replace(0.0, np.nan)

    if z.isna().all():
        fallback_std = values.std(ddof=0)
        if pd.notna(fallback_std) and fallback_std > 0:
            z = (values - values.mean()) / fallback_std

    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-4.0, 4.0)


def first_available_column(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(index=frame.index, dtype=float)


def match_news_keywords(text: str) -> list[str]:
    return [keyword for keyword in NEWS_KEYWORDS if NEWS_KEYWORD_PATTERNS[keyword].search(text)]


def classify_news_category(matched_keywords: Iterable[str], text: str) -> str:
    matched = {keyword.lower() for keyword in matched_keywords}
    if matched & INDUSTRIAL_NEWS_KEYWORDS:
        return "工业相关"

    industrial_terms = (
        "industrial",
        "solar",
        "photovoltaic",
        "electronics",
        "semiconductor",
        "chip",
        "electric vehicle",
        "ev",
        "battery",
        "grid",
        "electrification",
        "supply deficit",
        "silver shortage",
    )
    if any(term in text for term in industrial_terms):
        return "工业相关"

    return "宏观/其他"


def has_geopolitical_conflict_text(text: str) -> bool:
    if any(term in text for term in NON_GEOPOLITICAL_WAR_TERMS):
        return False

    conflict_patterns = (
        NEWS_KEYWORD_PATTERNS["war"],
        re.compile(r"\battack\b", re.IGNORECASE),
        re.compile(r"\bconflicts?\b", re.IGNORECASE),
        re.compile(r"\bgeopolitical\b", re.IGNORECASE),
        re.compile(r"\bsanctions?\b", re.IGNORECASE),
        re.compile(r"\btensions?\b", re.IGNORECASE),
    )
    return any(pattern.search(text) for pattern in conflict_patterns)


def normalize_yahoo_period(period: str, interval: str) -> tuple[str, str | None]:
    if interval != "1d" and period in {"5y", "10y"}:
        return INTRADAY_MAX_PERIOD, f"Yahoo Chart 小时线最多稳定请求约 {INTRADAY_MAX_PERIOD}，已将 {period} 小时线窗口改为 {INTRADAY_MAX_PERIOD}。"
    return period, None


def request_yahoo_chart(symbol: str, period: str, interval: str) -> tuple[pd.DataFrame, dict[str, object]]:
    request_period, _ = normalize_yahoo_period(period, interval)
    response = requests.get(
        YAHOO_CHART_URL.format(symbol=quote(symbol, safe="")),
        params={
            "range": request_period,
            "interval": interval,
            "includePrePost": "false",
            "events": "history",
        },
        headers=DATA_HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        description = error.get("description") if isinstance(error, dict) else str(error)
        raise RuntimeError(description)

    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo Chart 未返回 result。")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
    if not timestamps or not quote_data:
        raise RuntimeError("Yahoo Chart 未返回可解析的时间序列。")

    index = pd.to_datetime(timestamps, unit="s", utc=True)
    timezone = str(result.get("meta", {}).get("exchangeTimezoneName") or "UTC")
    try:
        index = index.tz_convert(timezone).tz_localize(None)
    except Exception:
        index = index.tz_localize(None)

    data = pd.DataFrame(
        {
            "Open": quote_data.get("open", []),
            "High": quote_data.get("high", []),
            "Low": quote_data.get("low", []),
            "Close": quote_data.get("close", []),
            "Volume": quote_data.get("volume", []),
        },
        index=index,
    )
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    multiplier = YAHOO_VALUE_MULTIPLIERS.get(symbol, 1.0)
    if multiplier != 1.0:
        for column in ["Open", "High", "Low", "Close"]:
            data[column] = data[column] * multiplier

    data = normalize_index(data).dropna(subset=["Close"])
    return data, dict(result.get("meta", {}))


def fetch_yahoo_close_frame(symbol: str, column_name: str, period: str, interval: str) -> FetchResult:
    warnings: list[str] = []
    _, period_warning = normalize_yahoo_period(period, interval)
    if period_warning:
        warnings.append(period_warning)

    try:
        data, _ = request_yahoo_chart(symbol, period, interval)
    except Exception as exc:
        return FetchResult(pd.DataFrame(), warnings + [f"Yahoo Chart 获取 {symbol} 失败：{exc}"])

    if data.empty or data["Close"].dropna().empty:
        return FetchResult(pd.DataFrame(), warnings + [f"Yahoo Chart 未返回 {symbol} 可用收盘价。"])

    return FetchResult(data[["Close"]].rename(columns={"Close": column_name}), warnings)


def fetch_fred_series(
    series_id: str,
    start: datetime,
    end: datetime,
    column_name: str | None = None,
) -> FetchResult:
    output_name = column_name or series_id
    params = {
        "id": series_id,
        "cosd": start.strftime("%Y-%m-%d"),
        "coed": end.strftime("%Y-%m-%d"),
    }

    try:
        response = requests.get(FRED_CSV_URL, params=params, headers=DATA_HEADERS, timeout=15)
        response.raise_for_status()
        data = pd.read_csv(StringIO(response.text), na_values=["."])
    except Exception as exc:
        return FetchResult(pd.DataFrame(), [f"FRED {series_id} 下载失败：{exc}"])

    date_column = "DATE" if "DATE" in data.columns else "observation_date"
    if data.empty or date_column not in data.columns or series_id not in data.columns:
        return FetchResult(pd.DataFrame(), [f"FRED {series_id} 未返回可解析数据。"])

    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[series_id] = pd.to_numeric(data[series_id], errors="coerce")
    data = data.dropna(subset=[date_column]).set_index(date_column)[[series_id]]
    data = normalize_index(data.rename(columns={series_id: output_name}))
    if data[output_name].dropna().empty:
        return FetchResult(pd.DataFrame(), [f"FRED {series_id} 数据为空。"])

    return FetchResult(data, [])


def align_to_index(frame: pd.DataFrame, column: str, index: pd.Index) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(index=index, dtype=float)

    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return pd.Series(index=index, dtype=float)

    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.reindex(index, method="ffill")


def dedupe_warnings(warnings: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        if warning and warning not in seen:
            result.append(warning)
            seen.add(warning)
    return result


def yahoo_market_timestamp(meta: dict[str, object]) -> datetime | None:
    timestamp = meta.get("regularMarketTime")
    if timestamp is None or pd.isna(timestamp):
        return None

    try:
        return pd.Timestamp(float(timestamp), unit="s", utc=True).tz_convert("Asia/Shanghai").to_pydatetime()
    except Exception:
        return None


def current_timestamp() -> datetime:
    return pd.Timestamp.now(tz="Asia/Shanghai").to_pydatetime()


def format_quote_timestamp(value: datetime | None) -> str:
    if value is None or pd.isna(value):
        return "暂无"

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Shanghai")
    else:
        timestamp = timestamp.tz_convert("Asia/Shanghai")
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def fetch_yfinance_market_data(period: str, interval: str) -> FetchResult:
    warnings: list[str] = []
    tickers = [SILVER_TICKER, DXY_TICKER, TNX_TICKER]

    try:
        raw = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
            timeout=20,
        )
    except Exception as exc:
        return FetchResult(pd.DataFrame(), [f"yfinance 备用行情下载失败：{exc}"])

    if raw.empty:
        return FetchResult(pd.DataFrame(), ["yfinance 备用行情未返回可用数据。"])

    try:
        if isinstance(raw.columns, pd.MultiIndex):
            field = "Close" if "Close" in raw.columns.get_level_values(0) else "Adj Close"
            close = raw[field].copy()
        else:
            close = raw[["Close"]].rename(columns={"Close": SILVER_TICKER})
    except Exception as exc:
        return FetchResult(pd.DataFrame(), [f"无法解析 yfinance 备用收盘价数据：{exc}"])

    close = normalize_index(close)
    for ticker in tickers:
        if ticker not in close.columns or close[ticker].dropna().empty:
            warnings.append(f"yfinance 备用源缺失或返回空数据：{ticker}")

    return FetchResult(close, warnings)


def fetch_yfinance_silver_ohlc(period: str, interval: str) -> FetchResult:
    try:
        raw = yf.download(
            tickers=SILVER_TICKER,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
            timeout=20,
        )
    except Exception as exc:
        return FetchResult(pd.DataFrame(), [f"yfinance 备用 SI=F OHLC 下载失败：{exc}"])

    if raw.empty:
        return FetchResult(pd.DataFrame(), ["yfinance 备用 SI=F OHLC 为空。"])

    if isinstance(raw.columns, pd.MultiIndex):
        ohlc = raw.droplevel(1, axis=1).copy()
    else:
        ohlc = raw.copy()

    required = ["Open", "High", "Low", "Close"]
    missing = [name for name in required if name not in ohlc.columns]
    if missing:
        return FetchResult(pd.DataFrame(), [f"yfinance 备用 SI=F OHLC 缺少字段：{', '.join(missing)}"])

    cleaned = normalize_index(ohlc[required].copy()).dropna(subset=["Close"])
    return FetchResult(cleaned, [])


def fetch_market_data(period: str, interval: str) -> FetchResult:
    warnings: list[str] = []
    silver = fetch_yahoo_close_frame(SILVER_TICKER, SILVER_TICKER, period, interval)
    warnings.extend(silver.warnings)

    if silver.data.empty:
        fallback = fetch_yfinance_market_data(period, interval)
        fallback.warnings = dedupe_warnings(
            silver.warnings + ["Yahoo Chart 主行情源不可用，已尝试 yfinance 备用源。"] + fallback.warnings
        )
        return fallback

    result = pd.DataFrame(index=silver.data.index)
    result[SILVER_TICKER] = pd.to_numeric(silver.data[SILVER_TICKER], errors="coerce")
    start = result.index.min().to_pydatetime() - timedelta(days=10)
    end = result.index.max().to_pydatetime() + timedelta(days=1)

    if interval == "1d":
        dollar = fetch_fred_series(FRED_DOLLAR_INDEX, start, end, DXY_TICKER)
        rate = fetch_fred_series(FRED_TREASURY_10Y, start, end, TNX_TICKER)

        if dollar.data.empty:
            warnings.extend(dollar.warnings)
            dollar = fetch_yahoo_close_frame(DXY_TICKER, DXY_TICKER, period, interval)
        if rate.data.empty:
            warnings.extend(rate.warnings)
            rate = fetch_yahoo_close_frame(TNX_TICKER, TNX_TICKER, period, interval)
    else:
        dollar = fetch_yahoo_close_frame(DXY_TICKER, DXY_TICKER, period, interval)
        rate = fetch_yahoo_close_frame(TNX_TICKER, TNX_TICKER, period, interval)

        if dollar.data.empty:
            warnings.extend(dollar.warnings)
            dollar = fetch_fred_series(FRED_DOLLAR_INDEX, start, end, DXY_TICKER)
        if rate.data.empty:
            warnings.extend(rate.warnings)
            rate = fetch_fred_series(FRED_TREASURY_10Y, start, end, TNX_TICKER)

    warnings.extend(dollar.warnings)
    warnings.extend(rate.warnings)
    result[DXY_TICKER] = align_to_index(dollar.data, DXY_TICKER, result.index)
    result[TNX_TICKER] = align_to_index(rate.data, TNX_TICKER, result.index)
    result = normalize_index(result)

    for ticker in [SILVER_TICKER, DXY_TICKER, TNX_TICKER]:
        if ticker not in result.columns or result[ticker].dropna().empty:
            warnings.append(f"行情源缺失或返回空数据：{ticker}")

    if result[SILVER_TICKER].dropna().empty:
        fallback = fetch_yfinance_market_data(period, interval)
        fallback.warnings = dedupe_warnings(
            warnings + ["主行情源清洗后没有可用白银价格，已尝试 yfinance 备用源。"] + fallback.warnings
        )
        return fallback

    return FetchResult(result, dedupe_warnings(warnings))


def fetch_silver_ohlc(period: str, interval: str) -> FetchResult:
    warnings: list[str] = []
    _, period_warning = normalize_yahoo_period(period, interval)
    if period_warning:
        warnings.append(period_warning)

    try:
        data, _ = request_yahoo_chart(SILVER_TICKER, period, interval)
    except Exception as exc:
        fallback = fetch_yfinance_silver_ohlc(period, interval)
        fallback.warnings = dedupe_warnings(
            warnings + [f"Yahoo Chart SI=F OHLC 获取失败，已尝试 yfinance 备用源：{exc}"] + fallback.warnings
        )
        return fallback

    required = ["Open", "High", "Low", "Close"]
    cleaned = normalize_index(data[required].copy()).dropna(subset=["Close"])
    if cleaned.empty:
        fallback = fetch_yfinance_silver_ohlc(period, interval)
        fallback.warnings = dedupe_warnings(
            warnings + ["Yahoo Chart SI=F OHLC 清洗后为空，已尝试 yfinance 备用源。"] + fallback.warnings
        )
        return fallback

    return FetchResult(cleaned, dedupe_warnings(warnings))


def calculate_adx(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    if ohlc.empty:
        return pd.Series(dtype=float)

    data = ohlc.copy()
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")
    close = pd.to_numeric(data["Close"], errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    tr_components = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_dm_smoothed = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_smoothed = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * plus_dm_smoothed / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm_smoothed / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx.replace([np.inf, -np.inf], np.nan)


def fetch_latest_silver_price() -> QuoteResult:
    primary_warning: str | None = None
    try:
        intraday, meta = request_yahoo_chart(SILVER_TICKER, "1d", "1m")
        market_price = meta.get("regularMarketPrice")
        if market_price is not None and not pd.isna(market_price):
            return QuoteResult(float(market_price), yahoo_market_timestamp(meta) or current_timestamp(), "Yahoo Chart")
        if not intraday.empty and "Close" in intraday:
            latest_close = intraday["Close"].dropna()
            if not latest_close.empty:
                return QuoteResult(float(latest_close.iloc[-1]), intraday.index[-1].to_pydatetime(), "Yahoo Chart")
    except Exception as exc:
        primary_warning = f"Yahoo Chart 白银实时报价获取失败：{exc}"

    try:
        ticker = yf.Ticker(SILVER_TICKER)
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            last_price = fast_info.get("last_price")
            if last_price is not None and not pd.isna(last_price):
                return QuoteResult(float(last_price), current_timestamp(), "yfinance fast_info", primary_warning)
        intraday = ticker.history(period="1d", interval="1m", timeout=10)
        if not intraday.empty and "Close" in intraday:
            latest_close = intraday["Close"].dropna()
            if not latest_close.empty:
                timestamp = latest_close.index[-1]
                if hasattr(timestamp, "to_pydatetime"):
                    timestamp = timestamp.to_pydatetime()
                return QuoteResult(float(latest_close.iloc[-1]), timestamp, "yfinance history", primary_warning)
    except Exception as exc:
        if primary_warning:
            return QuoteResult(None, None, "不可用", f"{primary_warning}；yfinance 备用实时报价获取失败：{exc}")
        return QuoteResult(None, None, "不可用", f"白银实时报价获取失败：{exc}")

    if primary_warning:
        return QuoteResult(None, None, "不可用", f"{primary_warning}；yfinance 备用实时报价暂不可用。")
    return QuoteResult(None, None, "不可用", "白银实时报价暂不可用。")


def fetch_real_rate(start: datetime, end: datetime) -> FetchResult:
    return fetch_fred_series(FRED_REAL_RATE, start, end, FRED_REAL_RATE)


def parse_entry_date(entry: dict) -> pd.Timestamp:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return pd.Timestamp(datetime(*parsed[:6]))
        except Exception:
            pass
    return pd.Timestamp.now(tz="UTC").tz_convert(None)


def fetch_news_sentiment(max_items: int = NEWS_FETCH_ITEMS_PER_SOURCE) -> FetchResult:
    warnings: list[str] = []
    analyzer = SentimentIntensityAnalyzer()
    rows: list[dict[str, object]] = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 Silver-Macro-Kalman-Terminal "
            "(local research dashboard; contact: local)"
        )
    }

    for source, url in RSS_FEEDS.items():
        try:
            response = requests.get(url, headers=headers, timeout=8)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        except Exception as exc:
            warnings.append(f"{source} RSS 抓取失败：{exc}")
            continue

        for entry in feed.entries[:max_items]:
            title = str(entry.get("title", "")).strip()
            summary = str(entry.get("summary", "")).strip()
            text = f"{title} {summary}".lower()
            matched = match_news_keywords(text)
            if not title or not matched:
                continue
            if matched == ["war"] and any(term in text for term in NON_GEOPOLITICAL_WAR_TERMS):
                continue

            score = analyzer.polarity_scores(title)["compound"]
            rows.append(
                {
                    "published": parse_entry_date(entry),
                    "source": source,
                    "title": title,
                    "summary": summary,
                    "score": float(score),
                    "keyword": ", ".join(matched),
                    "category": classify_news_category(matched, text),
                    "link": entry.get("link", ""),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return FetchResult(frame, warnings + ["未找到匹配关键词的 RSS 新闻标题。"])

    frame["_dedupe_key"] = frame["link"].replace("", np.nan).fillna(frame["title"].str.lower().str.strip())
    frame = frame.drop_duplicates("_dedupe_key", keep="first").drop(columns=["_dedupe_key"])
    frame = frame.sort_values("published", ascending=False).reset_index(drop=True)
    return FetchResult(frame, warnings)


def daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty or "published" not in news:
        return pd.DataFrame(columns=["sentiment", "headline_count"])

    data = news.copy()
    data["date"] = pd.to_datetime(data["published"]).dt.normalize()
    grouped = data.groupby("date").agg(sentiment=("score", "mean"), headline_count=("score", "size"))
    return grouped.sort_index()


def compute_macro_score(
    closes: pd.DataFrame,
    real_rate: pd.DataFrame,
    sentiment: pd.DataFrame,
    z_window: int,
    sentiment_weight: float = SENTIMENT_WEIGHT,
    dollar_weight: float = DOLLAR_WEIGHT,
    rate_weight: float = RATE_WEIGHT,
) -> pd.DataFrame:
    macro = pd.DataFrame(index=closes.index)
    macro["silver"] = first_available_column(closes, [SILVER_TICKER])
    macro["dxy"] = first_available_column(closes, [DXY_TICKER]).ffill()
    macro["tnx"] = first_available_column(closes, [TNX_TICKER]).ffill()

    macro["dxy_delta"] = macro["dxy"].pct_change() * 100.0
    macro["tnx_delta"] = macro["tnx"].diff()

    if not real_rate.empty and FRED_REAL_RATE in real_rate.columns:
        aligned_real_rate = real_rate[[FRED_REAL_RATE]].reindex(macro.index, method="ffill")
        macro["real_rate"] = pd.to_numeric(aligned_real_rate[FRED_REAL_RATE], errors="coerce")
        macro["rate_delta"] = macro["real_rate"].diff().combine_first(macro["tnx_delta"])
    else:
        macro["real_rate"] = np.nan
        macro["rate_delta"] = macro["tnx_delta"]

    if not sentiment.empty and "sentiment" in sentiment:
        sentiment_map = sentiment["sentiment"].copy()
        sentiment_map.index = pd.to_datetime(sentiment_map.index).normalize()
        macro_dates = pd.Series(pd.to_datetime(macro.index).normalize(), index=macro.index)
        macro["sentiment_raw"] = macro_dates.map(sentiment_map).fillna(0.0)
    else:
        macro["sentiment_raw"] = 0.0

    macro["z_sentiment"] = rolling_zscore(macro["sentiment_raw"], z_window)
    macro["z_dxy"] = rolling_zscore(macro["dxy_delta"], z_window)
    macro["z_rate"] = rolling_zscore(macro["rate_delta"], z_window)
    macro["sentiment_factor"] = sentiment_weight * macro["z_sentiment"]
    macro["dxy_factor"] = -dollar_weight * macro["z_dxy"]
    macro["rate_factor"] = -rate_weight * macro["z_rate"]
    macro["U_score"] = (
        macro["sentiment_factor"] + macro["dxy_factor"] + macro["rate_factor"]
    ).clip(-5.0, 5.0)

    abs_return = macro["silver"].pct_change().abs()
    vol_fast = abs_return.rolling(window=14, min_periods=5).mean()
    vol_slow = abs_return.rolling(window=60, min_periods=20).mean()
    macro["vol_ratio"] = (
        vol_fast.divide(vol_slow.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .clip(0.30, 3.50)
        .fillna(1.0)
    )
    return macro


def run_kalman(
    price: pd.Series,
    u_score: pd.Series,
    vol_ratio: pd.Series,
    rho: float,
    alpha: float,
    q: float,
    r: float,
) -> pd.DataFrame:
    observed = pd.to_numeric(price, errors="coerce").dropna()
    if observed.empty:
        return pd.DataFrame()

    u_aligned = pd.to_numeric(u_score.reindex(observed.index), errors="coerce").fillna(0.0)
    vol_aligned = pd.to_numeric(vol_ratio.reindex(observed.index), errors="coerce").fillna(1.0)
    model = MacroKalman2D(
        initial_price=float(observed.iloc[0]),
        rho=rho,
        alpha=alpha,
        q=q,
        r=r,
    )

    rows: list[dict[str, float]] = []
    for timestamp, value in observed.items():
        estimated_price, velocity = model.step(
            value,
            u_aligned.loc[timestamp],
            vol_ratio=vol_aligned.loc[timestamp],
        )
        rows.append(
            {
                "timestamp": timestamp,
                "observed": float(value),
                "kalman_price": estimated_price,
                "velocity": velocity,
                "U_score": float(u_aligned.loc[timestamp]),
                "vol_ratio": float(vol_aligned.loc[timestamp]),
            }
        )

    result = pd.DataFrame(rows).set_index("timestamp")
    previous_velocity = result["velocity"].shift(1)
    result["buy_signal"] = (result["velocity"] > 0.0) & (previous_velocity <= 0.0)
    result["sell_signal"] = (result["velocity"] < 0.0) & (previous_velocity >= 0.0)
    return result


def build_signal_frame(
    filtered: pd.DataFrame,
    macro: pd.DataFrame,
    adx: pd.Series,
    long_threshold: float,
    short_threshold: float,
    min_velocity: float,
    stop_mult: float,
    reward_risk: float,
    adx_threshold: float = 20.0,
) -> pd.DataFrame:
    factor_columns = [
        "sentiment_factor",
        "dxy_factor",
        "rate_factor",
        "sentiment_raw",
        "z_sentiment",
        "dxy",
        "dxy_delta",
        "z_dxy",
        "real_rate",
        "tnx",
        "rate_delta",
        "z_rate",
    ]
    data = filtered.join(macro[factor_columns], how="left").copy()
    data["adx"] = pd.to_numeric(adx.reindex(data.index), errors="coerce")
    data["price_bias"] = data["observed"] - data["kalman_price"]
    data["price_bias_pct"] = data["price_bias"] / data["kalman_price"].replace(0.0, np.nan) * 100.0

    close_buffer = data["observed"].diff().abs().rolling(14, min_periods=5).mean()
    pct_buffer = data["observed"] * data["observed"].pct_change().abs().rolling(14, min_periods=5).mean()
    data["vol_buffer"] = (
        close_buffer.combine_first(pct_buffer)
        .replace([np.inf, -np.inf, 0.0], np.nan)
        .ffill()
        .bfill()
        .fillna(data["observed"] * 0.01)
    )
    data["recent_buy_cross"] = data["buy_signal"].rolling(3, min_periods=1).max().astype(bool)
    data["recent_sell_cross"] = data["sell_signal"].rolling(3, min_periods=1).max().astype(bool)

    data["long_vote_count"] = (
        (data["velocity"] > min_velocity).astype(int)
        + (data["U_score"] >= long_threshold).astype(int)
        + (data["price_bias"] >= 0.0).astype(int)
        + (data["sentiment_factor"] >= 0.0).astype(int)
        + (data["dxy_factor"] >= 0.0).astype(int)
        + (data["rate_factor"] >= 0.0).astype(int)
    )
    data["short_vote_count"] = (
        (data["velocity"] < -min_velocity).astype(int)
        + (data["U_score"] <= short_threshold).astype(int)
        + (data["price_bias"] <= 0.0).astype(int)
        + (data["sentiment_factor"] <= 0.0).astype(int)
        + (data["dxy_factor"] <= 0.0).astype(int)
        + (data["rate_factor"] <= 0.0).astype(int)
    )
    data["long_score_pct"] = data["long_vote_count"] / 6.0 * 100.0
    data["short_score_pct"] = data["short_vote_count"] / 6.0 * 100.0

    data["long_setup"] = (
        (data["velocity"] > min_velocity)
        & (data["U_score"] >= long_threshold)
        & (data["price_bias"] >= 0.0)
    )
    data["short_setup"] = (
        (data["velocity"] < -min_velocity)
        & (data["U_score"] <= short_threshold)
        & (data["price_bias"] <= 0.0)
    )
    data["adx_range_filter"] = data["adx"].notna() & (data["adx"] < float(adx_threshold))
    data.loc[data["adx_range_filter"], "long_setup"] = False
    data.loc[data["adx_range_filter"], "short_setup"] = False

    data["entry_side"] = np.select(
        [
            data["adx_range_filter"],
            data["long_setup"],
            data["short_setup"],
            data["long_vote_count"] >= 4,
            data["short_vote_count"] >= 4,
        ],
        ["震荡市观望(ADX<20)", "多头入场窗口", "空头入场窗口", "多头观察", "空头观察"],
        default="观望",
    )
    data["stop_mult"] = stop_mult
    data["reward_risk"] = reward_risk
    return data


def build_opportunity_table(signal_frame: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = signal_frame[signal_frame["long_setup"] | signal_frame["short_setup"]].tail(limit)

    for timestamp, row in candidates.iterrows():
        is_long = bool(row["long_setup"])
        side = "做多" if is_long else "做空"
        buffer = max(float(row["vol_buffer"]), float(row["observed"]) * 0.002)
        entry_low = float(row["observed"]) - buffer * 0.25
        entry_high = float(row["observed"]) + buffer * 0.25
        stop_distance = buffer * float(row["stop_mult"])
        target_distance = stop_distance * float(row["reward_risk"])
        stop = float(row["observed"]) - stop_distance if is_long else float(row["observed"]) + stop_distance
        target = float(row["observed"]) + target_distance if is_long else float(row["observed"]) - target_distance
        strength = float(row["long_score_pct"] if is_long else row["short_score_pct"])
        trigger = (
            "动量转强 / 宏观顺风 / 价格站上滤波线"
            if is_long
            else "动量转弱 / 宏观逆风 / 价格跌破滤波线"
        )

        rows.append(
            {
                "时间": timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(timestamp, "strftime") else str(timestamp),
                "方向": side,
                "信号强度": strength,
                "参考入场区间": f"{entry_low:,.2f} - {entry_high:,.2f}",
                "参考止损": stop,
                "第一目标": target,
                "U_score": float(row["U_score"]),
                "动量": float(row["velocity"]),
                "ADX": float(row["adx"]) if pd.notna(row.get("adx")) else np.nan,
                "触发逻辑": trigger,
            }
        )

    return pd.DataFrame(rows)


def current_trade_plan(signal_frame: pd.DataFrame, latest_quote: float) -> dict[str, object]:
    row = signal_frame.iloc[-1]
    buffer = max(float(row["vol_buffer"]), float(latest_quote) * 0.002)
    entry_low = latest_quote - buffer * 0.25
    entry_high = latest_quote + buffer * 0.25

    if bool(row.get("adx_range_filter", False)):
        side = "震荡市观望(ADX<20)"
        action = "趋势强度不足，暂停多空入场"
        stop = np.nan
        target = np.nan
        strength = max(float(row["long_score_pct"]), float(row["short_score_pct"]))
        tone = "flat"
    elif bool(row["long_setup"]):
        side = "多头入场窗口"
        action = "可关注回踩不破后的多头入场"
        stop = latest_quote - buffer * float(row["stop_mult"])
        target = latest_quote + buffer * float(row["stop_mult"]) * float(row["reward_risk"])
        strength = float(row["long_score_pct"])
        tone = "long"
    elif bool(row["short_setup"]):
        side = "空头入场窗口"
        action = "可关注反抽不过后的空头入场"
        stop = latest_quote + buffer * float(row["stop_mult"])
        target = latest_quote - buffer * float(row["stop_mult"]) * float(row["reward_risk"])
        strength = float(row["short_score_pct"])
        tone = "short"
    elif float(row["long_score_pct"]) > float(row["short_score_pct"]):
        side = "多头观察"
        action = "等待 U_score 或动量进一步确认"
        stop = np.nan
        target = np.nan
        strength = float(row["long_score_pct"])
        tone = "watch"
    elif float(row["short_score_pct"]) > float(row["long_score_pct"]):
        side = "空头观察"
        action = "等待 U_score 或动量进一步确认"
        stop = np.nan
        target = np.nan
        strength = float(row["short_score_pct"])
        tone = "watch"
    else:
        side = "观望"
        action = "多空条件不充分，避免追单"
        stop = np.nan
        target = np.nan
        strength = 0.0
        tone = "flat"

    return {
        "方向": side,
        "动作": action,
        "入场区间": f"{entry_low:,.2f} - {entry_high:,.2f}",
        "参考止损": stop,
        "第一目标": target,
        "信号强度": strength,
        "tone": tone,
    }


def format_news_table(news: pd.DataFrame, limit: int = 20, keyword: str | None = None) -> pd.DataFrame:
    columns = ["发布时间", "类别", "来源", "关键词", "情绪分", "新闻内容（中文翻译）", "对白银影响", "链接"]
    if news.empty:
        return pd.DataFrame(columns=columns)

    frame = news.copy()
    if keyword:
        frame = frame[frame["keyword"].str.contains(keyword, case=False, na=False)]
    frame = frame.head(limit).copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    defaults = {
        "published": pd.NaT,
        "source": "",
        "title": "",
        "summary": "",
        "score": 0.0,
        "keyword": "",
        "category": "宏观/其他",
        "link": "",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    frame["发布时间"] = pd.to_datetime(frame["published"]).dt.strftime("%Y-%m-%d %H:%M")
    frame["类别"] = frame["category"].fillna("宏观/其他")
    frame["来源"] = frame["source"]
    frame["关键词"] = frame["keyword"]
    frame["_score_value"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)
    frame["情绪分"] = frame["_score_value"]
    frame["新闻内容（中文翻译）"] = frame.apply(
        lambda row: build_chinese_news_summary(
            row.get("title", ""),
            row.get("summary", ""),
            row.get("keyword", ""),
        ),
        axis=1,
    )
    frame["对白银影响"] = frame.apply(
        lambda row: describe_silver_impact(
            row.get("title", ""),
            row.get("summary", ""),
            row.get("keyword", ""),
            float(row.get("_score_value", 0.0)),
        ),
        axis=1,
    )
    frame["链接"] = frame["link"]
    return frame[columns]


def clean_news_text(value: object, max_length: int = 360) -> str:
    text = unescape(str(value or ""))
    replacements = {
        "бк": "-",
        "вЂ“": "-",
        "вЂ”": "-",
        "вЂ™": "'",
        "вЂњ": '"',
        "вЂќ": '"',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def parse_google_translation(payload: object) -> str:
    if not isinstance(payload, list) or not payload:
        return ""

    chunks: list[str] = []
    first_block = payload[0]
    if not isinstance(first_block, list):
        return ""

    for item in first_block:
        if isinstance(item, list) and item:
            chunks.append(str(item[0] or ""))

    return clean_news_text("".join(chunks), max_length=1400)


def translate_with_google(text: str, max_length: int) -> str:
    response = requests.get(
        GOOGLE_TRANSLATION_URL,
        params={
            "client": "gtx",
            "sl": "auto",
            "tl": "zh-CN",
            "dt": "t",
            "q": text,
        },
        headers=DATA_HEADERS,
        timeout=10,
    )
    response.raise_for_status()
    return clean_news_text(parse_google_translation(response.json()), max_length=max_length)


@st.cache_data(ttl=86400, show_spinner=False)
def translate_news_text(text: str, max_length: int = 520) -> str:
    cleaned = clean_news_text(text, max_length=max_length)
    if not cleaned or contains_chinese(cleaned):
        return cleaned

    try:
        response = requests.get(
            TRANSLATION_URL,
            params={"q": cleaned, "langpair": "en|zh-CN"},
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
        translated = str(payload.get("responseData", {}).get("translatedText", "")).strip()
        translated = clean_news_text(translated, max_length=max_length)
        if translated and contains_chinese(translated) and translated.lower() != cleaned.lower():
            return translated
    except Exception:
        pass

    try:
        translated = translate_with_google(cleaned, max_length=max_length)
        if translated and contains_chinese(translated):
            return translated
    except Exception:
        pass

    return cleaned


def build_chinese_news_summary(title: object, summary: object, keyword: object) -> str:
    clean_title = clean_news_text(title, max_length=600)
    clean_summary = clean_news_text(summary, max_length=1200)

    translated_title = translate_news_text(clean_title, max_length=700)
    translated_summary = translate_news_text(clean_summary, max_length=1200) if clean_summary else ""

    if translated_summary and translated_summary.lower() not in translated_title.lower():
        return f"标题：{translated_title}；摘要：{translated_summary}"
    return f"标题：{translated_title}"


def contains_market_term(text: str, terms: Iterable[str]) -> bool:
    for term in terms:
        if len(term) <= 4 and term.replace(" ", "").isalpha():
            if re.search(rf"\b{re.escape(term)}\b", text):
                return True
        elif term in text:
            return True
    return False


def describe_silver_impact(title: object, summary: object, keyword: object, score: float) -> str:
    text = f"{title} {summary}".lower()
    keyword_text = str(keyword or "").lower()
    hawkish_terms = (
        "rate hike",
        "rate hikes",
        "higher rate",
        "higher rates",
        "higher-for-longer",
        "higher yield",
        "higher yields",
        "yield rises",
        "yields rise",
        "sticky inflation",
        "strong dollar",
        "dollar rises",
        "tightening",
        "tariff inflation",
        "inflation pressure",
        "inflationary",
    )
    dovish_terms = (
        "rate cut",
        "rate cuts",
        "lower rate",
        "lower rates",
        "lower yield",
        "lower yields",
        "dovish",
        "easing",
        "weaker dollar",
        "dollar falls",
        "stimulus",
    )
    deescalation_terms = ("ceasefire", "truce", "peace deal", "de-escalation", "deescalation")
    energy_shock_terms = (
        "hormuz",
        "strait of hormuz",
        "oil",
        "crude",
        "brent",
        "wti",
        "tanker",
        "freight",
        "shipping",
        "energy prices",
        "energy stocks",
        "gasoline",
    )
    inflation_terms = (
        "inflation",
        "cpi",
        "ppi",
        "price pressure",
        "commodity prices",
        "higher prices",
    )
    strong_dollar_terms = ("strong dollar", "dollar rises", "dollar strength", "greenback rises")
    weak_dollar_terms = ("weaker dollar", "dollar falls", "dollar weakness", "greenback falls")
    demand_terms = (
        "silver demand",
        "solar",
        "photovoltaic",
        "pv module",
        "industrial demand",
        "electronics",
        "electrical",
        "semiconductor",
        "chip",
        "electric vehicle",
        "ev",
        "battery",
        "grid",
        "electrification",
        "conductive",
        "conductivity",
        "supply deficit",
        "silver shortage",
    )
    weak_growth_terms = ("slowdown", "recession", "weak demand", "factory contraction")
    has_geopolitical_conflict = has_geopolitical_conflict_text(text)

    impact_score = 0.0
    drivers: list[str] = []

    if contains_market_term(text, deescalation_terms):
        impact_score -= 0.8
        drivers.append("冲突缓和会削弱避险买盘")

    if has_geopolitical_conflict:
        impact_score += 0.8
        drivers.append("地缘风险会带来一定避险需求")

    if has_geopolitical_conflict and contains_market_term(text, energy_shock_terms):
        impact_score -= 2.0
        drivers.append("霍尔木兹/油价/航运冲击可能推高能源价格和通胀，进而强化美联储维持高利率或加息的压力")
    elif contains_market_term(text, energy_shock_terms) and contains_market_term(text, inflation_terms):
        impact_score -= 1.5
        drivers.append("能源价格和通胀压力上升会抬高实际利率预期，对无息白银不利")
    elif contains_market_term(text, inflation_terms):
        impact_score -= 0.7
        drivers.append("通胀压力若引发更紧货币政策，会压制白银估值")

    if contains_market_term(text, hawkish_terms):
        impact_score -= 1.8
        drivers.append("加息、高收益率或偏鹰政策会提高持有白银的机会成本")
    if contains_market_term(text, dovish_terms):
        impact_score += 1.8
        drivers.append("降息、收益率下行或宽松政策会减轻贵金属的利率压力")

    if contains_market_term(text, strong_dollar_terms):
        impact_score -= 1.2
        drivers.append("美元走强通常压制以美元计价的白银")
    if contains_market_term(text, weak_dollar_terms):
        impact_score += 1.2
        drivers.append("美元走弱通常支撑以美元计价的白银")

    if contains_market_term(text, demand_terms):
        impact_score += 1.2
        drivers.append("工业、光伏、电子、电动车或供需缺口改善会支撑白银实物需求")
    if contains_market_term(text, weak_growth_terms):
        impact_score -= 1.0
        drivers.append("经济放缓会压制白银工业需求")

    if "fed" in keyword_text or "federal reserve" in text:
        if score <= -0.20:
            impact_score -= 0.7
            drivers.append("美联储相关不确定性会让实际收益率路径更偏压力")
        elif score >= 0.20:
            impact_score += 0.5
            drivers.append("若市场解读为更温和的美联储路径，则对白银有一定支撑")
        else:
            drivers.append("美联储新闻需要继续观察其对降息预期、实际收益率和美元的影响")
    elif not drivers:
        if score >= 0.20:
            impact_score += 0.3
            drivers.append("标题情绪偏正面，可能改善金属风险偏好")
        elif score <= -0.20:
            impact_score -= 0.3
            drivers.append("标题情绪偏负面，可能压制周期需求预期")

    if impact_score >= 1.2:
        label = "偏多白银"
    elif impact_score >= 0.4:
        label = "轻微偏多白银"
    elif impact_score <= -1.2:
        label = "偏空白银"
    elif impact_score <= -0.4:
        label = "轻微偏空白银"
    else:
        label = "多空影响交织"

    if not drivers:
        drivers.append("新闻未明显改变利率、美元、通胀、需求或避险驱动")
    return f"{label}：{'；'.join(drivers[:3])}。"


def format_home_news_briefing(news: pd.DataFrame, limit: int = HOME_NEWS_PAGE_SIZE, offset: int = 0) -> pd.DataFrame:
    columns = [
        "主要内容（中文翻译）",
        "对白银影响",
        "时间",
        "类别",
        "来源",
        "触发词",
        "情绪分",
        "链接",
    ]
    if news.empty:
        return pd.DataFrame(columns=columns)

    start = max(int(offset), 0)
    end = start + max(int(limit), 1)
    frame = news.iloc[start:end].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    defaults = {
        "published": pd.NaT,
        "source": "",
        "title": "",
        "summary": "",
        "score": 0.0,
        "keyword": "",
        "category": "宏观/其他",
        "link": "",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    frame["时间"] = pd.to_datetime(frame["published"]).dt.strftime("%Y-%m-%d %H:%M")
    frame["类别"] = frame["category"].fillna("宏观/其他")
    frame["来源"] = frame["source"]
    frame["触发词"] = frame["keyword"]
    frame["_score_value"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)
    frame["情绪分"] = frame["_score_value"]
    frame["主要内容（中文翻译）"] = frame.apply(
        lambda row: build_chinese_news_summary(
            row.get("title", ""),
            row.get("summary", ""),
            row.get("keyword", ""),
        ),
        axis=1,
    )
    frame["对白银影响"] = frame.apply(
        lambda row: describe_silver_impact(
            row.get("title", ""),
            row.get("summary", ""),
            row.get("keyword", ""),
            float(row.get("_score_value", 0.0)),
        ),
        axis=1,
    )
    frame["链接"] = frame["link"]
    return frame[columns]


def render_home_news_briefing(
    news: pd.DataFrame,
    page_size: int = HOME_NEWS_PAGE_SIZE,
    page_state_key: str = "home_news_page",
    empty_message: str = "当前没有可展示的相关新闻。",
) -> None:
    if news.empty:
        st.info(empty_message)
        return

    total_items = len(news)
    total_pages = max(1, int(np.ceil(total_items / page_size)))
    current_page = int(st.session_state.get(page_state_key, 0))
    current_page = min(max(current_page, 0), total_pages - 1)
    st.session_state[page_state_key] = current_page

    left, middle, right = st.columns([1, 2, 1])
    with left:
        if st.button("上一页", disabled=current_page <= 0, key=f"{page_state_key}_prev"):
            st.session_state[page_state_key] = max(current_page - 1, 0)
            st.rerun()
    with middle:
        st.markdown(
            f"<div style='text-align:center;color:#667789;padding-top:0.45rem;'>第 {current_page + 1} / {total_pages} 页 · 共 {total_items} 条相关新闻</div>",
            unsafe_allow_html=True,
        )
    with right:
        if st.button("下一页", disabled=current_page >= total_pages - 1, key=f"{page_state_key}_next"):
            st.session_state[page_state_key] = min(current_page + 1, total_pages - 1)
            st.rerun()

    home_news = format_home_news_briefing(news, limit=page_size, offset=current_page * page_size)
    if home_news.empty:
        st.info("当前页没有可展示的相关新闻。")
        return

    st.markdown(
        """
        <style>
        .home-news-item {
            border: 1px solid #D7DEE8;
            border-left: 4px solid #2F4858;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
            background: #FFFFFF;
        }
        .home-news-main {
            color: #17212B;
            font-size: 0.98rem;
            line-height: 1.62;
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            word-break: break-word;
        }
        .home-news-impact {
            margin-top: 0.55rem;
            color: #364B5F;
            line-height: 1.55;
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            word-break: break-word;
        }
        .home-news-meta {
            margin-top: 0.55rem;
            color: #667789;
            font-size: 0.84rem;
            line-height: 1.45;
            white-space: normal;
            word-break: break-word;
        }
        .home-news-meta a {
            color: #1F6FEB;
            text-decoration: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for _, row in home_news.iterrows():
        main_text = escape(str(row.get("主要内容（中文翻译）", "")))
        impact = escape(str(row.get("对白银影响", "")))
        category = escape(str(row.get("类别", "")))
        source = escape(str(row.get("来源", "")))
        published = escape(str(row.get("时间", "")))
        keyword = escape(str(row.get("触发词", "")))
        score = pd.to_numeric(row.get("情绪分", np.nan), errors="coerce")
        score_text = "" if pd.isna(score) else f"{float(score):.3f}"
        link = str(row.get("链接", "") or "").strip()
        link_html = f' · <a href="{escape(link)}" target="_blank" rel="noopener noreferrer">原文链接</a>' if link else ""

        st.markdown(
            f"""
            <article class="home-news-item">
                <div class="home-news-main">{main_text}</div>
                <div class="home-news-impact"><strong>对白银影响：</strong>{impact}</div>
                <div class="home-news-meta">{published} · {category} · {source} · 触发词：{keyword} · 情绪分：{score_text}{link_html}</div>
            </article>
            """,
            unsafe_allow_html=True,
        )


def render_categorized_home_news(news: pd.DataFrame) -> None:
    if news.empty:
        st.info("当前没有可展示的相关新闻。")
        return

    if "category" not in news.columns:
        news = news.copy()
        news["category"] = "宏观/其他"

    industrial_news = news[news["category"] == "工业相关"].copy()
    other_news = news[news["category"] != "工业相关"].copy()

    industrial_tab, other_tab = st.tabs(
        [
            f"工业相关新闻 ({len(industrial_news)})",
            f"宏观/其他新闻 ({len(other_news)})",
        ]
    )
    with industrial_tab:
        render_home_news_briefing(
            industrial_news,
            page_size=HOME_NEWS_PAGE_SIZE,
            page_state_key="home_industrial_news_page",
            empty_message="当前没有匹配到工业、光伏、电子、半导体、电动车或供需缺口相关新闻。",
        )
    with other_tab:
        render_home_news_briefing(
            other_news,
            page_size=HOME_NEWS_PAGE_SIZE,
            page_state_key="home_other_news_page",
            empty_message="当前没有可展示的宏观/其他相关新闻。",
        )


def build_price_chart(filtered: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered.index,
            y=filtered["observed"],
            mode="lines",
            name="SI=F 实际价格",
            line=dict(color="#2F4858", width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=filtered.index,
            y=filtered["kalman_price"],
            mode="lines",
            name="卡尔曼滤波价格",
            line=dict(color="#F28E2B", width=2.2),
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=35, b=10),
        title="白银期货实际价格 vs 卡尔曼滤波均线",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title="美元 / 盎司",
    )
    return fig


def build_macro_chart(macro: pd.DataFrame) -> go.Figure:
    chart = macro.dropna(subset=["U_score"]).copy()
    colors = np.where(chart["U_score"] >= 0.0, "#2E7D32", "#B23A48")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart.index,
            y=chart["U_score"],
            marker_color=colors,
            name="U_score",
            hovertemplate="%{x|%Y-%m-%d}<br>宏观总分 U_score=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_color="#777777", line_width=1)
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=35, b=10),
        title="每日宏观综合打分",
        xaxis_title=None,
        yaxis_title="加权 Z-Score",
        showlegend=False,
    )
    return fig


def build_velocity_chart(filtered: pd.DataFrame) -> go.Figure:
    buy = filtered[filtered["buy_signal"]]
    sell = filtered[filtered["sell_signal"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered.index,
            y=filtered["velocity"],
            mode="lines",
            name="隐藏动量",
            line=dict(color="#4E79A7", width=2.0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=buy.index,
            y=buy["velocity"],
            mode="markers",
            name="买入",
            marker=dict(color="#2E7D32", size=10, symbol="triangle-up"),
            hovertemplate="%{x|%Y-%m-%d}<br>买入动量=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sell.index,
            y=sell["velocity"],
            mode="markers",
            name="卖出",
            marker=dict(color="#B23A48", size=10, symbol="triangle-down"),
            hovertemplate="%{x|%Y-%m-%d}<br>卖出动量=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_color="#777777", line_width=1)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=35, b=10),
        title="隐藏动量状态切换",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title="动量",
    )
    return fig


def build_contribution_chart(macro: pd.DataFrame) -> go.Figure:
    chart = macro.tail(180).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart.index, y=chart["sentiment_factor"], name="情绪贡献", marker_color="#59A14F"))
    fig.add_trace(go.Bar(x=chart.index, y=chart["dxy_factor"], name="美元指数贡献", marker_color="#4E79A7"))
    fig.add_trace(go.Bar(x=chart.index, y=chart["rate_factor"], name="利率贡献", marker_color="#B07AA1"))
    fig.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart["U_score"],
            name="U_score",
            mode="lines",
            line=dict(color="#F28E2B", width=2.4),
        )
    )
    fig.add_hline(y=0.0, line_color="#777777", line_width=1)
    fig.update_layout(
        barmode="relative",
        height=380,
        margin=dict(l=10, r=10, t=35, b=10),
        title="U_score 拆解：情绪 + 美元 + 利率",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title="分数贡献",
    )
    return fig


def build_factor_bar_line_chart(
    frame: pd.DataFrame,
    bar_col: str,
    line_col: str,
    title: str,
    bar_name: str,
    line_name: str,
) -> go.Figure:
    chart = frame.tail(180).copy()
    colors = np.where(chart[bar_col] >= 0.0, "#2E7D32", "#B23A48")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart.index,
            y=chart[bar_col],
            name=bar_name,
            marker_color=colors,
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart[line_col],
            name=line_name,
            mode="lines",
            yaxis="y2",
            line=dict(color="#4E79A7", width=2.0),
        )
    )
    fig.add_hline(y=0.0, line_color="#777777", line_width=1)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=35, b=10),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis=dict(title=bar_name),
        yaxis2=dict(title=line_name, overlaying="y", side="right", showgrid=False),
    )
    return fig


def sidebar_controls() -> dict[str, bool | float | int | str]:
    st.sidebar.title(APP_NAME)
    st.sidebar.caption("本地白银宏观情绪卡尔曼交易面板。")

    period_labels = {"6mo": "6 个月", "1y": "1 年", "2y": "2 年", "5y": "5 年", "10y": "10 年"}
    interval_labels = {"1d": "日线", "1h": "小时线"}
    period = st.sidebar.selectbox(
        "历史窗口",
        ["6mo", "1y", "2y", "5y", "10y"],
        index=2,
        format_func=lambda value: period_labels[value],
    )
    interval = st.sidebar.selectbox(
        "行情周期",
        ["1d", "1h"],
        index=0,
        format_func=lambda value: interval_labels[value],
    )
    z_window = st.sidebar.slider("Z-Score 滚动窗口", 20, 180, 60, 5)

    st.sidebar.divider()
    rho = st.sidebar.slider("Rho：动量衰减", 0.00, 1.00, 0.86, 0.01)
    alpha = st.sidebar.slider("Alpha：宏观敏感度", 0.00, 2.00, 0.25, 0.01)
    q = st.sidebar.slider("Q：过程噪声", 0.0001, 5.0000, 0.0800, 0.0001, format="%.4f")
    r = st.sidebar.slider("R：测量噪声", 0.0100, 50.0000, 2.5000, 0.0100, format="%.4f")

    st.sidebar.divider()
    st.sidebar.caption("交易机会阈值")
    long_threshold = st.sidebar.slider("多头 U_score 阈值", -2.00, 2.00, 0.25, 0.05)
    short_threshold = st.sidebar.slider("空头 U_score 阈值", -2.00, 2.00, -0.25, 0.05)
    min_velocity = st.sidebar.slider("最小动量强度", 0.00, 3.00, 0.05, 0.01)
    stop_mult = st.sidebar.slider("止损波动倍数", 0.50, 5.00, 1.50, 0.10)
    reward_risk = st.sidebar.slider("目标 / 风险比", 0.50, 5.00, 2.00, 0.10)

    st.sidebar.divider()
    live_quote_refresh = st.sidebar.toggle("秒级刷新白银报价", value=True)
    st.sidebar.caption(f"白银实时报价每 {LIVE_QUOTE_REFRESH_SECONDS} 秒刷新；行情、FRED 和 RSS 仍走缓存，避免页面过慢。")
    if st.sidebar.button("刷新缓存"):
        st.cache_data.clear()
        st.rerun()

    return {
        "period": period,
        "interval": interval,
        "z_window": z_window,
        "rho": rho,
        "alpha": alpha,
        "q": q,
        "r": r,
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "min_velocity": min_velocity,
        "stop_mult": stop_mult,
        "reward_risk": reward_risk,
        "live_quote_refresh": live_quote_refresh,
    }


def render_warnings(warnings: Iterable[str]) -> None:
    for warning in warnings:
        st.warning(warning)


live_quote_fragment = (
    st.fragment(run_every=LIVE_QUOTE_REFRESH_SECONDS) if hasattr(st, "fragment") else (lambda function: function)
)


@live_quote_fragment
def render_live_silver_quote(enabled: bool, fallback_quote: float) -> None:
    if not enabled:
        return

    quote = fetch_latest_silver_price()
    display_price = quote.price if quote.price is not None else fallback_quote
    c1, c2, c3 = st.columns([1.1, 1.0, 1.6])
    c1.metric("秒级白银报价", f"{float(display_price):,.2f}")
    c2.metric("报价源", quote.source)
    c3.metric("报价更新时间", format_quote_timestamp(quote.timestamp))
    st.caption(f"该报价区块每 {LIVE_QUOTE_REFRESH_SECONDS} 秒独立刷新；策略计算仍使用当前历史窗口和缓存行情。")
    if quote.warning:
        st.warning(quote.warning)


def render_trade_tab(
    filtered: pd.DataFrame,
    macro: pd.DataFrame,
    signal_frame: pd.DataFrame,
    opportunities: pd.DataFrame,
    latest_quote: float,
) -> None:
    st.warning("本页面只输出量化策略信号，不构成投资建议，也不会自动下单。期货杠杆风险高，入场前需要结合保证金、滑点、合约乘数和账户风险限额。")

    plan = current_trade_plan(signal_frame, latest_quote)
    latest = signal_frame.iloc[-1]

    if plan["tone"] == "long":
        st.success(f"当前信号：{plan['方向']}。{plan['动作']}。")
    elif plan["tone"] == "short":
        st.error(f"当前信号：{plan['方向']}。{plan['动作']}。")
    elif plan["tone"] == "watch":
        st.info(f"当前信号：{plan['方向']}。{plan['动作']}。")
    else:
        st.info(f"当前信号：{plan['方向']}。{plan['动作']}。")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("建议状态", str(plan["方向"]))
    c2.metric("信号强度", f"{float(plan['信号强度']):.0f}%")
    c3.metric("参考入场区间", str(plan["入场区间"]))
    stop_text = "等待确认" if pd.isna(plan["参考止损"]) else f"{float(plan['参考止损']):,.2f}"
    target_text = "等待确认" if pd.isna(plan["第一目标"]) else f"{float(plan['第一目标']):,.2f}"
    c4.metric("参考止损", stop_text)
    c5.metric("第一目标", target_text)

    st.markdown(
        f"""
        **触发规则**

        多头入场窗口：隐藏动量 > 阈值、U_score >= 多头阈值、价格在卡尔曼滤波线上方。

        空头入场窗口：隐藏动量 < -阈值、U_score <= 空头阈值、价格在卡尔曼滤波线下方。

        震荡过滤：`ADX < 20` 时强制观望，不给出开多或开空。

        当前价格偏离滤波线：`{latest['price_bias_pct']:.2f}%`；当前波动缓冲：`{latest['vol_buffer']:.3f}`；当前 ADX：`{latest.get('adx', np.nan):.2f}`。
        """
    )

    st.plotly_chart(build_price_chart(filtered), use_container_width=True, key="trade_price_chart")
    st.plotly_chart(build_velocity_chart(filtered), use_container_width=True, key="trade_velocity_chart")

    st.subheader("最近多空入场机会")
    if opportunities.empty:
        st.info("当前历史窗口内没有满足完整条件的多空入场机会。可以降低阈值，或等待动量与宏观分数共振。")
    else:
        st.dataframe(
            opportunities.style.format(
                {
                    "信号强度": "{:.0f}%",
                    "参考止损": "{:.2f}",
                    "第一目标": "{:.2f}",
                    "U_score": "{:.2f}",
                    "动量": "{:.3f}",
                    "ADX": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.plotly_chart(build_macro_chart(macro.loc[filtered.index]), use_container_width=True, key="trade_macro_chart")


def render_sentiment_tab(news: pd.DataFrame, sentiment_daily: pd.DataFrame, macro: pd.DataFrame) -> None:
    st.subheader("情绪因子：新闻标题 -> VADER -> 日均值 -> Z-Score")
    st.markdown(
        f"""
        1. 从 CNBC、Yahoo Finance、Google News RSS 抓取包含 `Silver / Fed / War / 工业需求 / 光伏 / 电子 / 半导体` 的标题。
        2. 使用 VADER `compound` 得到每条标题的情绪分，范围为 `-1` 到 `1`。
        3. 按日期求平均，得到 `sentiment_raw`。
        4. 对 `sentiment_raw` 做 EWM Z-Score，得到 `z_sentiment`（近期权重更高）。
        5. 情绪贡献 = `{SENTIMENT_WEIGHT:.2f} * z_sentiment`。
        """
    )

    chart = macro.tail(180).copy()
    fig = build_factor_bar_line_chart(
        chart,
        bar_col="sentiment_factor",
        line_col="sentiment_raw",
        title="情绪贡献与原始新闻情绪",
        bar_name="情绪贡献",
        line_name="原始情绪分",
    )
    st.plotly_chart(fig, use_container_width=True, key="sentiment_factor_chart")

    daily_table = chart[["sentiment_raw", "z_sentiment", "sentiment_factor"]].tail(20).rename(
        columns={"sentiment_raw": "原始情绪分", "z_sentiment": "情绪 Z-Score", "sentiment_factor": "情绪贡献"}
    )
    st.dataframe(daily_table.style.format("{:.3f}"), use_container_width=True)

    st.subheader("相关新闻")
    news_table = format_news_table(news, limit=30)
    if news_table.empty:
        st.info("当前 RSS 源没有返回匹配新闻。")
    else:
        st.dataframe(news_table.style.format({"情绪分": "{:.3f}"}), use_container_width=True, hide_index=True)


def render_dollar_tab(macro: pd.DataFrame) -> None:
    st.subheader("美元指数因子：美元走强通常压制白银")
    st.markdown(
        f"""
        1. 日线优先使用 FRED `{FRED_DOLLAR_INDEX}` 美元广义指数；小时线优先使用 Yahoo Chart `{DXY_TICKER}`。
        2. 当主源不可用时，自动回退到备用源并在首页显示告警。
        3. 计算变动：`dxy_delta = pct_change(DXY) * 100`。
        4. 对 `dxy_delta` 做 EWM Z-Score，得到 `z_dxy`。
        5. 美元因子采用反向符号：`dxy_factor = -{DOLLAR_WEIGHT:.2f} * z_dxy`。
        """
    )

    fig = build_factor_bar_line_chart(
        macro,
        bar_col="dxy_factor",
        line_col="dxy",
        title="美元指数贡献与 DXY 水平",
        bar_name="美元指数贡献",
        line_name="DXY",
    )
    st.plotly_chart(fig, use_container_width=True, key="dollar_factor_chart")

    table = macro[["dxy", "dxy_delta", "z_dxy", "dxy_factor"]].tail(30).rename(
        columns={"dxy": "美元指数", "dxy_delta": "美元指数变动%", "z_dxy": "美元指数 Z-Score", "dxy_factor": "美元指数贡献"}
    )
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)


def render_rate_tab(macro: pd.DataFrame) -> None:
    st.subheader("利率因子：实际利率或收益率上行通常压制白银")
    st.markdown(
        f"""
        1. 优先使用 FRED `{FRED_REAL_RATE}` 十年期实际利率。
        2. 名义 10 年期收益率优先使用 FRED `{FRED_TREASURY_10Y}`；小时线或 FRED 不可用时回退 Yahoo Chart `{TNX_TICKER}`。
        3. 计算 `rate_delta`，再做 EWM Z-Score 得到 `z_rate`。
        4. 利率因子采用反向符号：`rate_factor = -{RATE_WEIGHT:.2f} * z_rate`。
        """
    )

    fig = build_factor_bar_line_chart(
        macro,
        bar_col="rate_factor",
        line_col="real_rate",
        title="利率贡献与十年期实际利率",
        bar_name="利率贡献",
        line_name="实际利率",
    )
    st.plotly_chart(fig, use_container_width=True, key="rate_factor_chart")

    table = macro[["real_rate", "tnx", "rate_delta", "z_rate", "rate_factor"]].tail(30).rename(
        columns={
            "real_rate": "十年期实际利率",
            "tnx": "十年期美债收益率",
            "rate_delta": "利率变动",
            "z_rate": "利率 Z-Score",
            "rate_factor": "利率贡献",
        }
    )
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)


def render_composite_tab(macro: pd.DataFrame) -> None:
    st.subheader("宏观综合分 U_score")
    st.markdown(
        f"""
        `U_score = {SENTIMENT_WEIGHT:.2f} * z_sentiment - {DOLLAR_WEIGHT:.2f} * z_dxy - {RATE_WEIGHT:.2f} * z_rate`

        解释：情绪越正面越利多；美元指数走强、实际利率或收益率上行通常利空白银。
        """
    )
    st.plotly_chart(build_contribution_chart(macro), use_container_width=True, key="composite_contribution_chart")

    table = macro[
        ["sentiment_factor", "dxy_factor", "rate_factor", "U_score", "sentiment_raw", "dxy_delta", "rate_delta"]
    ].tail(30).rename(
        columns={
            "sentiment_factor": "情绪贡献",
            "dxy_factor": "美元贡献",
            "rate_factor": "利率贡献",
            "U_score": "宏观总分 U_score",
            "sentiment_raw": "原始情绪分",
            "dxy_delta": "美元指数变动%",
            "rate_delta": "利率变动",
        }
    )
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)


def render_kalman_tab(filtered: pd.DataFrame, signal_frame: pd.DataFrame, controls: dict[str, object]) -> None:
    st.subheader("卡尔曼动量因子")
    st.markdown(
        f"""
        状态向量：`x = [价格, 动量]^T`

        状态转移矩阵：`F = [[1, 1], [0, Rho]]`，当前 `Rho = {float(controls['rho']):.2f}`。

        控制矩阵：`B = [[0.5 * Alpha], [Alpha]]`，当前 `Alpha = {float(controls['alpha']):.2f}`。

        动态测量噪声：`R_dynamic = R_base * vol_ratio`，其中 `vol_ratio` 来自 14/60 期绝对波动率比值。

        宏观总分 `U_score` 会作为外生冲击映射到动量加速度；动量转正且宏观顺风时偏多，动量转负且宏观逆风时偏空。
        """
    )
    st.plotly_chart(build_price_chart(filtered), use_container_width=True, key="kalman_price_chart")
    st.plotly_chart(build_velocity_chart(filtered), use_container_width=True, key="kalman_velocity_chart")

    table = signal_frame[
        [
            "observed",
            "kalman_price",
            "price_bias_pct",
            "velocity",
            "adx",
            "U_score",
            "entry_side",
            "long_score_pct",
            "short_score_pct",
        ]
    ].tail(30).rename(
        columns={
            "observed": "实际价格",
            "kalman_price": "卡尔曼价格",
            "price_bias_pct": "偏离滤波线%",
            "velocity": "隐藏动量",
            "adx": "ADX",
            "U_score": "宏观总分 U_score",
            "entry_side": "信号状态",
            "long_score_pct": "多头强度",
            "short_score_pct": "空头强度",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "实际价格": "{:.2f}",
                "卡尔曼价格": "{:.2f}",
                "偏离滤波线%": "{:.2f}",
                "隐藏动量": "{:.3f}",
                "ADX": "{:.2f}",
                "宏观总分 U_score": "{:.2f}",
                "多头强度": "{:.0f}%",
                "空头强度": "{:.0f}%",
            }
        ),
        use_container_width=True,
    )


def render_raw_data_tab(
    market: pd.DataFrame,
    real_rate: pd.DataFrame,
    macro: pd.DataFrame,
    news: pd.DataFrame,
) -> None:
    st.subheader("行情与宏观原始数据")
    st.dataframe(market.tail(80), use_container_width=True)

    st.subheader("FRED 实际利率数据")
    if real_rate.empty:
        st.info("FRED 数据为空。")
    else:
        st.dataframe(real_rate.tail(80), use_container_width=True)

    st.subheader("宏观计算结果")
    st.dataframe(macro.tail(80), use_container_width=True)

    st.subheader("新闻原始抓取结果")
    news_table = format_news_table(news, limit=80)
    if news_table.empty:
        st.info("新闻数据为空。")
    else:
        st.dataframe(news_table.style.format({"情绪分": "{:.3f}"}), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} | 白银宏观卡尔曼交易终端", page_icon="Ag", layout="wide")
    controls = sidebar_controls()

    st.title("白银宏观卡尔曼交易终端")
    st.caption(
        "以 RSS 情绪、工业需求新闻、FRED / Yahoo Chart 行情、实际利率或美债收益率变动作为宏观控制输入，"
        "对 SI=F 白银期货价格状态进行二维卡尔曼滤波，并生成多空入场窗口。"
    )

    with st.spinner("正在加载 Yahoo Chart / FRED 行情、实际利率和 RSS 情绪数据..."):
        market = fetch_market_data(str(controls["period"]), str(controls["interval"]))
        silver_ohlc = fetch_silver_ohlc(str(controls["period"]), str(controls["interval"]))
        latest_quote_result = fetch_latest_silver_price()

        if market.data.empty:
            render_warnings(market.warnings)
            st.stop()

        start = market.data.index.min().to_pydatetime() - timedelta(days=10)
        end = market.data.index.max().to_pydatetime() + timedelta(days=1)
        real_rate = fetch_real_rate(start, end)
        news = fetch_news_sentiment(max_items=NEWS_FETCH_ITEMS_PER_SOURCE)

    all_warnings = dedupe_warnings(market.warnings + silver_ohlc.warnings + real_rate.warnings + news.warnings)
    if latest_quote_result.warning:
        all_warnings = dedupe_warnings(all_warnings + [latest_quote_result.warning])

    sentiment_daily = daily_sentiment(news.data)
    macro = compute_macro_score(
        closes=market.data,
        real_rate=real_rate.data,
        sentiment=sentiment_daily,
        z_window=int(controls["z_window"]),
    )
    filtered = run_kalman(
        price=macro["silver"],
        u_score=macro["U_score"],
        vol_ratio=macro["vol_ratio"],
        rho=float(controls["rho"]),
        alpha=float(controls["alpha"]),
        q=float(controls["q"]),
        r=float(controls["r"]),
    )
    adx_series = calculate_adx(silver_ohlc.data, period=14)

    if filtered.empty:
        st.error("数据清洗后没有可用的 SI=F 价格序列。")
        render_warnings(all_warnings)
        st.stop()

    latest_row = filtered.iloc[-1]
    latest_macro = macro.reindex(filtered.index).iloc[-1]
    latest_quote = latest_quote_result.price if latest_quote_result.price is not None else latest_row["observed"]
    signal_frame = build_signal_frame(
        filtered=filtered,
        macro=macro,
        adx=adx_series,
        long_threshold=float(controls["long_threshold"]),
        short_threshold=float(controls["short_threshold"]),
        min_velocity=float(controls["min_velocity"]),
        stop_mult=float(controls["stop_mult"]),
        reward_risk=float(controls["reward_risk"]),
    )
    opportunities = build_opportunity_table(signal_frame)
    current_plan = current_trade_plan(signal_frame, float(latest_quote))

    render_live_silver_quote(bool(controls["live_quote_refresh"]), float(latest_quote))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("策略参考报价", f"{latest_quote:,.2f}")
    c2.metric("当前信号", str(current_plan["方向"]))
    c3.metric("信号强度", f"{float(current_plan['信号强度']):.0f}%")
    c4.metric("卡尔曼价格", f"{latest_row['kalman_price']:,.2f}")
    c5.metric("隐藏动量", f"{latest_row['velocity']:,.3f}")
    c6.metric("宏观总分 U_score", f"{latest_row['U_score']:,.2f}")

    with st.expander("卡尔曼价格是什么意思？", expanded=False):
        st.markdown(
            f"""
            **卡尔曼价格不是实时成交价，也不是目标价。** 它是模型根据历史 `SI=F` 价格、
            宏观总分 `U_score` 和近期波动状态估计出的“去噪趋势价格”。

            可以把它理解为当前白银价格的动态平滑中枢：实际价格高于卡尔曼价格，
            说明价格站在模型趋势线上方；实际价格低于卡尔曼价格，说明价格弱于模型趋势中枢。
            本页会把它和“隐藏动量”一起使用，用来判断多空窗口，而不是单独给出买卖结论。

            当前显示的 `{latest_row['kalman_price']:,.2f}` 表示模型对最新时点白银期货趋势价格的估计。
            """
        )

    if all_warnings:
        with st.expander("数据源告警", expanded=False):
            render_warnings(all_warnings)

    st.subheader("相关新闻中文翻译与白银影响")
    render_categorized_home_news(news.data)

    tabs = st.tabs(["交易机会", "情绪因子", "美元指数因子", "利率因子", "综合分拆解", "卡尔曼动量", "原始数据"])
    with tabs[0]:
        render_trade_tab(filtered, macro, signal_frame, opportunities, float(latest_quote))
    with tabs[1]:
        render_sentiment_tab(news.data, sentiment_daily, macro)
    with tabs[2]:
        render_dollar_tab(macro)
    with tabs[3]:
        render_rate_tab(macro)
    with tabs[4]:
        render_composite_tab(macro)
    with tabs[5]:
        render_kalman_tab(filtered, signal_frame, controls)
    with tabs[6]:
        render_raw_data_tab(market.data, real_rate.data, macro, news.data)


if __name__ == "__main__":
    main()
