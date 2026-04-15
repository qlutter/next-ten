#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
넥스트 팔란티어 스캐너 v3.0  ─  10배 가능 종목 발굴 시스템
═══════════════════════════════════════════════════════════════════════════════
리서치 근거: next_palantir_research_full.txt
핵심 인사이트:
  - 팔란티어는 ATM 없이 SBC 희석만 → SBC/ATM 반드시 분리 평가
  - PER/PBR/DCF 아닌 계약가시성(RPO/ARR/TCV) + 총마진 + FCF가 주가 선행
  - 엔터프라이즈·미션크리티컬·규제산업·락인 구조 = 팔란티어형 멀티플 정당화
  - 섹터별 다른 지표 (소프트웨어/AI/국방 vs 우주 vs 양자)
  - Street 컨센서스가 항상 뒤처지는 종목 = 진짜 넥스트 팔란티어 신호

NPS (Next Palantir Score) 0~100점 구성:
  Block A: 성장의 질          (25pt) — 속도보다 반복성·계약지속성
  Block B: 마진·수익성        (20pt) — 총마진 80%+, Rule-of-40
  Block C: 계약 가시성        (20pt) — ARR/RPO/Deferred Rev 기반
  Block D: 희석 통제           (15pt) — ATM=강한 페널티, SBC=정당화 가능
  Block E: 팔란티어 유사성    (10pt) — 엔터프라이즈·미션크리티컬·락인
  Block F: 밸류에이션 매력도  (10pt) — EV/GP, 가격·목표가 괴리, <$50 보너스

Usage:
  python next_palantir_scanner.py                   # ticker.txt 읽기
  python next_palantir_scanner.py --tickers SAIL WAY PLTR SNOW
  python next_palantir_scanner.py --file my.txt --output out.csv --workers 5
  python next_palantir_scanner.py --tickers PLTR --log-level DEBUG
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# ── 설정 상수 ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# 네트워크
FETCH_DELAY_SEC:        float = 0.4
RETRY_MAX:              int   = 3
RETRY_BASE_DELAY:       float = 2.0
DEFAULT_WORKERS:        int   = 5
DEFAULT_TICKER_FILE:    str   = "ticker.txt"
DEFAULT_OUTPUT_FILE:    str   = "results/nps_results.csv"

# ── 단계 분류 임계값 ─────────────────────────────────────────────────────────
STAGE_EARLY_REV_MAX:        float = 3e8    # $300M 미만 → 초기
STAGE_HYPER_GROWTH_MIN:     float = 0.50   # 50%+ → 하이퍼
STAGE_HIGH_GROWTH_MIN:      float = 0.25   # 25%+ → 고성장가속
STAGE_MATURE_FCF_MIN:       float = 0.10   # FCF 10%+ → 성숙

# ── 희석 임계값 ──────────────────────────────────────────────────────────────
ATM_HIGH_RISK:              float = 0.08   # 8%+ ATM → 고위험
ATM_WARN:                   float = 0.04   # 4%+ ATM → 경고
SBC_HIGH:                   float = 0.12   # 12%+ SBC → 고희석
SBC_JUSTIFIED_GM:           float = 0.70   # 총마진 70%+ → SBC 정당화 조건1
SBC_JUSTIFIED_FCF:          float = 0.08   # FCF마진 8%+ → SBC 정당화 조건2
SBC_JUSTIFIED_GR:           float = 0.20   # 성장률 20%+ → SBC 정당화 조건3

# ── 계약가시성 임계값 ────────────────────────────────────────────────────────
VISIBILITY_HIGH:            float = 0.20   # Deferred/매출 20%+ → 높음
VISIBILITY_MED:             float = 0.08   # 8%+ → 보통

# ── 팔란티어 유사성 기준 ─────────────────────────────────────────────────────
# 섹터 키워드: 소프트웨어/AI/국방, 우주, 양자, 핀테크, 헬스케어
PLTR_SECTORS_SW    = {"Technology", "Software", "Information Technology"}
PLTR_SECTORS_SPACE = {"Industrials", "Aerospace", "Defense"}
PLTR_SECTORS_QUANT = {"Technology"}           # 추가 키워드로 구분
PLTR_SECTORS_FIN   = {"Financial Services", "Financial Technology"}
PLTR_SECTORS_HEALTH= {"Healthcare", "Health Care"}

# 넥스트 팔란티어 가격 필터 ($50 이하 = 1점 보너스)
PRICE_FILTER_MAX:           float = 50.0

# ── Block 배점 ───────────────────────────────────────────────────────────────
#  A: 성장의 질
A_GROWTH_SPEED:             float = 10.0   # 매출성장 속도
A_NRR_QUALITY:              float = 8.0    # NRR / 고객 확장성
A_RECURRING_QUALITY:        float = 7.0    # 반복매출 비중 프록시

#  B: 마진·수익성
B_GROSS_MARGIN:             float = 10.0   # 총마진
B_FCF_MARGIN:               float = 6.0    # FCF마진
B_RULE40:                   float = 4.0    # Rule-of-40

#  C: 계약 가시성
C_DEFERRED_RATIO:           float = 10.0   # Deferred Rev / 매출
C_DEFERRED_GROWTH:          float = 5.0    # Deferred Rev YoY 성장
C_CASH_RUNWAY:              float = 5.0    # 현금 런웨이(초기 단계용)

#  D: 희석 통제
D_ATM_PENALTY_MAX:          float = -15.0  # ATM 최대 페널티
D_SBC_PENALTY_MAX:          float = -8.0   # SBC 최대 페널티
D_SBC_JUSTIFIED_CREDIT:     float = 5.0    # SBC 정당화 시 최대 크레딧
D_BUYBACK_BONUS:            float = 2.0    # 자사주 매입 보너스

#  E: 팔란티어 유사성
E_ENTERPRISE_MOAT:          float = 5.0    # 엔터프라이즈/미션크리티컬
E_GOV_COMMERCIAL_MIX:       float = 3.0    # 정부+상업 믹스
E_SECTOR_FIT:               float = 2.0    # 섹터 적합도

#  F: 밸류에이션 매력도
F_EV_GP_SCORE:              float = 4.0    # EV/GP 멀티플
F_PRICE_DISCOUNT:           float = 3.0    # 목표가 대비 할인 (괴리)
F_PRICE_FILTER:             float = 2.0    # $50 이하 보너스
F_EV_FCF_SCORE:             float = 1.0    # EV/FCF 보너스


# ══════════════════════════════════════════════════════════════════════════════
# ── 로거 ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(level: str = "INFO") -> logging.Logger:
    lg = logging.getLogger("nps")
    lg.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        lg.addHandler(h)
    return lg

logger = setup_logger()


# ══════════════════════════════════════════════════════════════════════════════
# ── 데이터 모델 ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DilutionProfile:
    """
    ATM vs SBC 희석 분리 모델.

    [리서치 핵심 인사이트]
    팔란티어는 ATM / 전통적 후속 유상증자 없이 SBC만으로 희석된 케이스.
    → SBC 희석은 총마진·FCF·성장률로 '정당화(Justified)' 가능.
    → ATM 희석은 주주가치 순감소이므로 강한 페널티.
    """
    sbc_abs:            float = 0.0
    sbc_ratio:          float = 0.0
    atm_dilution_pct:   float = 0.0
    shares_yoy_pct:     float = 0.0
    has_buyback:        bool  = False
    is_sbc_justified:   bool  = False
    justification_msg:  str   = ""

    @property
    def net_sbc_ratio_for_penalty(self) -> float:
        """Justified SBC: 55% 감면 후 페널티 계산"""
        return self.sbc_ratio * 0.45 if self.is_sbc_justified else self.sbc_ratio

    @property
    def atm_flag(self) -> str:
        if self.atm_dilution_pct >= ATM_HIGH_RISK: return "🔴 ATM고위험"
        if self.atm_dilution_pct >= ATM_WARN:       return "🟡 ATM주의"
        return "🟢 ATM낮음"

    @property
    def sbc_flag(self) -> str:
        if self.sbc_ratio >= SBC_HIGH and self.is_sbc_justified:
            return "🔵 SBC정당화"
        if self.sbc_ratio >= SBC_HIGH: return "🔴 SBC고비율"
        if self.sbc_ratio >= 0.06:     return "🟡 SBC보통"
        return "🟢 SBC낮음"


@dataclass
class ContractVisibility:
    """
    계약 가시성 모델 (Deferred Revenue 기반 프록시).

    [리서치 근거]
    팔란티어 핵심 지표: TCV 108억, Total RDV 112억, RPO 41억, contract liabilities 8.12억.
    Yahoo Finance는 RPO/TCV/Backlog 직접 미제공 → Deferred Revenue로 하한선 추정.
    """
    deferred_rev:           float = 0.0
    deferred_ratio:         float = 0.0
    deferred_yoy_growth:    float = 0.0
    score:                  float = 0.0
    label:                  str   = "N/A"

    @property
    def flag(self) -> str:
        if self.deferred_ratio >= VISIBILITY_HIGH: return "🟢 가시성높음"
        if self.deferred_ratio >= VISIBILITY_MED:  return "🟡 가시성보통"
        if self.deferred_ratio > 0:                 return "🔴 가시성낮음"
        return "⚪ 데이터없음"


@dataclass
class SectorProfile:
    """
    섹터 분류 및 섹터별 특화 점수.

    [리서치 근거] 섹터별 핵심 지표가 다름:
    소프트웨어/AI/국방: EV/NTM GP, Rule-of-40, RPO/ARR, FCF margin, SBC/Revenue
    우주 섹터:         EV/Backlog, Book-to-Bill, Launch Cadence, Capex/Revenue, Cash Runway
    양자 섹터:         Bookings Growth, Cash Runway, Burn Multiple, 기술 마일스톤
    """
    raw_sector:     str   = ""
    category:       str   = ""   # software_ai_defense / space / quantum / fintech / healthcare / other
    moat_score:     float = 0.0  # 엔터프라이즈·미션크리티컬·락인 점수 (0~5)
    gov_mix_score:  float = 0.0  # 정부+상업 믹스 점수 (0~3)
    sector_fit:     float = 0.0  # 팔란티어 섹터 유사성 (0~2)

    @property
    def label(self) -> str:
        labels = {
            "software_ai_defense": "💻 SW/AI/국방",
            "space":               "🚀 우주",
            "quantum":             "⚛ 양자",
            "fintech":             "💳 핀테크",
            "healthcare":          "🏥 헬스케어",
            "other":               "📦 기타",
        }
        return labels.get(self.category, "📦 기타")


@dataclass
class PalantirSimilarity:
    """
    팔란티어 유사성 점수 (0~10).
    [리서치 핵심] 엔터프라이즈, 미션크리티컬, 규제산업, 락인 구조가 핵심.
    """
    enterprise_score:    float = 0.0   # B2B 엔터프라이즈, 미션크리티컬 (0~5)
    gov_commercial_score: float = 0.0  # 정부+상업 믹스 (0~3)
    sector_score:        float = 0.0   # 섹터 적합도 (0~2)
    notes:               str   = ""

    @property
    def total(self) -> float:
        return min(self.enterprise_score + self.gov_commercial_score + self.sector_score, 10.0)

    @property
    def grade(self) -> str:
        t = self.total
        if t >= 8.5: return "S (팔란티어형)"
        if t >= 7.0: return "A (유사)"
        if t >= 5.0: return "B (부분유사)"
        if t >= 3.0: return "C (비슷한 요소)"
        return "D (다른 구조)"


@dataclass
class TenBaggerSignal:
    """
    10배 가능성 신호 집약.
    [리서치 근거] 팔란티어는 상장 1년 내 컨센서스(Moderate Sell) 무시하고 주가가 먼저 움직임.
    진짜 넥스트 팔란티어 조건: Street가 저평가 → 실적 서프라이즈 → 재평가 사이클.
    """
    street_discount_pct:    float = 0.0   # (목표가/현재가 - 1) * 100
    early_stage_bonus:      bool  = False  # 초기 단계 보너스
    price_filter_pass:      bool  = False  # $50 이하 통과
    high_visibility_bonus:  bool  = False  # 계약가시성 높음
    sbc_justified_bonus:    bool  = False  # SBC 정당화 (팔란티어형)
    no_atm_bonus:           bool  = False  # ATM 없음
    potential_score:        float = 0.0    # 10배 가능성 보너스 점수

    @property
    def signal_count(self) -> int:
        return sum([
            self.early_stage_bonus,
            self.price_filter_pass,
            self.high_visibility_bonus,
            self.sbc_justified_bonus,
            self.no_atm_bonus,
            self.street_discount_pct > 40,
        ])

    @property
    def signal_label(self) -> str:
        sc = self.signal_count
        if sc >= 5: return "🔥 최강 매수신호"
        if sc >= 4: return "⚡ 강한 매수신호"
        if sc >= 3: return "📈 매수신호"
        if sc >= 2: return "👀 관찰"
        return "⚪ 신호없음"


@dataclass
class TickerMetrics:
    """단일 티커의 모든 원시 재무 지표."""
    symbol:             str
    name:               str            = ""
    sector:             str            = ""
    industry:           str            = ""
    error:              Optional[str]  = None

    # 밸류
    market_cap:         Optional[float] = None
    ev:                 Optional[float] = None
    price:              Optional[float] = None
    target_price:       Optional[float] = None   # 애널리스트 목표가

    # 손익
    revenue:            Optional[float] = None
    rev_growth:         Optional[float] = None
    gross_margin:       Optional[float] = None
    op_income:          Optional[float] = None
    net_income:         Optional[float] = None

    # 현금흐름
    fcf:                Optional[float] = None
    operating_cf:       Optional[float] = None
    capex:              Optional[float] = None

    # 재무상태
    cash:               float = 0.0
    debt:               float = 0.0
    shares_out:         Optional[float] = None
    shares_out_prior:   Optional[float] = None

    # 계약가시성 프록시
    deferred_rev_cur:   float = 0.0
    deferred_rev_prior: float = 0.0

    # 내부 (SBC 저장용)
    _sbc_abs:           float = field(default=0.0, repr=False, compare=False)

    # ── 파생 지표 ──────────────────────────────────────────────────────────
    @property
    def net_cash(self) -> float: return self.cash - self.debt

    @property
    def fcf_margin(self) -> Optional[float]:
        return (self.fcf / self.revenue) if (self.fcf is not None and self.revenue) else None

    @property
    def capex_ratio(self) -> float:
        return abs(self.capex) / self.revenue if (self.capex and self.revenue) else 0.0

    @property
    def rule_of_40(self) -> float:
        return ((self.rev_growth or 0.0) + (self.fcf_margin or 0.0)) * 100

    @property
    def ev_rev(self) -> Optional[float]:
        return (self.ev / self.revenue) if (self.ev and self.revenue) else None

    @property
    def ev_gross_profit(self) -> Optional[float]:
        if self.ev and self.revenue and self.gross_margin:
            gp = self.revenue * self.gross_margin
            return self.ev / gp if gp else None
        return None

    @property
    def ev_fcf(self) -> Optional[float]:
        return (self.ev / self.fcf) if (self.ev and self.fcf and self.fcf > 0) else None

    @property
    def fcf_quality(self) -> Optional[float]:
        if self.operating_cf and self.operating_cf > 0 and self.fcf is not None:
            return self.fcf / self.operating_cf
        return None

    @property
    def cash_runway_years(self) -> Optional[float]:
        """현금 런웨이 = 순현금 / 연간 burn (음수 FCF 시)"""
        nc = self.net_cash
        f  = self.fcf or 0.0
        if nc > 0 and f < 0:
            return nc / abs(f)
        if nc > 0 and f >= 0:
            return 99.0   # 실질적으로 무한 런웨이
        return None

    @property
    def is_error(self) -> bool: return self.error is not None


@dataclass
class NPSResult:
    """NPS(Next Palantir Score) 종합 결과."""
    symbol:             str
    stage:              str   = ""
    nps_total:          float = 0.0
    nps_grade:          str   = "N/A"

    # Block 점수
    block_a:            float = 0.0   # 성장의 질
    block_b:            float = 0.0   # 마진·수익성
    block_c:            float = 0.0   # 계약 가시성
    block_d:            float = 0.0   # 희석 통제 (페널티 포함)
    block_e:            float = 0.0   # 팔란티어 유사성
    block_f:            float = 0.0   # 밸류에이션 매력도

    # 상세 점수
    a_growth_speed:     float = 0.0
    a_nrr_quality:      float = 0.0
    a_recurring:        float = 0.0
    b_gross_margin:     float = 0.0
    b_fcf:              float = 0.0
    b_rule40:           float = 0.0
    c_deferred:         float = 0.0
    c_deferred_growth:  float = 0.0
    c_cash_runway:      float = 0.0
    d_atm_penalty:      float = 0.0
    d_sbc_penalty:      float = 0.0
    d_buyback_bonus:    float = 0.0
    e_enterprise:       float = 0.0
    e_gov_mix:          float = 0.0
    e_sector_fit:       float = 0.0
    f_ev_gp:            float = 0.0
    f_price_discount:   float = 0.0
    f_price_filter:     float = 0.0

    # 서브 모델
    dilution:           DilutionProfile    = field(default_factory=DilutionProfile)
    visibility:         ContractVisibility = field(default_factory=ContractVisibility)
    similarity:         PalantirSimilarity = field(default_factory=PalantirSimilarity)
    tenbagger:          TenBaggerSignal    = field(default_factory=TenBaggerSignal)

    @property
    def key_metric(self) -> str:
        if "초기" in self.stage or "하이퍼" in self.stage:
            return "핵심: 순현금·런웨이"
        if "가속" in self.stage:
            return "핵심: EV/GP"
        return "핵심: EV/FCF"


# ══════════════════════════════════════════════════════════════════════════════
# ── 유틸 ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _cf_value(cf: pd.DataFrame, labels: list[str]) -> float:
    if cf is None or cf.empty:
        return 0.0
    for lb in labels:
        if lb in cf.index:
            v = _safe_float(cf.loc[lb].iloc[0])
            if v is not None:
                return v
    return 0.0


def _bs_value(bs: pd.DataFrame, labels: list[str], col: int = 0) -> float:
    if bs is None or bs.empty or col >= len(bs.columns):
        return 0.0
    for lb in labels:
        if lb in bs.index:
            v = _safe_float(bs.loc[lb].iloc[col])
            if v is not None:
                return v
    return 0.0


def _fmt(v: Optional[float], mult: float = 1.0,
         suffix: str = "", dec: int = 1) -> str:
    if v is None:
        return "N/A"
    try:
        f = float(v) * mult
        return f"{f:.{dec}f}{suffix}" if np.isfinite(f) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


# ══════════════════════════════════════════════════════════════════════════════
# ── 데이터 Fetch ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ticker(symbol: str) -> TickerMetrics:
    """Exponential Backoff 재시도 포함 단일 티커 Fetch."""
    symbol    = symbol.strip().upper()
    last_exc: Optional[Exception] = None

    for attempt in range(1, RETRY_MAX + 1):
        try:
            tk   = yf.Ticker(symbol)
            info = tk.info or {}

            m = TickerMetrics(
                symbol        = symbol,
                name          = info.get("shortName", symbol),
                sector        = info.get("sector", "N/A"),
                industry      = info.get("industry", "N/A"),
                market_cap    = _safe_float(info.get("marketCap")),
                ev            = _safe_float(info.get("enterpriseValue")),
                price         = _safe_float(info.get("currentPrice")
                                             or info.get("regularMarketPrice")),
                target_price  = _safe_float(info.get("targetMeanPrice")),
                revenue       = _safe_float(info.get("totalRevenue")),
                rev_growth    = _safe_float(info.get("revenueGrowth")),
                gross_margin  = _safe_float(info.get("grossMargins")),
                op_income     = _safe_float(info.get("operatingIncome")) or 0.0,
                net_income    = _safe_float(info.get("netIncomeToCommon")) or 0.0,
                fcf           = _safe_float(info.get("freeCashflow")),
                cash          = _safe_float(info.get("totalCash"))  or 0.0,
                debt          = _safe_float(info.get("totalDebt"))  or 0.0,
                capex         = _safe_float(info.get("capitalExpenditures")),
                shares_out    = _safe_float(info.get("sharesOutstanding")),
            )

            # 현금흐름표
            try:
                cf = tk.cashflow
                m.operating_cf = _cf_value(cf, [
                    "Operating Cash Flow", "Cash Flow From Operations",
                ]) or _safe_float(info.get("operatingCashflow"))
                m._sbc_abs = _cf_value(cf, [
                    "Stock Based Compensation",
                    "Share Based Compensation",
                    "stockBasedCompensation",
                ])
            except Exception as e:
                logger.debug("[%s] CF 파싱 오류: %s", symbol, e)

            # 재무상태표 (이연매출, 전년 주식수)
            try:
                bs = tk.balance_sheet
                m.deferred_rev_cur   = _bs_value(bs, [
                    "Deferred Revenue", "DeferredRevenue",
                    "Contract With Customer Liability",
                ], col=0)
                m.deferred_rev_prior = _bs_value(bs, [
                    "Deferred Revenue", "DeferredRevenue",
                    "Contract With Customer Liability",
                ], col=1)
                prior_sh = _bs_value(bs, [
                    "Common Stock", "Ordinary Shares Number",
                    "commonStockSharesOutstanding",
                ], col=1)
                if prior_sh > 1e6:
                    m.shares_out_prior = prior_sh
            except Exception as e:
                logger.debug("[%s] BS 파싱 오류: %s", symbol, e)

            # 전년 주식수 fallback
            if m.shares_out_prior is None and m.shares_out:
                try:
                    sh_hist = tk.get_shares_full(start="2022-01-01")
                    if sh_hist is not None and len(sh_hist) > 0:
                        series = sh_hist.sort_index()
                        yr_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
                        prior  = series[series.index <= yr_ago]
                        if not prior.empty:
                            m.shares_out_prior = float(prior.iloc[-1])
                except Exception as e:
                    logger.debug("[%s] shares_full 오류: %s", symbol, e)

            logger.debug("[%s] Fetch OK (시도 %d/%d)", symbol, attempt, RETRY_MAX)
            return m

        except Exception as exc:
            last_exc = exc
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning("[%s] Fetch 실패 (%d/%d): %s → %.0fs 대기",
                           symbol, attempt, RETRY_MAX, exc, wait)
            if attempt < RETRY_MAX:
                time.sleep(wait)

    logger.error("[%s] 최종 실패: %s", symbol, last_exc)
    return TickerMetrics(symbol=symbol, error=str(last_exc))


# ══════════════════════════════════════════════════════════════════════════════
# ── 희석 계산 ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def calc_dilution(m: TickerMetrics) -> DilutionProfile:
    """
    ATM / SBC 희석 분리.
    [리서치 핵심] 팔란티어 = ATM 없이 SBC만. SBC는 FCF·총마진으로 정당화 가능.
    """
    dp  = DilutionProfile()
    rev = m.revenue or 1.0

    dp.sbc_abs   = m._sbc_abs
    dp.sbc_ratio = dp.sbc_abs / rev if rev > 0 else 0.0

    if m.shares_out and m.shares_out_prior and m.shares_out_prior > 0:
        dp.shares_yoy_pct = (m.shares_out - m.shares_out_prior) / m.shares_out_prior
    else:
        dp.shares_yoy_pct = 0.0

    # ATM 추정 = 총 희석 - SBC 기여 희석
    sbc_implied = (dp.sbc_abs / m.market_cap
                   if (m.market_cap and m.market_cap > 0 and dp.sbc_abs > 0)
                   else 0.0)
    dp.atm_dilution_pct = max(0.0, dp.shares_yoy_pct - sbc_implied)

    # SBC Justified (2개 이상 조건 충족 시)
    gm = m.gross_margin or 0.0
    fm = m.fcf_margin   or 0.0
    gr = m.rev_growth   or 0.0
    reasons: list[str] = []
    if gm >= SBC_JUSTIFIED_GM:  reasons.append(f"총마진{gm*100:.0f}%")
    if fm >= SBC_JUSTIFIED_FCF: reasons.append(f"FCF{fm*100:.0f}%")
    if gr >= SBC_JUSTIFIED_GR:  reasons.append(f"성장{gr*100:.0f}%")

    if dp.sbc_ratio >= SBC_HIGH and len(reasons) >= 2:
        dp.is_sbc_justified  = True
        dp.justification_msg = "OK:" + "+".join(reasons)
    elif dp.sbc_ratio >= SBC_HIGH:
        m_list: list[str] = []
        if gm < SBC_JUSTIFIED_GM:  m_list.append(f"총마진미달({gm*100:.0f}%)")
        if fm < SBC_JUSTIFIED_FCF: m_list.append(f"FCF미달({fm*100:.0f}%)")
        if gr < SBC_JUSTIFIED_GR:  m_list.append(f"성장미달({gr*100:.0f}%)")
        dp.justification_msg = "미충족:" + ",".join(m_list)

    # 자사주 매입 신호 (FCF가 양수이고 순현금 충분)
    dp.has_buyback = (m.fcf or 0) > 0 and m.net_cash > 0

    return dp


# ══════════════════════════════════════════════════════════════════════════════
# ── 계약 가시성 계산 ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def calc_visibility(m: TickerMetrics) -> ContractVisibility:
    """Deferred Revenue 기반 계약 가시성 프록시."""
    cv = ContractVisibility()
    if not m.revenue or m.revenue <= 0:
        return cv

    cv.deferred_rev = m.deferred_rev_cur
    if m.deferred_rev_cur > 0:
        cv.deferred_ratio = m.deferred_rev_cur / m.revenue
        if m.deferred_rev_prior > 0:
            cv.deferred_yoy_growth = (
                (m.deferred_rev_cur - m.deferred_rev_prior) / m.deferred_rev_prior
            )

    # 계약가시성 C블록 기본 점수 (0~10)
    if cv.deferred_ratio >= VISIBILITY_HIGH:
        base = C_DEFERRED_RATIO
    elif cv.deferred_ratio >= VISIBILITY_MED:
        base = C_DEFERRED_RATIO * 0.6
    else:
        base = C_DEFERRED_RATIO * min(cv.deferred_ratio / VISIBILITY_MED, 1.0) * 0.4
    cv.score = _clamp(base, 0, C_DEFERRED_RATIO)

    if cv.deferred_ratio >= VISIBILITY_HIGH:
        cv.label = f"{cv.deferred_ratio*100:.0f}%(높음)"
    elif cv.deferred_ratio >= VISIBILITY_MED:
        cv.label = f"{cv.deferred_ratio*100:.0f}%(보통)"
    elif cv.deferred_ratio > 0:
        cv.label = f"{cv.deferred_ratio*100:.1f}%(낮음)"

    return cv


# ══════════════════════════════════════════════════════════════════════════════
# ── 섹터 분류 & 팔란티어 유사성 ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_QUANTUM_KEYWORDS = {"quantum", "ionq", "rigetti", "d-wave", "qbts", "rgti"}
_SPACE_KEYWORDS   = {"aerospace", "space", "satellite", "launch", "rocket",
                     "rklb", "lunr", "asts", "spce"}
_GOV_DEF_KEYWORDS = {"defense", "government", "federal", "military", "intelligence",
                     "pltr", "bbai", "bwxt", "flir"}
_AI_SW_KEYWORDS   = {"software", "platform", "saas", "data", "analytics",
                     "cybersecurity", "identity", "ai", "cloud"}

def classify_sector(m: TickerMetrics) -> SectorProfile:
    """
    섹터 자동 분류.
    [리서치 근거] 섹터별 다른 지표 적용: 소프트웨어/AI/국방, 우주, 양자, 핀테크, 헬스케어
    """
    sp = SectorProfile(raw_sector=m.sector)

    sym_low  = m.symbol.lower()
    sec_low  = (m.sector   or "").lower()
    ind_low  = (m.industry or "").lower()
    combined = sym_low + " " + sec_low + " " + ind_low

    # 1. 양자 (특정 심볼 + 키워드)
    if any(k in combined for k in _QUANTUM_KEYWORDS):
        sp.category    = "quantum"
        sp.moat_score  = 3.5   # 기술적 해자 높으나 상업화 초기
        sp.gov_mix_score = 2.0
        sp.sector_fit  = 1.5

    # 2. 우주
    elif any(k in combined for k in _SPACE_KEYWORDS) or "aerospace" in ind_low:
        sp.category    = "space"
        sp.moat_score  = 3.0
        sp.gov_mix_score = 2.5
        sp.sector_fit  = 1.5

    # 3. 소프트웨어/AI/국방 (팔란티어형 최고 유사)
    elif (m.sector in PLTR_SECTORS_SW
          or any(k in combined for k in _AI_SW_KEYWORDS | _GOV_DEF_KEYWORDS)):
        sp.category    = "software_ai_defense"
        sp.moat_score  = 5.0   # 최고 해자 가능성
        sp.gov_mix_score = 3.0
        sp.sector_fit  = 2.0

    # 4. 핀테크
    elif m.sector in PLTR_SECTORS_FIN or "fintech" in ind_low or "financial" in ind_low:
        sp.category    = "fintech"
        sp.moat_score  = 2.5
        sp.gov_mix_score = 1.0
        sp.sector_fit  = 1.0

    # 5. 헬스케어
    elif m.sector in PLTR_SECTORS_HEALTH or "health" in sec_low:
        sp.category    = "healthcare"
        sp.moat_score  = 3.0
        sp.gov_mix_score = 1.5
        sp.sector_fit  = 1.5

    else:
        sp.category    = "other"
        sp.moat_score  = 1.5
        sp.gov_mix_score = 0.5
        sp.sector_fit  = 0.5

    return sp


def calc_palantir_similarity(m: TickerMetrics, sp: SectorProfile) -> PalantirSimilarity:
    """
    팔란티어 유사성 점수 (0~10).
    [리서치 핵심] 엔터프라이즈, 미션크리티컬, 규제산업, 락인 구조 = 멀티플 정당화 근거.
    """
    ps = PalantirSimilarity()

    # ── E1: 엔터프라이즈·미션크리티컬 (0~5) ──────────────────────────────
    # 총마진 80%+ → 소프트웨어형 고마진 구조 → 기업 소프트웨어 락인 신호
    gm    = m.gross_margin or 0.0
    fm_ok = (m.fcf_margin or 0.0) >= 0.05
    notes: list[str] = []

    if gm >= 0.80:
        ps.enterprise_score = 5.0
        notes.append("총마진80%+")
    elif gm >= 0.65:
        ps.enterprise_score = 3.5
        notes.append("총마진65%+")
    elif gm >= 0.50:
        ps.enterprise_score = 2.0
        notes.append("총마진50%+")
    else:
        ps.enterprise_score = max(0.5, gm * 5)

    if fm_ok:
        ps.enterprise_score = min(ps.enterprise_score + 0.5, 5.0)

    # ── E2: 정부+상업 믹스 (0~3) ─────────────────────────────────────────
    # 섹터 프로파일 기반 + 정부 계약 키워드 보너스
    ps.gov_commercial_score = sp.gov_mix_score
    if any(k in (m.industry or "").lower() for k in _GOV_DEF_KEYWORDS):
        ps.gov_commercial_score = min(ps.gov_commercial_score + 0.5, 3.0)
        notes.append("국방/정부계약")

    # ── E3: 섹터 적합도 (0~2) ────────────────────────────────────────────
    ps.sector_score = sp.sector_fit
    ps.notes = " | ".join(notes)

    return ps


# ══════════════════════════════════════════════════════════════════════════════
# ── 단계 분류 (5단계) ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def classify_stage(m: TickerMetrics) -> str:
    rev    = m.revenue    or 0.0
    growth = m.rev_growth or 0.0
    fm     = m.fcf_margin or 0.0
    op_inc = m.op_income  or 0.0

    if growth >= STAGE_HYPER_GROWTH_MIN and fm < 0.05:
        return "⚡ 하이퍼성장"
    if rev < STAGE_EARLY_REV_MAX or op_inc < 0:
        return "🌱 초기단계"
    if growth >= STAGE_HIGH_GROWTH_MIN and fm < STAGE_MATURE_FCF_MIN:
        return "🚀 성장가속"
    if growth >= 0.10 and fm >= STAGE_MATURE_FCF_MIN:
        return "💎 성숙고성장"
    return "🏛 성숙전환"


# ══════════════════════════════════════════════════════════════════════════════
# ── NPS 점수 엔진 ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def calc_nps(m: TickerMetrics,
             dp: DilutionProfile,
             cv: ContractVisibility,
             ps: PalantirSimilarity,
             stage: str) -> NPSResult:
    """
    NPS (Next Palantir Score) 0~100 산출.

    Block A (25pt): 성장의 질
    Block B (20pt): 마진·수익성
    Block C (20pt): 계약 가시성
    Block D (15pt): 희석 통제 (페널티)
    Block E (10pt): 팔란티어 유사성
    Block F (10pt): 밸류에이션 매력도

    ─────────────────────────────────────────────────────
    [리서치 핵심 반영]
    - ATM: 무조건 강한 페널티 (초기 단계 50% 경감)
    - SBC: Justified 시 크레딧 부여 (팔란티어형)
    - 계약가시성: Deferred Rev + YoY Growth 기반
    - 팔란티어 유사성: 총마진 + 섹터 + 정부믹스
    - Street 괴리: 목표가 vs 현재가 (클수록 재평가 여지)
    """
    r = NPSResult(symbol=m.symbol, stage=stage,
                  dilution=dp, visibility=cv, similarity=ps)

    g  = m.rev_growth   or 0.0
    gm = m.gross_margin or 0.0
    fm = m.fcf_margin   or 0.0
    r40 = m.rule_of_40
    fq  = m.fcf_quality

    # ══ Block A: 성장의 질 (25pt) ══════════════════════════════════════════
    # A1. 성장 속도 (0~10)
    r.a_growth_speed = _clamp(g / 0.50, 0, 1) * A_GROWTH_SPEED
    if g >= 0.80: r.a_growth_speed = min(r.a_growth_speed * 1.1, A_GROWTH_SPEED)

    # A2. NRR 프록시 — 가시성 성장이 매출 성장보다 빠르면 고객 확장 신호 (0~8)
    #     Deferred Rev YoY > 매출 YoY → NRR 좋은 편 추정
    if cv.deferred_yoy_growth > 0:
        nrr_proxy = cv.deferred_yoy_growth - g   # 양수 = 계약 쌓임
        r.a_nrr_quality = _clamp(0.5 + nrr_proxy * 5, 0, 1) * A_NRR_QUALITY
    else:
        r.a_nrr_quality = _clamp(g / 0.40, 0, 1) * A_NRR_QUALITY * 0.5

    # A3. 반복매출 프록시 — Deferred > 0 이면 구독/계약 기반 매출 (0~7)
    if cv.deferred_ratio >= VISIBILITY_HIGH:
        r.a_recurring = A_RECURRING_QUALITY
    elif cv.deferred_ratio >= VISIBILITY_MED:
        r.a_recurring = A_RECURRING_QUALITY * 0.7
    elif cv.deferred_ratio > 0:
        r.a_recurring = A_RECURRING_QUALITY * 0.3
    else:
        # 반복매출 근거 없음 → 성장률로 일부 보완
        r.a_recurring = _clamp(g / 0.50, 0, 0.5) * A_RECURRING_QUALITY

    r.block_a = r.a_growth_speed + r.a_nrr_quality + r.a_recurring

    # ══ Block B: 마진·수익성 (20pt) ════════════════════════════════════════
    # B1. 총마진 (0~10): 팔란티어 82%가 기준
    r.b_gross_margin = _clamp(gm / 0.82, 0, 1) * B_GROSS_MARGIN

    # B2. FCF마진 (0~6): 20%+가 만점
    r.b_fcf = _clamp(max(fm, 0) / 0.20, 0, 1) * B_FCF_MARGIN

    # B3. Rule-of-40 (0~4): 팔란티어 106% 기준
    r.b_rule40 = _clamp(max(r40, 0) / 60.0, 0, 1) * B_RULE40

    r.block_b = r.b_gross_margin + r.b_fcf + r.b_rule40

    # ══ Block C: 계약 가시성 (20pt) ════════════════════════════════════════
    # C1. Deferred/매출 비율 (0~10)
    r.c_deferred = cv.score   # 이미 0~10 범위로 계산됨

    # C2. Deferred YoY 성장 (0~5): 매출보다 빠를수록 선주문 쌓임
    if cv.deferred_yoy_growth > 0.40:
        r.c_deferred_growth = C_DEFERRED_GROWTH
    elif cv.deferred_yoy_growth > 0.20:
        r.c_deferred_growth = C_DEFERRED_GROWTH * 0.7
    elif cv.deferred_yoy_growth > 0.05:
        r.c_deferred_growth = C_DEFERRED_GROWTH * 0.4

    # C3. 현금 런웨이 (0~5): 초기 단계에서 특히 중요
    runway = m.cash_runway_years
    if runway is not None:
        if runway >= 99:
            r.c_cash_runway = C_CASH_RUNWAY            # 무한 런웨이 (FCF 양수)
        elif runway >= 5:
            r.c_cash_runway = C_CASH_RUNWAY * 0.8
        elif runway >= 3:
            r.c_cash_runway = C_CASH_RUNWAY * 0.5
        elif runway >= 1.5:
            r.c_cash_runway = C_CASH_RUNWAY * 0.2

    r.block_c = r.c_deferred + r.c_deferred_growth + r.c_cash_runway

    # ══ Block D: 희석 통제 (15pt, 페널티 포함) ═══════════════════════════════
    # 초기/하이퍼 단계는 자금조달 불가피 → ATM 페널티 50% 경감
    is_early   = any(s in stage for s in ["초기", "하이퍼"])
    atm_disc   = 0.50 if is_early else 1.0

    # D1. ATM 페널티 (0 ~ -15)
    r.d_atm_penalty = -(
        _clamp(dp.atm_dilution_pct / ATM_HIGH_RISK, 0, 1)
        * abs(D_ATM_PENALTY_MAX) * atm_disc
    )

    # D2. SBC 페널티 (0 ~ -8), Justified 시 크레딧
    base_sbc = -(
        _clamp(dp.net_sbc_penalty_ratio / 0.15, 0, 1) * abs(D_SBC_PENALTY_MAX)
    )
    if dp.is_sbc_justified:
        sbc_credit = _clamp(
            ((gm - SBC_JUSTIFIED_GM)  / 0.10) * (D_SBC_JUSTIFIED_CREDIT * 0.5)
            + ((fm - SBC_JUSTIFIED_FCF) / 0.10) * (D_SBC_JUSTIFIED_CREDIT * 0.5),
            0, D_SBC_JUSTIFIED_CREDIT,
        )
        r.d_sbc_penalty = base_sbc + sbc_credit
    else:
        r.d_sbc_penalty = base_sbc

    # D3. 자사주 매입 보너스 (+2)
    r.d_buyback_bonus = D_BUYBACK_BONUS if dp.has_buyback else 0.0

    # Block D = 15 (기본) + 페널티 + 보너스
    # 페널티가 없으면 15점, 최대 페널티 시 0점도 가능
    BASE_D = 15.0
    r.block_d = _clamp(BASE_D + r.d_atm_penalty + r.d_sbc_penalty + r.d_buyback_bonus,
                       0, BASE_D + D_BUYBACK_BONUS)

    # ══ Block E: 팔란티어 유사성 (10pt) ════════════════════════════════════
    r.e_enterprise  = _clamp(ps.enterprise_score,     0, E_ENTERPRISE_MOAT)
    r.e_gov_mix     = _clamp(ps.gov_commercial_score, 0, E_GOV_COMMERCIAL_MIX)
    r.e_sector_fit  = _clamp(ps.sector_score,         0, E_SECTOR_FIT)
    r.block_e = r.e_enterprise + r.e_gov_mix + r.e_sector_fit

    # ══ Block F: 밸류에이션 매력도 (10pt) ══════════════════════════════════
    # F1. EV/GP 매력도 (0~4): 낮을수록 저평가
    ev_gp = m.ev_gross_profit
    if ev_gp is not None and ev_gp > 0:
        if   ev_gp <= 5:   r.f_ev_gp = F_EV_GP_SCORE
        elif ev_gp <= 10:  r.f_ev_gp = F_EV_GP_SCORE * 0.75
        elif ev_gp <= 20:  r.f_ev_gp = F_EV_GP_SCORE * 0.50
        elif ev_gp <= 40:  r.f_ev_gp = F_EV_GP_SCORE * 0.25
        else:              r.f_ev_gp = 0.0

    # F2. Street 목표가 vs 현재가 괴리 (0~3)
    # [리서치 핵심] 팔란티어: Street가 항상 뒤처짐 → 괴리가 클수록 재평가 여지
    if m.target_price and m.price and m.price > 0:
        discount = (m.target_price / m.price) - 1.0
        if   discount >= 0.80: r.f_price_discount = F_PRICE_DISCOUNT
        elif discount >= 0.50: r.f_price_discount = F_PRICE_DISCOUNT * 0.75
        elif discount >= 0.30: r.f_price_discount = F_PRICE_DISCOUNT * 0.50
        elif discount >= 0.10: r.f_price_discount = F_PRICE_DISCOUNT * 0.25
        else:                  r.f_price_discount = 0.0
    else:
        # 목표가 데이터 없으면 EV/Rev 기반 보완
        ev_rev = m.ev_rev
        if ev_rev is not None and ev_rev > 0:
            if ev_rev <= 3: r.f_price_discount = F_PRICE_DISCOUNT
            elif ev_rev <= 6: r.f_price_discount = F_PRICE_DISCOUNT * 0.5

    # F3. $50 이하 가격 보너스 (0~2): [리서치] 저가 + 저평가 조합
    if m.price and m.price <= PRICE_FILTER_MAX:
        r.f_price_filter = F_PRICE_FILTER
    elif m.price and m.price <= PRICE_FILTER_MAX * 2:
        r.f_price_filter = F_PRICE_FILTER * 0.5

    r.block_f = r.f_ev_gp + r.f_price_discount + r.f_price_filter

    # ══ NPS 합산 ════════════════════════════════════════════════════════════
    raw = r.block_a + r.block_b + r.block_c + r.block_d + r.block_e + r.block_f
    r.nps_total = round(_clamp(raw, 0, 100), 1)
    r.nps_grade = _nps_grade(r.nps_total)

    # ══ 10배 신호 ═══════════════════════════════════════════════════════════
    tb = TenBaggerSignal()
    if m.target_price and m.price and m.price > 0:
        tb.street_discount_pct = ((m.target_price / m.price) - 1.0) * 100
    tb.early_stage_bonus    = any(s in stage for s in ["초기", "하이퍼"])
    tb.price_filter_pass    = bool(m.price and m.price <= PRICE_FILTER_MAX)
    tb.high_visibility_bonus = cv.deferred_ratio >= VISIBILITY_HIGH
    tb.sbc_justified_bonus   = dp.is_sbc_justified
    tb.no_atm_bonus          = dp.atm_dilution_pct < ATM_WARN
    tb.potential_score       = r.nps_total
    r.tenbagger = tb

    return r


def _nps_grade(score: float) -> str:
    if score >= 85: return "S+ (10배 후보)"
    if score >= 75: return "S  (강력매수)"
    if score >= 65: return "A  (매수)"
    if score >= 52: return "B  (관찰)"
    if score >= 38: return "C  (중립)"
    return             "D  (제외)"


# ══════════════════════════════════════════════════════════════════════════════
# ── 결과 행 생성 ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_row(m: TickerMetrics, r: NPSResult, sp: SectorProfile) -> dict:
    dp = r.dilution
    cv = r.visibility
    ps = r.similarity
    tb = r.tenbagger

    price = m.price or 0.0
    tp    = m.target_price
    upside = f"{((tp / price - 1) * 100):.0f}%" if (tp and price > 0) else "N/A"

    return {
        # ── 식별 ──────────────────────────────────────────────────────────
        "티커":             m.symbol,
        "기업명":           m.name,
        "섹터분류":         sp.label,
        "섹터(원본)":       m.sector,
        "단계":             r.stage,

        # ── NPS 종합 ──────────────────────────────────────────────────────
        "NPS점수":          r.nps_total,
        "NPS등급":          r.nps_grade,
        "10배신호":         tb.signal_label,
        "신호수":           tb.signal_count,
        "팔란티어유사성":   ps.grade,

        # ── 가격 & 밸류 ───────────────────────────────────────────────────
        "현재가($)":        _fmt(m.price),
        "목표가($)":        _fmt(tp),
        "업사이드":         upside,
        "시가총액(B$)":     _fmt(m.market_cap, 1/1e9),
        "EV(B$)":           _fmt(m.ev,         1/1e9),

        # ── 성장 & 마진 ───────────────────────────────────────────────────
        "매출성장(%)":      _fmt(m.rev_growth,   100),
        "총마진(%)":        _fmt(m.gross_margin, 100),
        "FCF마진(%)":       _fmt(m.fcf_margin,   100),
        "Rule-of-40":       _fmt(m.rule_of_40,   1, "", 1),

        # ── EV 멀티플 ─────────────────────────────────────────────────────
        "EV/매출":          _fmt(m.ev_rev,           suffix="x"),
        "EV/GP":            _fmt(m.ev_gross_profit,   suffix="x"),
        "EV/FCF":           _fmt(m.ev_fcf,            suffix="x"),

        # ── 희석 구조 (핵심) ──────────────────────────────────────────────
        "SBC/매출(%)":      _fmt(dp.sbc_ratio,         100),
        "SBC구분":          dp.sbc_flag,
        "ATM희석(%)":       _fmt(dp.atm_dilution_pct,  100),
        "ATM구분":          dp.atm_flag,
        "주식수YoY(%)":     _fmt(dp.shares_yoy_pct,    100),
        "SBC정당화":        dp.justification_msg if dp.sbc_ratio >= SBC_HIGH else "-",

        # ── 계약 가시성 (핵심) ────────────────────────────────────────────
        "계약가시성":        cv.flag,
        "이연매출/매출":     cv.label,
        "이연매출YoY(%)":   _fmt(cv.deferred_yoy_growth, 100),

        # ── 재무 안전성 ───────────────────────────────────────────────────
        "순현금(B$)":       _fmt(m.net_cash,       1/1e9),
        "런웨이(년)":       _fmt(m.cash_runway_years, 1, "", 1),
        "Capex/매출(%)":    _fmt(m.capex_ratio,     100),

        # ── Block 점수 내역 ───────────────────────────────────────────────
        "_A_성장질":        round(r.block_a, 1),
        "_B_마진수익":      round(r.block_b, 1),
        "_C_계약가시성":    round(r.block_c, 1),
        "_D_희석통제":      round(r.block_d, 1),
        "_E_팔란티어유사":  round(r.block_e, 1),
        "_F_밸류매력":      round(r.block_f, 1),

        # ── 세부 점수 ─────────────────────────────────────────────────────
        "_a1_성장속도":     round(r.a_growth_speed, 1),
        "_a2_NRR":          round(r.a_nrr_quality, 1),
        "_a3_반복매출":     round(r.a_recurring, 1),
        "_b1_총마진":       round(r.b_gross_margin, 1),
        "_b2_FCF":          round(r.b_fcf, 1),
        "_b3_rule40":       round(r.b_rule40, 1),
        "_c1_Deferred":     round(r.c_deferred, 1),
        "_c2_DR성장":       round(r.c_deferred_growth, 1),
        "_c3_런웨이":       round(r.c_cash_runway, 1),
        "_d_ATM페널티":     round(r.d_atm_penalty, 1),
        "_d_SBC페널티":     round(r.d_sbc_penalty, 1),
        "_d_바이백보너스":  round(r.d_buyback_bonus, 1),
        "_e1_엔터프라이즈": round(r.e_enterprise, 1),
        "_e2_정부믹스":     round(r.e_gov_mix, 1),
        "_e3_섹터적합":     round(r.e_sector_fit, 1),
        "_f1_EVGP":         round(r.f_ev_gp, 1),
        "_f2_목표가괴리":   round(r.f_price_discount, 1),
        "_f3_가격필터":     round(r.f_price_filter, 1),
        "_Street업사이드%": round(tb.street_discount_pct, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── 단일 심볼 파이프라인 ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def analyze_symbol(symbol: str) -> dict:
    """Fetch → 희석 → 가시성 → 섹터 → 유사성 → NPS → Row 전체 파이프라인."""
    m = fetch_ticker(symbol)
    if m.is_error:
        return {"티커": symbol, "기업명": "ERROR", "단계": "❌",
                "NPS점수": 0.0, "NPS등급": "F", "_error": m.error}

    dp    = calc_dilution(m)
    cv    = calc_visibility(m)
    sp    = classify_sector(m)
    ps    = calc_palantir_similarity(m, sp)
    stage = classify_stage(m)
    r     = calc_nps(m, dp, cv, ps, stage)
    return build_row(m, r, sp)


# ══════════════════════════════════════════════════════════════════════════════
# ── 배치 처리 ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def analyze_batch(symbols: list[str],
                  workers: int = DEFAULT_WORKERS,
                  verbose: bool = True) -> pd.DataFrame:
    total  = len(symbols)
    order  = {sym: i for i, sym in enumerate(symbols)}
    done:  dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(analyze_symbol, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                logger.error("[%s] 예기치 않은 오류: %s", sym, exc)
                row = {"티커": sym, "기업명": "ERROR", "단계": "❌",
                       "NPS점수": 0.0, "NPS등급": "F", "_error": str(exc)}
            done[sym] = row
            if verbose:
                cnt = len(done)
                err = row.get("_error", "")
                if err:
                    print(f"  [{cnt:>3}/{total}] {sym:<8} ⚠  {str(err)[:50]}")
                else:
                    print(
                        f"  [{cnt:>3}/{total}] {sym:<8} ✓  "
                        f"{row.get('단계','?'):<14} "
                        f"NPS={row.get('NPS점수',0):<5} "
                        f"{row.get('NPS등급','?'):<18} "
                        f"신호={row.get('10배신호','')}"
                    )
            time.sleep(FETCH_DELAY_SEC / max(workers, 1))

    sorted_rows = sorted(done.values(),
                         key=lambda r: order.get(r.get("티커",""), 9999))
    df = pd.DataFrame(sorted_rows)
    if "_error" in df.columns:
        df = df.drop(columns=["_error"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ── 콘솔 리포트 ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_MAIN_COLS = [
    "티커", "섹터분류", "단계", "NPS점수", "NPS등급",
    "10배신호", "매출성장(%)", "총마진(%)", "FCF마진(%)", "Rule-of-40",
    "EV/GP", "SBC구분", "ATM구분", "계약가시성", "업사이드",
]
_DILUTION_COLS = [
    "티커", "SBC/매출(%)", "SBC구분", "ATM희석(%)", "ATM구분",
    "주식수YoY(%)", "총마진(%)", "FCF마진(%)", "매출성장(%)", "SBC정당화",
]
_VISIBILITY_COLS = [
    "티커", "계약가시성", "이연매출/매출", "이연매출YoY(%)",
    "EV/GP", "EV/FCF", "FCF마진(%)", "NPS등급",
]
_BLOCK_COLS = [
    "티커", "NPS점수", "NPS등급",
    "_A_성장질", "_B_마진수익", "_C_계약가시성", "_D_희석통제",
    "_E_팔란티어유사", "_F_밸류매력",
]
_TENBAGGER_COLS = [
    "티커", "NPS등급", "NPS점수", "10배신호", "신호수",
    "팔란티어유사성", "현재가($)", "목표가($)", "업사이드",
    "SBC정당화", "이연매출/매출", "단계",
]


def print_report(df: pd.DataFrame) -> None:
    sdf = df.sort_values("NPS점수", ascending=False).reset_index(drop=True)
    W   = 140

    print("\n" + "═" * W)
    print(f"  🎯  넥스트 팔란티어 스캐너 v3.0  ──  "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M')}  "
          f"({len(sdf)}개 종목 분석)")
    print("═" * W)

    # ── 메인 테이블 ───────────────────────────────────────────────────────
    pc = [c for c in _MAIN_COLS if c in sdf.columns]
    try:
        print(sdf[pc].to_string(index=False, max_colwidth=20))
    except Exception:
        print(sdf[pc].to_string(index=False))
    print("═" * W)

    # ── 10배 후보 TOP ─────────────────────────────────────────────────────
    tc = [c for c in _TENBAGGER_COLS if c in sdf.columns]
    top = sdf[sdf["NPS등급"].str.startswith(("S+", "S ", "A"), na=False)][tc]
    if not top.empty:
        print("\n【 🔥 S+/S/A 등급 — 10배 후보 】")
        print(top.to_string(index=False))

    # ── 섹터별 분포 ───────────────────────────────────────────────────────
    print("\n【 섹터별 분포 】")
    for sec, grp in sdf.groupby("섹터분류", sort=False):
        avg = grp["NPS점수"].mean()
        tks = ", ".join(grp["티커"].tolist())
        print(f"  {sec:<14}  {len(grp):>2}개  평균NPS={avg:.0f}  →  {tks}")

    # ── 단계별 분포 ───────────────────────────────────────────────────────
    print("\n【 성장 단계별 분포 】")
    for stage, grp in sdf.groupby("단계", sort=False):
        print(f"  {stage:<14}  {len(grp):>2}개  →  {', '.join(grp['티커'].tolist())}")

    # ── 희석 구조 분석 ────────────────────────────────────────────────────
    dc = [c for c in _DILUTION_COLS if c in sdf.columns]
    print("\n【 희석 구조 분석 (ATM vs SBC — 팔란티어 연구 기반) 】")
    try:
        justified = sdf[sdf["SBC정당화"].str.startswith("OK:", na=False)]
        warn_sbc  = sdf[
            sdf["SBC구분"].str.contains("고비율", na=False)
            & ~sdf["SBC정당화"].str.startswith("OK:", na=False)
        ]
        warn_atm  = sdf[sdf["ATM구분"].str.contains("주의|고위험", na=False)]

        if not justified.empty:
            print("\n  ▶ SBC 고비율이지만 펀더멘털로 정당화 (팔란티어형 구조):")
            print(justified[dc].to_string(index=False))
        if not warn_sbc.empty:
            print("\n  ▶ ⚠ SBC 고비율 / 미정당화 (투자전 확인 필요):")
            print(warn_sbc[dc].to_string(index=False))
        if not warn_atm.empty:
            print("\n  ▶ 🔴 ATM 희석 위험 (순수 주주가치 희석):")
            print(warn_atm[dc].to_string(index=False))
    except Exception as e:
        logger.debug("희석 출력 오류: %s", e)

    # ── 계약 가시성 ───────────────────────────────────────────────────────
    vc = [c for c in _VISIBILITY_COLS if c in sdf.columns]
    print("\n【 계약 가시성 (Deferred Revenue 기반, RPO·ARR·TCV 프록시) 】")
    print(sdf[vc].to_string(index=False))

    # ── Block 점수 내역 ───────────────────────────────────────────────────
    bc = [c for c in _BLOCK_COLS if c in sdf.columns]
    if bc:
        print("\n【 NPS Block 점수 내역 (A=성장질/B=마진/C=가시성/D=희석/E=유사성/F=밸류) 】")
        print(sdf[bc].to_string(index=False))

    print()


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI ───────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="넥스트 팔란티어 스캐너 v3.0 — 10배 종목 발굴 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tickers",  nargs="+", metavar="SYM",
                   help="직접 티커  예: SAIL WAY PLTR SNOW RKLB IONQ")
    p.add_argument("--file",     default=DEFAULT_TICKER_FILE, metavar="PATH",
                   help=f"티커 목록 파일 (기본: {DEFAULT_TICKER_FILE})")
    p.add_argument("--output",   default=DEFAULT_OUTPUT_FILE, metavar="PATH",
                   help=f"CSV 저장 경로 (기본: {DEFAULT_OUTPUT_FILE})")
    p.add_argument("--workers",  type=int, default=DEFAULT_WORKERS, metavar="N",
                   help=f"동시 Fetch 워커 수 (기본: {DEFAULT_WORKERS})")
    p.add_argument("--no-csv",   action="store_true")
    p.add_argument("--no-detail",action="store_true",
                   help="CSV에서 세부 점수(_a1_/_b1_/...) 컬럼 제외")
    p.add_argument("--quiet",    action="store_true")
    p.add_argument("--log-level",default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


def load_symbols(fpath: Path) -> list[str]:
    if not fpath.exists():
        logger.error("파일 없음: %s", fpath)
        sys.exit(1)
    syms: list[str] = []
    for line in fpath.read_text(encoding="utf-8").splitlines():
        c = line.split("#")[0].strip().upper()
        if c:
            syms.append(c)
    return syms


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    symbols = (
        [s.strip().upper() for s in args.tickers if s.strip()]
        if args.tickers else load_symbols(Path(args.file))
    )
    if not symbols:
        logger.error("분석할 티커가 없습니다.")
        sys.exit(1)

    seen: set[str] = set()
    unique = [s for s in symbols if not (s in seen or seen.add(s))]  # type: ignore

    print(f"\n🎯  넥스트 팔란티어 스캐너 시작 — {len(unique)}개 티커  "
          f"(워커: {args.workers})")
    print(f"    {', '.join(unique[:15])}{'...' if len(unique)>15 else ''}\n")

    t0 = time.monotonic()
    df = analyze_batch(unique, workers=args.workers, verbose=not args.quiet)
    elapsed = time.monotonic() - t0

    print(f"\n⏱  Fetch 완료: {elapsed:.1f}초")
    print_report(df)

    if not args.no_csv:
        out = df.sort_values("NPS점수", ascending=False)
        if args.no_detail:
            drop = [c for c in out.columns
                    if c.startswith(("_a","_b","_c","_d","_e","_f","_S"))]
            out = out.drop(columns=drop, errors="ignore")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"💾  CSV 저장: {out_path}\n")

    print(f"✅  완료  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
