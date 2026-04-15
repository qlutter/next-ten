#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고성장 주식 밸류에이션 분석기 v2.0  (Production-Ready)
═══════════════════════════════════════════════════════════════════════════════
핵심 공식:
  Score = (고성장 × 높은총마진 × 계약가시성 × 강한현금창출)
          ÷ (ATM희석 × SBC희석_net × Capex부담 × 기술/실행실패확률)

[v2.0 주요 개선]
  ① ATM 희석 vs SBC 희석 분리 계산
     - ATM : 주식 신규 발행 → 무조건 페널티 (가치 희석)
     - SBC : 비현금 비용 → 총마진·FCF·성장률로 정당화 가능 (Justified SBC)
  ② 계약가시성 프록시 (Deferred Revenue 기반) 구현
  ③ 동시 Fetch + Exponential Backoff 재시도
  ④ 구조화 로깅 (logging 모듈)
  ⑤ 타입 안전 Dataclass 모델
  ⑥ FCF 품질 비율, 영업레버리지 측정
  ⑦ 단계 5단계 세분화

Usage:
  python high_growth_valuation.py                   # ticker.txt 자동 읽기
  python high_growth_valuation.py --tickers PLTR SNOW NVDA RKLB
  python high_growth_valuation.py --file my.txt --output out.csv --workers 4
  python high_growth_valuation.py --tickers PLTR --log-level DEBUG
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
# 설정 상수
# ══════════════════════════════════════════════════════════════════════════════
FETCH_DELAY_SEC:        float = 0.5
RETRY_MAX:              int   = 3
RETRY_BASE_DELAY:       float = 2.0
DEFAULT_WORKERS:        int   = 4
DEFAULT_TICKER_FILE:    str   = "ticker.txt"
DEFAULT_OUTPUT_FILE:    str   = "valuation_results.csv"

# 단계 분류
STAGE_EARLY_REV_MAX:    float = 5e8
STAGE_HYPER_GROWTH_MIN: float = 0.50
STAGE_HIGH_GROWTH_MIN:  float = 0.25
STAGE_MATURE_FCF_MIN:   float = 0.10

# 희석 임계값
ATM_DILUTION_WARN:          float = 0.05
SBC_HIGH_THRESHOLD:         float = 0.12
SBC_JUSTIFIED_GM_MIN:       float = 0.70
SBC_JUSTIFIED_FCF_MIN:      float = 0.08
SBC_JUSTIFIED_GR_MIN:       float = 0.20

# 계약가시성
VISIBILITY_HIGH_RATIO:  float = 0.20
VISIBILITY_MED_RATIO:   float = 0.08

# 배점
SCORE_MAX_GROWTH:           float = 35.0
SCORE_MAX_GROSS_MARGIN:     float = 20.0
SCORE_MAX_FCF:              float = 18.0
SCORE_MAX_RULE40:           float = 10.0
SCORE_MAX_VISIBILITY:       float = 12.0
SCORE_MAX_FCF_QUALITY:      float = 5.0
SCORE_PEN_ATM:              float = 20.0
SCORE_PEN_SBC:              float = 15.0
SCORE_PEN_SBC_JUSTIFIED:    float = 6.0
SCORE_PEN_CAPEX:            float = 8.0


# ══════════════════════════════════════════════════════════════════════════════
# 로거
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(level: str = "INFO") -> logging.Logger:
    lg = logging.getLogger("hgv")
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
# 데이터 모델
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DilutionProfile:
    """
    희석 구조 분석.

    ATM (At-The-Market) : 주식 신규 공모로 인한 희석
        회사 현금 조달은 되지만 기존 주주 지분 희석 → 강한 페널티

    SBC (Stock-Based Compensation) : 임직원 보상 주식
        비현금 비용으로 인재 확보 수단. 팔란티어형:
        SBC 비율이 높아도 FCF·총마진·성장률이 이를 정당화하면 페널티 감면.
    """
    sbc_abs:            float = 0.0
    sbc_ratio:          float = 0.0
    atm_dilution_pct:   float = 0.0
    shares_yoy_pct:     float = 0.0
    is_sbc_justified:   bool  = False
    justification_msg:  str   = ""

    @property
    def net_sbc_penalty_ratio(self) -> float:
        """Justified SBC: 페널티 55% 감면"""
        return self.sbc_ratio * 0.45 if self.is_sbc_justified else self.sbc_ratio

    @property
    def atm_flag(self) -> str:
        if self.atm_dilution_pct >= 0.08: return "ATM 고위험"
        if self.atm_dilution_pct >= 0.04: return "ATM 주의"
        return "ATM 낮음"

    @property
    def sbc_flag(self) -> str:
        if self.sbc_ratio >= SBC_HIGH_THRESHOLD and self.is_sbc_justified:
            return "SBC 고비율(정당화)"
        if self.sbc_ratio >= SBC_HIGH_THRESHOLD:
            return "SBC 고비율"
        if self.sbc_ratio >= 0.06:
            return "SBC 보통"
        return "SBC 낮음"


@dataclass
class ContractVisibility:
    """
    계약 가시성 프록시 (Deferred Revenue 기반).
    Yahoo Finance는 RPO/Backlog 미제공 → 이연매출로 대체.
    """
    deferred_rev:           float = 0.0
    deferred_ratio:         float = 0.0
    deferred_yoy_growth:    float = 0.0
    score:                  float = 0.0
    label:                  str   = "N/A"

    @property
    def flag(self) -> str:
        if self.deferred_ratio >= VISIBILITY_HIGH_RATIO: return "가시성 높음"
        if self.deferred_ratio >= VISIBILITY_MED_RATIO:  return "가시성 보통"
        if self.deferred_ratio > 0:                       return "가시성 낮음"
        return "데이터 없음"


@dataclass
class TickerMetrics:
    """단일 티커의 원시 재무 지표."""
    symbol:             str
    name:               str            = ""
    sector:             str            = ""
    error:              Optional[str]  = None

    market_cap:         Optional[float] = None
    ev:                 Optional[float] = None
    price:              Optional[float] = None

    revenue:            Optional[float] = None
    rev_growth:         Optional[float] = None
    gross_margin:       Optional[float] = None
    op_income:          Optional[float] = None
    net_income:         Optional[float] = None

    fcf:                Optional[float] = None
    operating_cf:       Optional[float] = None
    capex:              Optional[float] = None

    cash:               float = 0.0
    debt:               float = 0.0
    shares_out:         Optional[float] = None
    shares_out_prior:   Optional[float] = None

    deferred_rev_cur:   float = 0.0
    deferred_rev_prior: float = 0.0

    # SBC 내부 저장 (DilutionProfile 계산용)
    _sbc_abs:           float = field(default=0.0, repr=False, compare=False)

    @property
    def net_cash(self) -> float:
        return self.cash - self.debt

    @property
    def fcf_margin(self) -> Optional[float]:
        if self.fcf is not None and self.revenue:
            return self.fcf / self.revenue
        return None

    @property
    def capex_ratio(self) -> float:
        if self.capex and self.revenue:
            return abs(self.capex) / self.revenue
        return 0.0

    @property
    def rule_of_40(self) -> float:
        return ((self.rev_growth or 0.0) + (self.fcf_margin or 0.0)) * 100

    @property
    def ev_rev(self) -> Optional[float]:
        if self.ev and self.revenue: return self.ev / self.revenue
        return None

    @property
    def ev_gross_profit(self) -> Optional[float]:
        if self.ev and self.revenue and self.gross_margin:
            gp = self.revenue * self.gross_margin
            return self.ev / gp if gp else None
        return None

    @property
    def ev_fcf(self) -> Optional[float]:
        if self.ev and self.fcf and self.fcf > 0:
            return self.ev / self.fcf
        return None

    @property
    def fcf_quality(self) -> Optional[float]:
        """FCF / 영업현금흐름 — 1.0 근방이 가장 건전."""
        if self.operating_cf and self.operating_cf > 0 and self.fcf is not None:
            return self.fcf / self.operating_cf
        return None

    @property
    def is_error(self) -> bool:
        return self.error is not None


@dataclass
class ScoringResult:
    """종합 점수 및 구성 요소."""
    symbol:             str
    stage:              str   = ""
    total_score:        float = 0.0
    grade:              str   = "N/A"

    growth_pt:          float = 0.0
    gross_margin_pt:    float = 0.0
    fcf_pt:             float = 0.0
    rule40_pt:          float = 0.0
    visibility_pt:      float = 0.0
    fcf_quality_pt:     float = 0.0

    atm_penalty:        float = 0.0
    sbc_penalty:        float = 0.0
    capex_penalty:      float = 0.0

    dilution:           DilutionProfile    = field(default_factory=DilutionProfile)
    visibility:         ContractVisibility = field(default_factory=ContractVisibility)

    @property
    def numerator_sum(self) -> float:
        return (self.growth_pt + self.gross_margin_pt + self.fcf_pt
                + self.rule40_pt + self.visibility_pt + self.fcf_quality_pt)

    @property
    def penalty_sum(self) -> float:
        return self.atm_penalty + self.sbc_penalty + self.capex_penalty


# ══════════════════════════════════════════════════════════════════════════════
# 유틸
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
         suffix: str = "", decimals: int = 1) -> str:
    """None·NaN·inf 안전 포매터 (모듈 레벨 — 중첩 함수 제거)."""
    if v is None:
        return "N/A"
    try:
        f = float(v) * mult
        return f"{f:.{decimals}f}{suffix}" if np.isfinite(f) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 Fetch (재시도 + 로깅)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ticker_data(symbol: str) -> TickerMetrics:
    """
    Yahoo Finance에서 단일 티커를 Fetch합니다.
    최대 RETRY_MAX 회, Exponential Backoff 재시도.
    """
    symbol    = symbol.strip().upper()
    last_exc: Optional[Exception] = None

    for attempt in range(1, RETRY_MAX + 1):
        try:
            tk   = yf.Ticker(symbol)
            info = tk.info or {}

            m = TickerMetrics(
                symbol       = symbol,
                name         = info.get("shortName", symbol),
                sector       = info.get("sector", "N/A"),
                market_cap   = _safe_float(info.get("marketCap")),
                ev           = _safe_float(info.get("enterpriseValue")),
                price        = _safe_float(info.get("currentPrice")
                                           or info.get("regularMarketPrice")),
                revenue      = _safe_float(info.get("totalRevenue")),
                rev_growth   = _safe_float(info.get("revenueGrowth")),
                gross_margin = _safe_float(info.get("grossMargins")),
                op_income    = _safe_float(info.get("operatingIncome")) or 0.0,
                net_income   = _safe_float(info.get("netIncomeToCommon")) or 0.0,
                fcf          = _safe_float(info.get("freeCashflow")),
                cash         = _safe_float(info.get("totalCash"))  or 0.0,
                debt         = _safe_float(info.get("totalDebt"))  or 0.0,
                capex        = _safe_float(info.get("capitalExpenditures")),
                shares_out   = _safe_float(info.get("sharesOutstanding")),
            )

            # ── 현금흐름표 ─────────────────────────────────────────────
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
                logger.debug("[%s] 현금흐름표 오류: %s", symbol, e)

            # ── 재무상태표 ─────────────────────────────────────────────
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
                logger.debug("[%s] 재무상태표 오류: %s", symbol, e)

            # ── 전년 주식수 fallback ───────────────────────────────────
            if m.shares_out_prior is None and m.shares_out:
                try:
                    sh_hist = tk.get_shares_full(start="2022-01-01")
                    if sh_hist is not None and len(sh_hist) > 0:
                        series  = sh_hist.sort_index()
                        yr_ago  = pd.Timestamp.now() - pd.DateOffset(years=1)
                        prior   = series[series.index <= yr_ago]
                        if not prior.empty:
                            m.shares_out_prior = float(prior.iloc[-1])
                except Exception as e:
                    logger.debug("[%s] shares_full 오류: %s", symbol, e)

            logger.debug("[%s] Fetch 완료 (시도 %d/%d)", symbol, attempt, RETRY_MAX)
            return m

        except Exception as exc:
            last_exc = exc
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning("[%s] Fetch 실패 (시도 %d/%d): %s → %.0fs 재시도",
                           symbol, attempt, RETRY_MAX, exc, wait)
            if attempt < RETRY_MAX:
                time.sleep(wait)

    logger.error("[%s] 최종 Fetch 실패: %s", symbol, last_exc)
    return TickerMetrics(symbol=symbol, error=str(last_exc))


# ══════════════════════════════════════════════════════════════════════════════
# 희석 프로파일 계산
# ══════════════════════════════════════════════════════════════════════════════

def calc_dilution_profile(m: TickerMetrics) -> DilutionProfile:
    """
    ATM vs SBC 희석 분리.

    총 희석 = 주식수 YoY 증가율
    SBC 기여 희석 ≈ SBC_abs / Market_Cap  (rough estimate)
    ATM ≈ max(0, 총 희석 - SBC 기여)

    SBC Justified (팔란티어형) :
        총마진 ≥70% AND FCF마진 ≥8% AND 성장률 ≥20% 중 2가지 이상 충족
    """
    dp  = DilutionProfile()
    rev = m.revenue or 1.0

    dp.sbc_abs   = m._sbc_abs
    dp.sbc_ratio = dp.sbc_abs / rev if rev > 0 else 0.0

    # 주식수 YoY
    if m.shares_out and m.shares_out_prior and m.shares_out_prior > 0:
        dp.shares_yoy_pct = (m.shares_out - m.shares_out_prior) / m.shares_out_prior
    else:
        dp.shares_yoy_pct = 0.0

    # ATM 추정
    sbc_implied = (dp.sbc_abs / m.market_cap) if (m.market_cap and m.market_cap > 0
                                                   and dp.sbc_abs > 0) else 0.0
    dp.atm_dilution_pct = max(0.0, dp.shares_yoy_pct - sbc_implied)

    # SBC Justified 판단
    gm = m.gross_margin or 0.0
    fm = m.fcf_margin   or 0.0
    gr = m.rev_growth   or 0.0
    reasons: list[str] = []
    if gm >= SBC_JUSTIFIED_GM_MIN:  reasons.append(f"총마진 {gm*100:.0f}%")
    if fm >= SBC_JUSTIFIED_FCF_MIN: reasons.append(f"FCF마진 {fm*100:.0f}%")
    if gr >= SBC_JUSTIFIED_GR_MIN:  reasons.append(f"성장 {gr*100:.0f}%")

    if dp.sbc_ratio >= SBC_HIGH_THRESHOLD and len(reasons) >= 2:
        dp.is_sbc_justified   = True
        dp.justification_msg  = "OK: " + " + ".join(reasons)
    elif dp.sbc_ratio >= SBC_HIGH_THRESHOLD:
        missing: list[str] = []
        if gm < SBC_JUSTIFIED_GM_MIN:  missing.append(f"총마진 미달({gm*100:.0f}%)")
        if fm < SBC_JUSTIFIED_FCF_MIN: missing.append(f"FCF 미달({fm*100:.0f}%)")
        if gr < SBC_JUSTIFIED_GR_MIN:  missing.append(f"성장 미달({gr*100:.0f}%)")
        dp.justification_msg = "미충족: " + ", ".join(missing)

    return dp


# ══════════════════════════════════════════════════════════════════════════════
# 계약 가시성
# ══════════════════════════════════════════════════════════════════════════════

def calc_contract_visibility(m: TickerMetrics) -> ContractVisibility:
    """Deferred Revenue 기반 계약 가시성 프록시."""
    cv = ContractVisibility()
    if not m.revenue or m.revenue <= 0:
        return cv

    cv.deferred_rev   = m.deferred_rev_cur
    if m.deferred_rev_cur > 0:
        cv.deferred_ratio = m.deferred_rev_cur / m.revenue
        if m.deferred_rev_prior and m.deferred_rev_prior > 0:
            cv.deferred_yoy_growth = (
                (m.deferred_rev_cur - m.deferred_rev_prior) / m.deferred_rev_prior
            )

    # 점수 (0~12)
    if cv.deferred_ratio >= VISIBILITY_HIGH_RATIO:
        base = SCORE_MAX_VISIBILITY
    elif cv.deferred_ratio >= VISIBILITY_MED_RATIO:
        base = SCORE_MAX_VISIBILITY * 0.6
    else:
        base = SCORE_MAX_VISIBILITY * min(cv.deferred_ratio / VISIBILITY_MED_RATIO, 1.0) * 0.4

    growth_bonus = 0.0
    if cv.deferred_yoy_growth > 0.30:
        growth_bonus = SCORE_MAX_VISIBILITY * 0.20
    elif cv.deferred_yoy_growth > 0.10:
        growth_bonus = SCORE_MAX_VISIBILITY * 0.10

    cv.score = min(base + growth_bonus, SCORE_MAX_VISIBILITY)

    if cv.deferred_ratio >= VISIBILITY_HIGH_RATIO:
        cv.label = f"{cv.deferred_ratio*100:.0f}%(높음)"
    elif cv.deferred_ratio >= VISIBILITY_MED_RATIO:
        cv.label = f"{cv.deferred_ratio*100:.0f}%(보통)"
    elif cv.deferred_ratio > 0:
        cv.label = f"{cv.deferred_ratio*100:.1f}%(낮음)"

    return cv


# ══════════════════════════════════════════════════════════════════════════════
# 단계 분류 (5단계)
# ══════════════════════════════════════════════════════════════════════════════

def classify_stage(m: TickerMetrics) -> str:
    """
    기존 3단계 → 5단계 세분화.
    '$499M vs $500M 절벽' 문제 해소: 성장률 × FCF 조합 기반 연속 판단.

    ⚡ 하이퍼성장  : 성장 50%+ AND FCF 미형성
    🌱 초기단계    : 매출 <$500M OR 영업적자
    🚀 성장가속    : 성장 25%+ AND FCF 10% 미만
    💎 성숙고성장  : 성장 10%+ AND FCF 10%+
    🏛 성숙전환    : 그 외
    """
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
# 점수 산출
# ══════════════════════════════════════════════════════════════════════════════

def calc_score(m: TickerMetrics,
               dp: DilutionProfile,
               cv: ContractVisibility,
               stage: str) -> ScoringResult:
    """
    분자 (최대 100점):
      성장률     35pt  — 50%+가 만점, 80%+는 보너스
      총마진     20pt  — 80%+가 만점
      FCF마진    18pt  — 20%+가 만점
      Rule-of-40 10pt  — 60+가 만점
      계약가시성  12pt  — Deferred Rev 기반 (0~12)
      FCF품질     5pt  — FCF/OCF 품질

    페널티:
      ATM희석   -20pt  — 초기/하이퍼는 50% 경감
      SBC희석   -15pt  — Justified 시 최대 6pt 감면
      Capex부담  -8pt
    """
    sr = ScoringResult(symbol=m.symbol, stage=stage, dilution=dp, visibility=cv)

    g  = m.rev_growth    or 0.0
    gm = m.gross_margin  or 0.0
    fm = m.fcf_margin    or 0.0
    fq = m.fcf_quality

    # ── 분자 ──────────────────────────────────────────────────────────────
    sr.growth_pt        = min(g  / 0.50, 1.0) * SCORE_MAX_GROWTH
    if g >= 0.80:  # 하이퍼 성장 보너스 (10% 추가, 만점 내)
        sr.growth_pt = min(sr.growth_pt * 1.10, SCORE_MAX_GROWTH)

    sr.gross_margin_pt  = min(gm / 0.80, 1.0) * SCORE_MAX_GROSS_MARGIN
    sr.fcf_pt           = min(max(fm, 0.0) / 0.20, 1.0) * SCORE_MAX_FCF
    sr.rule40_pt        = min(max(m.rule_of_40, 0.0) / 60.0, 1.0) * SCORE_MAX_RULE40
    sr.visibility_pt    = cv.score

    if fq is not None:
        if 0.65 <= fq <= 1.15:
            sr.fcf_quality_pt = SCORE_MAX_FCF_QUALITY
        elif 0.40 <= fq < 0.65:
            sr.fcf_quality_pt = SCORE_MAX_FCF_QUALITY * 0.5
        elif fq > 1.30:   # working capital 일시 효과 의심
            sr.fcf_quality_pt = SCORE_MAX_FCF_QUALITY * 0.3

    # ── 페널티 ────────────────────────────────────────────────────────────
    is_early = any(s in stage for s in ["초기", "하이퍼"])
    atm_discount = 0.50 if is_early else 1.0
    sr.atm_penalty = min(dp.atm_dilution_pct / 0.08, 1.0) * SCORE_PEN_ATM * atm_discount

    base_sbc = min(dp.net_sbc_penalty_ratio / 0.15, 1.0) * SCORE_PEN_SBC
    if dp.is_sbc_justified:
        credit = min(
            ((gm - SBC_JUSTIFIED_GM_MIN) / 0.10) * (SCORE_PEN_SBC_JUSTIFIED * 0.5)
            + ((fm - SBC_JUSTIFIED_FCF_MIN) / 0.10) * (SCORE_PEN_SBC_JUSTIFIED * 0.5),
            SCORE_PEN_SBC_JUSTIFIED,
        )
        sr.sbc_penalty = max(0.0, base_sbc - credit)
    else:
        sr.sbc_penalty = base_sbc

    sr.capex_penalty = min(m.capex_ratio / 0.10, 1.0) * SCORE_PEN_CAPEX

    raw = sr.numerator_sum - sr.penalty_sum
    sr.total_score = round(max(min(raw, 100.0), 0.0), 1)
    sr.grade = _to_grade(sr.total_score)
    return sr


def _to_grade(score: float) -> str:
    if score >= 82: return "S+"
    if score >= 75: return "S"
    if score >= 65: return "A"
    if score >= 52: return "B"
    if score >= 38: return "C"
    return "D"


# ══════════════════════════════════════════════════════════════════════════════
# 결과 행 생성
# ══════════════════════════════════════════════════════════════════════════════

def build_row(m: TickerMetrics, sr: ScoringResult) -> dict:
    """DataFrame 한 행을 딕셔너리로 구성합니다."""
    dp = sr.dilution
    cv = sr.visibility

    if "초기" in sr.stage or "하이퍼" in sr.stage:
        key_val = f"순현금: {_fmt(m.net_cash, 1/1e9)}B$"
    elif "가속" in sr.stage:
        key_val = f"EV/GP: {_fmt(m.ev_gross_profit, suffix='x')}"
    else:
        key_val = f"EV/FCF: {_fmt(m.ev_fcf, suffix='x')}"

    return {
        "티커":              m.symbol,
        "기업명":            m.name,
        "섹터":              m.sector,
        "단계":              sr.stage,
        "종합점수":          sr.total_score,
        "등급":              sr.grade,
        "핵심지표":          key_val,

        "시가총액(B$)":      _fmt(m.market_cap,    1/1e9),
        "EV(B$)":            _fmt(m.ev,             1/1e9),
        "현재가($)":         _fmt(m.price),

        "매출성장(%)":       _fmt(m.rev_growth,    100),
        "총마진(%)":         _fmt(m.gross_margin,  100),
        "FCF마진(%)":        _fmt(m.fcf_margin,    100),
        "Rule-of-40":        _fmt(m.rule_of_40,    1, "", 1),
        "FCF품질(FCF/OCF)":  _fmt(m.fcf_quality,  1, "", 2),

        "EV/매출":           _fmt(m.ev_rev,             suffix="x"),
        "EV/GP":             _fmt(m.ev_gross_profit,     suffix="x"),
        "EV/FCF":            _fmt(m.ev_fcf,              suffix="x"),

        # ── 희석 분석 (핵심 신규) ──────────────────────────────────────
        "SBC/매출(%)":       _fmt(dp.sbc_ratio,          100),
        "SBC구분":           dp.sbc_flag,
        "ATM희석(%)":        _fmt(dp.atm_dilution_pct,   100),
        "ATM구분":           dp.atm_flag,
        "주식수YoY(%)":      _fmt(dp.shares_yoy_pct,     100),
        "SBC정당화":         dp.justification_msg if dp.sbc_ratio >= SBC_HIGH_THRESHOLD else "-",

        # ── 계약 가시성 (핵심 신규) ────────────────────────────────────
        "계약가시성":         cv.flag,
        "이연매출/매출(%)":   cv.label,
        "이연매출YoY(%)":    _fmt(cv.deferred_yoy_growth, 100),

        "순현금(B$)":        _fmt(m.net_cash,     1/1e9),
        "Capex/매출(%)":     _fmt(m.capex_ratio,  100),

        # 점수 내역 (CSV 분석용, 콘솔엔 별도 표)
        "_pt_growth":        round(sr.growth_pt, 1),
        "_pt_gmargin":       round(sr.gross_margin_pt, 1),
        "_pt_fcf":           round(sr.fcf_pt, 1),
        "_pt_r40":           round(sr.rule40_pt, 1),
        "_pt_visibility":    round(sr.visibility_pt, 1),
        "_pt_fcfq":          round(sr.fcf_quality_pt, 1),
        "_pen_atm":          round(sr.atm_penalty, 1),
        "_pen_sbc":          round(sr.sbc_penalty, 1),
        "_pen_capex":        round(sr.capex_penalty, 1),
        "_total_num":        round(sr.numerator_sum, 1),
        "_total_pen":        round(sr.penalty_sum, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 단일 심볼 전체 파이프라인
# ══════════════════════════════════════════════════════════════════════════════

def analyze_symbol(symbol: str) -> dict:
    """Fetch → Dilution → Visibility → Score → Row 단일 파이프라인."""
    m = fetch_ticker_data(symbol)
    if m.is_error:
        return {
            "티커": symbol, "기업명": "ERROR", "단계": "❌",
            "종합점수": 0.0, "등급": "F", "_error": m.error,
        }
    dp    = calc_dilution_profile(m)
    cv    = calc_contract_visibility(m)
    stage = classify_stage(m)
    sr    = calc_score(m, dp, cv, stage)
    return build_row(m, sr)


# ══════════════════════════════════════════════════════════════════════════════
# 배치 처리 (병렬)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_batch(symbols: list[str],
                  workers: int = DEFAULT_WORKERS,
                  verbose: bool = True) -> pd.DataFrame:
    """
    ThreadPoolExecutor로 병렬 Fetch.
    결과는 입력 순서로 재정렬되어 반환됩니다.
    """
    total   = len(symbols)
    order   = {sym: i for i, sym in enumerate(symbols)}
    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(analyze_symbol, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                logger.error("[%s] 예기치 않은 오류: %s", sym, exc)
                row = {"티커": sym, "기업명": "ERROR", "단계": "❌",
                       "종합점수": 0.0, "등급": "F", "_error": str(exc)}
            results[sym] = row
            if verbose:
                done = len(results)
                err  = row.get("_error", "")
                if err:
                    print(f"  [{done:>3}/{total}] {sym:<8} ⚠  {str(err)[:55]}")
                else:
                    sbc_flag = row.get("SBC구분", "")
                    atm_flag = row.get("ATM구분", "")
                    print(f"  [{done:>3}/{total}] {sym:<8} ✓  "
                          f"{row.get('단계','?'):<14}  "
                          f"점수={row.get('종합점수', 0):<5}  "
                          f"등급={row.get('등급','?')}  "
                          f"{sbc_flag} / {atm_flag}")
            time.sleep(FETCH_DELAY_SEC / max(workers, 1))

    sorted_rows = sorted(results.values(),
                         key=lambda r: order.get(r.get("티커", ""), 9999))
    df = pd.DataFrame(sorted_rows)
    if "_error" in df.columns:
        df = df.drop(columns=["_error"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 콘솔 리포트
# ══════════════════════════════════════════════════════════════════════════════

_MAIN_COLS = [
    "티커", "단계", "매출성장(%)", "총마진(%)", "FCF마진(%)",
    "Rule-of-40", "EV/GP", "EV/FCF",
    "SBC구분", "ATM구분", "계약가시성",
    "종합점수", "등급",
]
_SCORE_DETAIL_COLS = [
    "티커", "등급", "종합점수", "_total_num", "_total_pen",
    "_pt_growth", "_pt_gmargin", "_pt_fcf", "_pt_r40",
    "_pt_visibility", "_pen_atm", "_pen_sbc", "_pen_capex",
]
_DILUTION_COLS = [
    "티커", "SBC/매출(%)", "SBC구분", "ATM희석(%)", "ATM구분",
    "주식수YoY(%)", "FCF마진(%)", "총마진(%)", "매출성장(%)", "SBC정당화",
]
_VISIBILITY_COLS = [
    "티커", "계약가시성", "이연매출/매출(%)", "이연매출YoY(%)",
    "EV/GP", "EV/FCF", "FCF마진(%)", "등급",
]


def print_report(df: pd.DataFrame) -> None:
    sdf = df.sort_values("종합점수", ascending=False).reset_index(drop=True)
    W   = 130

    print("\n" + "═" * W)
    print(f"  📊  고성장 주식 밸류에이션 v2.0  ——  "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M')}  "
          f"({len(sdf)}개 종목)")
    print("═" * W)

    # ── 메인 테이블 ───────────────────────────────────────────────────────
    pc = [c for c in _MAIN_COLS if c in sdf.columns]
    try:
        print(sdf[pc].to_string(index=False, max_colwidth=20))
    except Exception:
        print(sdf[pc].to_string(index=False))
    print("═" * W)

    # ── 단계별 분포 ───────────────────────────────────────────────────────
    print("\n【 단계별 분포 】")
    for stage, grp in sdf.groupby("단계", sort=False):
        print(f"  {stage:<16}  {len(grp):>2}개  →  {', '.join(grp['티커'].tolist())}")

    # ── S+/S/A 등급 ───────────────────────────────────────────────────────
    hi_cols = [c for c in ["티커","등급","종합점수","단계","핵심지표","SBC정당화"]
               if c in sdf.columns]
    top = sdf[sdf["등급"].isin(["S+","S","A"])][hi_cols]
    if not top.empty:
        print("\n【 S+/S/A 등급 (투자 매력도 높음) 】")
        print(top.to_string(index=False))

    # ── 희석 분석 ─────────────────────────────────────────────────────────
    dc = [c for c in _DILUTION_COLS if c in sdf.columns]
    print("\n【 희석 구조 분석 (ATM vs SBC) 】")
    try:
        justified = sdf[sdf["SBC정당화"].str.startswith("OK:", na=False)]
        warn_sbc  = sdf[
            sdf["SBC구분"].str.contains("고비율", na=False)
            & ~sdf["SBC정당화"].str.startswith("OK:", na=False)
        ]
        warn_atm  = sdf[sdf["ATM구분"].str.contains("주의|고위험", na=False)]

        if not justified.empty:
            print("\n  ▶ SBC 고비율이지만 펀더멘털로 정당화된 케이스 (팔란티어형):")
            print(justified[dc].to_string(index=False))
        if not warn_sbc.empty:
            print("\n  ▶ ⚠ SBC 고비율 / 정당화 미충족 (모니터링 필요):")
            print(warn_sbc[dc].to_string(index=False))
        if not warn_atm.empty:
            print("\n  ▶ 🔴 ATM 희석 주의 (주식 신규 발행으로 인한 가치 희석):")
            print(warn_atm[dc].to_string(index=False))
    except Exception as e:
        logger.debug("희석 출력 오류: %s", e)

    # ── 계약 가시성 ───────────────────────────────────────────────────────
    vc = [c for c in _VISIBILITY_COLS if c in sdf.columns]
    if vc:
        print("\n【 계약 가시성 분석 (Deferred Revenue 기반) 】")
        print(sdf[vc].to_string(index=False))

    # ── 점수 내역 ─────────────────────────────────────────────────────────
    sc = [c for c in _SCORE_DETAIL_COLS if c in sdf.columns]
    if sc:
        print("\n【 점수 내역 (분자 / 페널티 분해) 】")
        print("  분자: growth + gmargin + fcf + r40 + visibility")
        print("  페널티: atm + sbc + capex")
        print(sdf[sc].to_string(index=False))

    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="고성장 주식 밸류에이션 분석기 v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tickers",  nargs="+", metavar="SYM",
                   help="직접 티커 입력  예: PLTR SNOW NVDA RKLB")
    p.add_argument("--file",     default=DEFAULT_TICKER_FILE, metavar="PATH",
                   help=f"티커 목록 파일 (기본: {DEFAULT_TICKER_FILE})")
    p.add_argument("--output",   default=DEFAULT_OUTPUT_FILE, metavar="PATH",
                   help=f"CSV 저장 경로 (기본: {DEFAULT_OUTPUT_FILE})")
    p.add_argument("--workers",  type=int, default=DEFAULT_WORKERS, metavar="N",
                   help=f"동시 Fetch 워커 수 (기본: {DEFAULT_WORKERS})")
    p.add_argument("--no-csv",   action="store_true", help="CSV 저장 생략")
    p.add_argument("--no-score-detail", action="store_true",
                   help="CSV에서 _pt_/_pen_ 컬럼 제외")
    p.add_argument("--quiet",    action="store_true", help="진행 로그 숨김")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"],
                   help="로그 레벨 (기본: INFO)")
    return p.parse_args()


def load_symbols(fpath: Path) -> list[str]:
    if not fpath.exists():
        logger.error("파일 없음: %s", fpath)
        sys.exit(1)
    syms: list[str] = []
    for line in fpath.read_text(encoding="utf-8").splitlines():
        cleaned = line.split("#")[0].strip().upper()
        if cleaned:
            syms.append(cleaned)
    return syms


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    symbols = (
        [s.strip().upper() for s in args.tickers if s.strip()]
        if args.tickers
        else load_symbols(Path(args.file))
    )
    if not symbols:
        logger.error("분석할 티커 없음")
        sys.exit(1)

    # 중복 제거 (입력 순서 보장)
    seen: set[str] = set()
    unique = [s for s in symbols if not (s in seen or seen.add(s))]  # type: ignore

    print(f"\n🔍  분석 시작 — {len(unique)}개 티커  (워커: {args.workers})")
    print(f"    {', '.join(unique[:12])}{'...' if len(unique)>12 else ''}\n")

    t0 = time.monotonic()
    df = analyze_batch(unique, workers=args.workers, verbose=not args.quiet)
    elapsed = time.monotonic() - t0

    print(f"\n⏱  Fetch 완료: {elapsed:.1f}초")
    print_report(df)

    if not args.no_csv:
        out = df.sort_values("종합점수", ascending=False)
        if args.no_score_detail:
            drop = [c for c in out.columns if c.startswith(("_pt_","_pen_","_total_"))]
            out  = out.drop(columns=drop, errors="ignore")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"💾  CSV 저장: {out_path}\n")

    print(f"✅  완료  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
