import re
import io
import json
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt


# ============ FAISS ============
try:
    import faiss  # faiss-cpu
    FAISS_OK = True
except ImportError:
    FAISS_OK = False


# =========================
# CI Theme / Title
# =========================
st.set_page_config(page_title="Overseas Unit Rate App", layout="wide")

CI_BLUE   = "#005EB8"
CI_TEAL   = "#00BFB3"
BG_LIGHT  = "#F6FAFC"

st.markdown("""
<style>
/* =====================================================
   TOKENS
===================================================== */
:root{
  --bg: #F6F8FC;
  --card: #FFFFFF;
  --text: #0F172A;
  --muted: #64748B;
  --border: rgba(15, 23, 42, 0.10);
  --shadow-sm: 0 6px 14px rgba(15, 23, 42, 0.05);
  --primary: #2563EB;

  --sb-bg: #FFFFFF;
  --sb-border: #E6EAF2;

  --chip-bg: #EEF2FF;
  --chip-border: #C7D2FE;
  --chip-text: #1E3A8A;
}

/* =====================================================
   LAYOUT / TYPO
===================================================== */
[data-testid="stAppViewContainer"]{ background: var(--bg) !important; }
html, body{ font-size: 14px !important; color: var(--text) !important; }
.main{ color: var(--text) !important; }
.main > div{
  padding: 16px 24px 24px 24px !important;
  max-width: 1280px;
  margin: 0 auto;
}

.main h1{ font-size: 26px !important; font-weight: 900 !important; letter-spacing: -0.5px !important; }
.main h2{ font-size: 20px !important; font-weight: 850 !important; letter-spacing: -0.3px !important; }
.main h3{ font-size: 16px !important; font-weight: 850 !important; }
.main .stCaption, .main small{ color: var(--muted) !important; font-size: 12.5px !important; }

/* =====================================================
   CARDS
===================================================== */
.gs-card{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 18px !important;
  margin: 14px 0 18px 0 !important;
  box-shadow: var(--shadow-sm) !important;
}
.gs-header{
  font-size: 18px !important;
  font-weight: 900 !important;
  letter-spacing: -0.3px !important;
  margin: 6px 0 14px 0 !important;
}

.dash-row{ display:flex; align-items:baseline; justify-content:space-between; gap:10px; margin:0 0 10px 0; }
.dash-title{ font-size: 14px !important; font-weight: 850 !important; letter-spacing: -0.2px !important; }
.dash-muted{ font-size: 12px !important; color: var(--muted) !important; white-space: nowrap !important; }

/* st.container(border=True) 카드화 */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 18px !important;
  margin: 14px 0 18px 0 !important;
  box-shadow: var(--shadow-sm) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{ padding: 0 !important; }

/* =====================================================
   SIDEBAR
===================================================== */
section[data-testid="stSidebar"]{
  background: var(--sb-bg) !important;
  border-right: 1px solid var(--sb-border) !important;
}
section[data-testid="stSidebar"] > div{ padding-top: 14px !important; }
section[data-testid="stSidebar"] hr{
  border: none !important;
  border-top: 1px solid var(--sb-border) !important;
  margin: 10px 0 !important;
}

.sb-row{ display:flex; align-items:baseline; justify-content:space-between; gap:10px; margin:2px 0 6px 0; }
.sb-title{ font-size: 14px !important; font-weight: 800 !important; letter-spacing: -0.2px !important; color: var(--text) !important; }
.sb-muted{ font-size: 12px !important; color: var(--muted) !important; white-space: nowrap !important; }
.sb-major{ font-size: 16px !important; font-weight: 900 !important; margin: 6px 0 10px 0 !important; letter-spacing: -0.2px !important; color: var(--text) !important; }

/* =====================================================
   WIDGETS (최소만)
===================================================== */

/* Select / Multiselect 컨트롤 */
div[data-baseweb="select"] > div{
  background: #FFFFFF !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  min-height: 42px !important;
  box-shadow: none !important;
}
div[data-baseweb="select"] input{
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
  caret-color: var(--text) !important;
  font-size: 13px !important;
}

/* Chip(Tag) */
div[data-baseweb="tag"], span[data-baseweb="tag"]{
  background: var(--chip-bg) !important;
  border: 1px solid var(--chip-border) !important;
  color: var(--chip-text) !important;
  border-radius: 999px !important;
}

/* Text input */
div[data-testid="stTextInput"] div[data-baseweb="input"] > div{
  background: #FFFFFF !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  min-height: 42px !important;
  box-shadow: none !important;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"]{
  background: #FFFFFF !important;
  border: 1px dashed rgba(15,23,42,0.18) !important;
  border-radius: 16px !important;
  padding: 16px !important;
}
[data-testid="stFileUploaderDropzone"] button{
  background: var(--primary) !important;
  color: #FFFFFF !important;
  border: 0 !important;
  border-radius: 12px !important;
  font-weight: 800 !important;
}

/* DataFrame/DataEditor: 컨테이너만 라이트 */
div[data-testid="stDataFrame"],
div[data-testid="stDataEditor"]{
  background: #FFFFFF !important;
  border-radius: 16px !important;
}
/* =====================================================
   TABS — 선택됨/미선택 크기 분리 (단일 정의로 정리)
===================================================== */

/* 기본(미선택) 탭 */
div[data-testid="stTabs"] button[role="tab"],
.stTabs button[role="tab"],
.stTabs [data-baseweb="tab"]{
  font-size: 18px !important;
  font-weight: 700 !important;
  padding: 10px 14px !important;
  line-height: 1.2 !important;
}

/* 미선택 탭 내부 텍스트 */
div[data-testid="stTabs"] button[role="tab"] *,
.stTabs button[role="tab"] *,
.stTabs [data-baseweb="tab"] *{
  font-size: 18px !important;
}

/* 선택된 탭 */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
.stTabs button[role="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"]{
  font-size: 24px !important;
  font-weight: 900 !important;
}

/* 선택된 탭 내부 텍스트 */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] *,
.stTabs button[role="tab"][aria-selected="true"] *,
.stTabs [data-baseweb="tab"][aria-selected="true"] *{
  font-size: 24px !important;
}

/* 이모지/텍스트 정렬 */
div[data-testid="stTabs"] button[role="tab"] > div,
.stTabs button[role="tab"] > div,
.stTabs [data-baseweb="tab"] > div{
  gap: 6px !important;
}

</style>
""", unsafe_allow_html=True)


def sidebar_hr(thick: bool = False, mt: int = 6, mb: int = 6):
    # ✅ 연한 회색 구분선 통일
    color = "#D9DDE3"  # 연한 회색
    h = "3px" if thick else "1px"
    st.sidebar.markdown(
        f"<hr style='margin:{mt}px 0 {mb}px 0; border:none; border-top:{h} solid {color};' />",
        unsafe_allow_html=True
    )

# =========================
# UI Helper (Style-aligned)
# =========================
def gs_header(text: str):
    st.markdown(f"<div class='gs-header'>{text}</div>", unsafe_allow_html=True)

def card_begin():
    st.markdown("<div class='gs-card'>", unsafe_allow_html=True)

def card_end():
    st.markdown("</div>", unsafe_allow_html=True)

def card_title(title: str, right: str = ""):
    st.markdown(
        f"""
        <div class="dash-row">
          <div class="dash-title">{title}</div>
          <div class="dash-muted">{right}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# Model (cached)
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# =========================
# Utils
# =========================
def norm_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def to_year_month_string(x) -> Optional[str]:
    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            s = str(x)
            s2 = re.sub(r"[^0-9]", "", s)[:6]
            dt = pd.to_datetime(s2, format="%Y%m", errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m")
    except Exception:
        return None


def robust_parse_contract_month(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    mask = dt.isna()
    if mask.any():
        cleaned = series[mask].astype(str).str.replace(r"[^0-9]", "", regex=True).str.slice(0, 6)
        dt2 = pd.to_datetime(cleaned, format="%Y%m", errors="coerce")
        dt.loc[mask] = dt2
    return dt.dt.to_period("M").dt.to_timestamp()


def file_fingerprint(df: pd.DataFrame, cols: list) -> str:
    hasher = hashlib.md5()
    sample = df[cols].astype(str).agg("|".join, axis=1)
    head = "|".join(sample.head(1000).tolist())
    tail = "|".join(sample.tail(1000).tolist())
    hasher.update(str(df.shape).encode())
    hasher.update(head.encode())
    hasher.update(tail.encode())
    return hasher.hexdigest()


def norm_site_code(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = s.strip('"\''"`")
    if s.endswith(".0"):
        s = s[:-2]
    s = s.split(".")[0].strip()
    s_digits = "".join(ch for ch in s if ch.isdigit())
    if s_digits:
        s = s_digits
    if s.isdigit() and len(s) < 6:
        s = s.zfill(6)
    return s

def norm_unit_kr(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace(" ", "")
    # 필요 시 여기서 단위 alias를 더 추가하세요
    alias = {
        "m2": "m2",
        "m3": "m3",
        "ea": "ea",
        "㎥": "m3",
        "㎡": "m2",
        "개": "ea",
    }
    return alias.get(s, s)

def norm_kr_boq_text(name, spec) -> str:
    # BOQ: 명칭 + 규격을 합쳐서 매칭 텍스트로 사용
    a = "" if pd.isna(name) else str(name)
    b = "" if pd.isna(spec) else str(spec)
    return norm_text(f"{a} {b}")

def norm_kr_db_text(exec_name, spec) -> str:
    # 국내 DB: 실행명칭 + 규격을 합쳐서 매칭 텍스트로 사용
    a = "" if pd.isna(exec_name) else str(exec_name)
    b = "" if pd.isna(spec) else str(spec)
    return norm_text(f"{a} {b}")

# =========================
# 보정 로직
# =========================
def get_cpi_ratio(price_index: pd.DataFrame, currency: str, contract_ym: str):
    try:
        df = price_index[price_index["국가"].astype(str).str.upper() == str(currency).upper()].copy()
        if df.empty:
            return 1.0, None, None, None
        df["년월_std"] = df["년월"].apply(to_year_month_string)
        latest_ym = df["년월_std"].dropna().max()
        base = df.loc[df["년월_std"] == contract_ym, "Index"].values
        now = df.loc[df["년월_std"] == latest_ym, "Index"].values
        if len(base) and len(now) and base[0] not in (0, None):
            return float(now[0]) / float(base[0]), float(base[0]), float(now[0]), latest_ym
    except Exception:
        pass
    return 1.0, None, None, None


def get_exchange_rate(exchange: pd.DataFrame, from_currency: str, to_currency: str) -> float:
    try:
        usd_from = exchange.loc[
            exchange["통화"].astype(str).str.upper() == str(from_currency).upper(), "USD당환율"
        ].values
        usd_to = exchange.loc[
            exchange["통화"].astype(str).str.upper() == str(to_currency).upper(), "USD당환율"
        ].values
        if len(usd_from) and len(usd_to) and float(usd_from[0]) != 0:
            return float(usd_to[0]) / float(usd_from[0])
    except Exception:
        pass
    return 1.0


def get_factor_ratio(factor: pd.DataFrame, from_currency: str, to_currency: str) -> float:
    try:
        f_from = factor.loc[
            factor["국가"].astype(str).str.upper() == str(from_currency).upper(), "지수"
        ].values
        f_to = factor.loc[
            factor["국가"].astype(str).str.upper() == str(to_currency).upper(), "지수"
        ].values
        if len(f_from) and len(f_to) and float(f_from[0]) != 0:
            return float(f_to[0]) / float(f_from[0])
    except Exception:
        pass
    return 1.0


# =========================
# Embedding Cache (Cloud 호환: /tmp)
# =========================
CACHE_DIR = Path("/tmp/overseas_unitrate_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def embeddings_cache_paths(tag: str):
    return CACHE_DIR / f"{tag}.npy", CACHE_DIR / f"{tag}.json"


def save_embeddings(tag: str, embs: np.ndarray, meta: dict):
    npy, meta_json = embeddings_cache_paths(tag)
    np.save(npy, embs.astype("float32"))
    meta_json.write_text(json.dumps(meta, ensure_ascii=False))


def load_embeddings_if_match(tag: str, expected_meta: dict) -> Optional[np.ndarray]:
    npy, meta_json = embeddings_cache_paths(tag)
    if not npy.exists() or not meta_json.exists():
        return None
    try:
        meta = json.loads(meta_json.read_text())
        if meta == expected_meta:
            return np.load(npy)
    except Exception:
        return None
    return None


@st.cache_resource(show_spinner=False)
def compute_or_load_embeddings(cost_db_norm: pd.Series, tag: str) -> np.ndarray:
    expected = {"model": "all-MiniLM-L6-v2", "tag": tag, "count": int(cost_db_norm.shape[0])}
    cached = load_embeddings_if_match(tag, expected)
    if cached is not None:
        return cached
    texts = cost_db_norm.tolist()
    embs = model.encode(texts, batch_size=256, convert_to_tensor=False, show_progress_bar=True)
    embs = np.asarray(embs, dtype="float32")
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    save_embeddings(tag, embs, expected)
    return embs


# =========================
# FAISS helpers
# =========================
def build_faiss_index(embs: np.ndarray):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    return index


def search_faiss(index, query_vecs: np.ndarray, top_k: int = 200):
    D, I = index.search(query_vecs, top_k)
    return D, I


# =========================
# Matching
# =========================
def hybrid_scores(boq_text_norm: str, db_texts_norm: pd.Series, sem_scores: np.ndarray, w_str: float, w_sem: float) -> np.ndarray:
    sem = np.clip(sem_scores, 0.0, 1.0)
    str_scores = np.array([fuzz.token_sort_ratio(boq_text_norm, s) / 100.0 for s in db_texts_norm.tolist()], dtype="float32")
    return (w_str * str_scores + w_sem * sem) * 100.0


def build_candidate_pool(
    cost_db: pd.DataFrame,
    boq: pd.DataFrame,
    price_index: pd.DataFrame,
    sim_w_str: float,
    sim_w_sem: float,
    top_k_sem: int,
    pool_per_boq: int = 400,
    progress=None,
    prog_text=None,
) -> pd.DataFrame:
    """
    ✅ 1단계(무거움): BOQ별 후보 풀 생성
    - FAISS 검색 + 문자열 점수 + __hyb 계산까지 여기서만 수행
    - 산출통화(FX/Factor)는 여기서 계산하지 않음(빠른 재계산에서 처리)
    - CPI는 통화/계약월에만 의존하므로 여기서 미리 계산해 둠
    """
    work = cost_db.copy()
    work["__내역_norm"] = work["내역"].apply(norm_text)
    work["__Unit_norm"] = work["Unit"].astype(str).str.lower().str.strip()
    work["_계약월"] = robust_parse_contract_month(work["계약년월"])
    work = work[(pd.to_numeric(work["Unit Price"], errors="coerce") > 0) & (work["_계약월"].notna())].copy()

    price_index2 = price_index.copy()
    price_index2["년월"] = price_index2["년월"].apply(to_year_month_string)

    fp = file_fingerprint(work, ["__내역_norm", "__Unit_norm", "통화", "Unit Price", "_계약월"])
    embs = compute_or_load_embeddings(work["__내역_norm"], tag=f"costdb_{fp}")
    index = build_faiss_index(embs) if FAISS_OK else None

    pool_rows = []
    total = len(boq) if len(boq) else 1

    for i, (_, boq_row) in enumerate(boq.iterrows(), start=1):
        if prog_text is not None:
            prog_text.text(f"후보 풀 생성: {i}/{total} 처리 중…")
        if progress is not None:
            progress.progress(i / total)

        boq_item = str(boq_row.get("내역", ""))
        boq_unit = str(boq_row.get("Unit", "")).lower().strip()
        boq_text_norm = norm_text(boq_item)

        q = model.encode([boq_text_norm], batch_size=1, convert_to_tensor=False)
        q = np.asarray(q, dtype="float32")
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

        if FAISS_OK:
            D, I = search_faiss(index, q, top_k=top_k_sem)
            cand_idx = I[0]
            sem_scores = D[0]
        else:
            all_sem = np.dot(embs, q[0])
            cand_idx = np.argsort(-all_sem)[:top_k_sem]
            sem_scores = all_sem[cand_idx]

        cand_df = work.iloc[cand_idx].copy()
        cand_df["__sem"] = sem_scores

        # Unit 일치 후보만
        unit_df = cand_df[cand_df["__Unit_norm"] == boq_unit].reset_index(drop=True)
        if unit_df.empty:
            continue

        # __hyb 계산(문자열+의미 유사도)
        hyb = hybrid_scores(
            boq_text_norm,
            unit_df["__내역_norm"],
            unit_df["__sem"].to_numpy(),
            sim_w_str,
            sim_w_sem
        )
        unit_df["__hyb"] = hyb

        # 너무 큰 풀 방지: hyb 상위 N개만 보관
        unit_df = unit_df.sort_values("__hyb", ascending=False).head(pool_per_boq).copy()

        # CPI는 통화+계약월 기준으로 미리 계산 (산출통화 바뀌어도 재사용 가능)
        unit_df["__contract_ym"] = unit_df["_계약월"].apply(to_year_month_string)

        cpi_list = []
        for _, r in unit_df.iterrows():
            c_currency = str(r.get("통화", "")).upper().strip()
            contract_ym = r.get("__contract_ym", None)
            cpi_ratio, base_cpi, latest_cpi, latest_ym = get_cpi_ratio(price_index2, c_currency, contract_ym)
            cpi_list.append((cpi_ratio, latest_ym))
        unit_df["__cpi_ratio"] = [x[0] for x in cpi_list]
        unit_df["__latest_ym"] = [x[1] for x in cpi_list]

        # BOQ 메타 붙이기
        boq_id = int(i)
        unit_df["BOQ_ID"] = boq_id
        unit_df["BOQ_내역"] = boq_item
        unit_df["BOQ_Unit"] = boq_unit

        pool_rows.append(unit_df)

    if not pool_rows:
        return pd.DataFrame()

    pool = pd.concat(pool_rows, ignore_index=True)

    keep_cols = [
        "BOQ_ID", "BOQ_내역", "BOQ_Unit",
        "공종코드", "공종명",
        "내역", "Unit",
        "Unit Price", "통화", "계약년월",
        "현장코드", "현장명",
        "협력사코드", "협력사명",
        "__hyb",
        "__cpi_ratio",
        "__latest_ym",
    ]
    for c in keep_cols:
        if c not in pool.columns:
            pool[c] = None
    return pool[keep_cols].copy()


def fast_recompute_from_pool(
    pool: pd.DataFrame,
    exchange: pd.DataFrame,
    factor: pd.DataFrame,
    sim_threshold: float,
    cut_ratio: float,
    target_currency: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ✅ 2단계(가벼움): 후보 풀에서 빠른 재계산
    - Threshold 필터
    - 산출통화 변경: __fx_ratio, __fac_ratio만 다시 계산
    - 컷비율로 Include/DefaultInclude 설정
    - __adj_price = Unit Price * __cpi_ratio * __fx_ratio * __fac_ratio
    """
    if pool is None or pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = pool.copy()

    # 1) Threshold 적용
    df["__hyb"] = pd.to_numeric(df["__hyb"], errors="coerce").fillna(0.0)
    df = df[df["__hyb"] >= float(sim_threshold)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 2) FX/Factor 맵(통화별)
    df["통화_std"] = df["통화"].astype(str).str.upper().str.strip()
    currencies = df["통화_std"].dropna().unique().tolist()

    fx_map = {c: get_exchange_rate(exchange, c, target_currency) for c in currencies}
    fac_map = {c: get_factor_ratio(factor, c, target_currency) for c in currencies}

    df["__fx_ratio"] = df["통화_std"].map(fx_map).fillna(1.0)
    df["__fac_ratio"] = df["통화_std"].map(fac_map).fillna(1.0)
    df["산출통화"] = target_currency

    # -----------------------------------
    # 🔵 여기부터 추가 (Target CPI)
    # -----------------------------------
    df["__contract_ym"] = df["계약년월"].apply(to_year_month_string)
    
    target_cpi_list = []
    
    for _, r in df.iterrows():
        contract_ym = r.get("__contract_ym", None)
        cpi_ratio_target, _, _, _ = get_cpi_ratio(
            price_index,
            target_currency,
            contract_ym
        )
        target_cpi_list.append(cpi_ratio_target)
    
    df["__cpi_target_ratio"] = pd.to_numeric(target_cpi_list, errors="coerce").fillna(1.0)

    # -----------------------------------
    # 🔵 PPP Ratio 계산 (Target CPI 아래)
    # -----------------------------------
    ppp_ratio_list = []
    
    for _, r in df.iterrows():
        source_cur = r["통화_std"]
    
        try:
            contract_year = pd.to_datetime(r["계약년월"]).year
        except:
            contract_year = None
    
        if contract_year and str(contract_year) in ppp.columns:
            ppp_source = ppp.loc[ppp["Currency"] == source_cur, str(contract_year)].values
            ppp_target = ppp.loc[ppp["Currency"] == target_currency, str(contract_year)].values
    
            if len(ppp_source) and len(ppp_target) and float(ppp_source[0]) != 0:
                ratio = float(ppp_target[0]) / float(ppp_source[0])
            else:
                ratio = 1.0
        else:
            ratio = 1.0
    
        ppp_ratio_list.append(ratio)
    
    df["__ppp_ratio"] = ppp_ratio_list

    # 3) __adj_price 계산
    unit_price = pd.to_numeric(df["Unit Price"], errors="coerce").fillna(0.0)
    cpi_source = pd.to_numeric(df["__cpi_ratio"], errors="coerce").fillna(1.0)
    
    # Location 계산
    df["__adj_loc"] = (
        unit_price
        * cpi_source
        * df["__fx_ratio"]
        * df["__fac_ratio"]
    )
    
    # PPP 계산
    df["__adj_ppp"] = (
        unit_price
        * df["__ppp_ratio"]
        * df["__cpi_target_ratio"]
    )
    
    # 최종 선택
    method = st.session_state.get("adjust_method", "Location Factor")
    loc_w = st.session_state.get("loc_weight", 1.0)
    ppp_w = st.session_state.get("ppp_weight", 0.0)
    
    if method == "Location Factor":
        df["__adj_price"] = df["__adj_loc"]
    
    elif method == "PPP Factor":
        df["__adj_price"] = df["__adj_ppp"]
    
    elif method == "혼합 방식":
        df["__adj_price"] = (
            loc_w * df["__adj_loc"]
            + ppp_w * df["__adj_ppp"]
        )

    # 4) BOQ별 컷 + Include/DefaultInclude 설정
    df["Include"] = False
    df["DefaultInclude"] = False

    for boq_id, gidx in df.groupby("BOQ_ID").groups.items():
        sub = df.loc[gidx].sort_values("__adj_price").copy()
        n = len(sub)
        cut = max(0, int(n * cut_ratio)) if n > 5 else 0

        if cut > 0:
            keep_mask = np.zeros(n, dtype=bool)
            keep_mask[cut:n - cut] = True
        else:
            keep_mask = np.ones(n, dtype=bool)

        kept_index = sub.index[keep_mask]
        df.loc[kept_index, "DefaultInclude"] = True
        df.loc[kept_index, "Include"] = True

    # 5) BOQ 결과(result_df)
    results = []
    for boq_id, sub in df.groupby("BOQ_ID"):
        inc = sub[sub["Include"] == True]

        if inc.empty:
            final_price = None
            reason_text = "매칭 후보 없음(또는 전부 제외)"
            top_work = ""
        else:
            final_price = float(pd.to_numeric(inc["__adj_price"], errors="coerce").mean())
            currencies2 = sorted(inc["통화_std"].unique().tolist())
            reason_text = f"{len(currencies2)}개국({', '.join(currencies2)}) {len(inc)}개 내역 근거"

            vc = inc["공종코드"].astype(str).value_counts()
            top_code = vc.index[0] if len(vc) else ""
            top_cnt = int(vc.iloc[0]) if len(vc) else 0
            top_work = f"{top_code} ({top_cnt}/{len(inc)})" if top_code else ""

        one = sub.iloc[0]
        results.append({
            "BOQ_ID": int(boq_id),
            "내역": one.get("BOQ_내역", ""),
            "Unit": one.get("BOQ_Unit", ""),
            "Final Price": f"{final_price:,.2f}" if final_price is not None else None,
            "산출통화": target_currency,
            "산출근거": reason_text,
            "근거공종(최빈)": top_work,
        })

    result_df = pd.DataFrame(results).sort_values("BOQ_ID").reset_index(drop=True)

    # 6) 산출 로그(log_df)
    log_cols = [
        "BOQ_ID", "BOQ_내역", "BOQ_Unit",
        "Include", "DefaultInclude",
        "공종코드", "공종명",
        "내역", "Unit",
        "Unit Price", "통화", "계약년월",
        "__adj_price", "산출통화",
        "__cpi_ratio", "__latest_ym",
        "__fx_ratio", "__fac_ratio",
        "__hyb",
        "현장코드", "현장명",
        "협력사코드", "협력사명",
    ]
    for c in log_cols:
        if c not in df.columns:
            df[c] = None
    log_df = df[log_cols].copy()

    return result_df, log_df

def build_candidate_pool_domestic(
    cost_db_kr: pd.DataFrame,
    boq_kr: pd.DataFrame,
    sim_w_str: float,
    sim_w_sem: float,
    top_k_sem: int,
    pool_per_boq: int = 400,
    progress=None,
    prog_text=None,
) -> pd.DataFrame:
    """
    국내 1단계(무거움): BOQ별 후보 풀 생성
    - 문자열+임베딩 하이브리드로 후보 생성
    - 단위 일치 필수
    - 보정단가(있으면) 우선, 없으면 계약단가를 __price_raw로 사용
    - fast_recompute_from_pool_domestic에서 threshold/cut/include 처리
    """

    if cost_db_kr is None or cost_db_kr.empty or boq_kr is None or boq_kr.empty:
        return pd.DataFrame()

    # 0) BOQ 표준 컬럼 보강
    b = boq_kr.copy()
    need_boq_cols = ["명칭", "규격", "단위", "수량", "단가"]
    for c in need_boq_cols:
        if c not in b.columns:
            b[c] = ""

    # 1) DB 표준 컬럼 보강
    db = cost_db_kr.copy()

    # 국내 DB에서 사용할 컬럼들(없으면 빈값 생성)
    # 실행명칭/규격/단위/수량/계약단가/보정단가/계약월은 이후 산출/로그에 필요
    must = [
        "현장코드","현장명","현장특성",
        "실행명칭","규격","단위","수량",
        "계약단가","보정단가","계약월",
        "업체코드","업체명",
        "공종Code분류","세부분류",
    ]
    for c in must:
        if c not in db.columns:
            db[c] = ""

    # 2) 매칭용 텍스트/단위 정규화
    db["__db_text_norm"] = db.apply(lambda r: norm_kr_db_text(r.get("실행명칭",""), r.get("규격","")), axis=1)
    db["__unit_norm"] = db["단위"].apply(norm_unit_kr)

    # 가격 원천: 보정단가 우선, 없으면 계약단가
    price_adj = pd.to_numeric(db["보정단가"], errors="coerce")
    price_ctr = pd.to_numeric(db["계약단가"], errors="coerce")
    db["__price_raw"] = price_adj.where(price_adj.notna() & (price_adj > 0), price_ctr)
    db = db[pd.to_numeric(db["__price_raw"], errors="coerce").fillna(0) > 0].copy()

    if db.empty:
        return pd.DataFrame()

    # 3) 임베딩(국내 DB)
    fp = file_fingerprint(db, ["__db_text_norm", "__unit_norm", "__price_raw"])
    embs = compute_or_load_embeddings(db["__db_text_norm"], tag=f"costdb_kr_{fp}")

    index = build_faiss_index(embs) if FAISS_OK else None

    pool_rows = []
    total = len(b) if len(b) else 1

    for i, (_, r) in enumerate(b.iterrows(), start=1):
        if prog_text is not None:
            prog_text.text(f"[국내] 후보 풀 생성: {i}/{total} 처리 중…")
        if progress is not None:
            progress.progress(i / total)

        boq_name = str(r.get("명칭", ""))
        boq_spec = str(r.get("규격", ""))
        boq_unit = norm_unit_kr(r.get("단위", ""))

        boq_text_norm = norm_kr_boq_text(boq_name, boq_spec)

        # Query embedding
        q = model.encode([boq_text_norm], batch_size=1, convert_to_tensor=False)
        q = np.asarray(q, dtype="float32")
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

        # semantic topK
        if FAISS_OK:
            D, I = search_faiss(index, q, top_k=top_k_sem)
            cand_idx = I[0]
            sem_scores = D[0]
        else:
            all_sem = np.dot(embs, q[0])
            cand_idx = np.argsort(-all_sem)[:top_k_sem]
            sem_scores = all_sem[cand_idx]

        cand = db.iloc[cand_idx].copy()
        cand["__sem"] = sem_scores

        # 단위 일치 필수
        cand = cand[cand["__unit_norm"] == boq_unit].reset_index(drop=True)
        if cand.empty:
            continue

        # hybrid score
        hyb = hybrid_scores(
            boq_text_norm,
            cand["__db_text_norm"],
            cand["__sem"].to_numpy(),
            sim_w_str,
            sim_w_sem,
        )
        cand["__hyb"] = hyb

        cand = cand.sort_values("__hyb", ascending=False).head(pool_per_boq).copy()

        # BOQ 메타
        cand["BOQ_ID"] = int(i)
        cand["BOQ_명칭"] = boq_name
        cand["BOQ_규격"] = boq_spec
        cand["BOQ_단위"] = boq_unit
        cand["BOQ_수량"] = r.get("수량", "")
        cand["BOQ_단가"] = r.get("단가", "")

        pool_rows.append(cand)

    if not pool_rows:
        return pd.DataFrame()

    pool = pd.concat(pool_rows, ignore_index=True)

    # 로그/후속 처리에서 필요한 컬럼들 포함해서 반환
    keep_cols = [
        "BOQ_ID","BOQ_명칭","BOQ_규격","BOQ_단위","BOQ_수량","BOQ_단가",
        "현장코드","현장명","현장특성",
        "실행명칭","규격","단위","수량",
        "계약단가","보정단가","계약월",
        "업체코드","업체명",
        "공종Code분류","세부분류",
        "__price_raw","__hyb",
    ]
    for c in keep_cols:
        if c not in pool.columns:
            pool[c] = None

    return pool[keep_cols].copy()

def fast_recompute_from_pool_domestic(
    pool: pd.DataFrame,
    sim_threshold: float,
    cut_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    국내 2단계(가벼움)
    - threshold 적용
    - 컷 적용 후 Include/DefaultInclude 설정
    - 산출단가 = __price_raw(보정단가/계약단가)
    """
    if pool is None or pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = pool.copy()
    df["__hyb"] = pd.to_numeric(df["__hyb"], errors="coerce").fillna(0.0)
    df = df[df["__hyb"] >= float(sim_threshold)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["__adj_price"] = pd.to_numeric(df["__price_raw"], errors="coerce").fillna(0.0)

    df["Include"] = False
    df["DefaultInclude"] = False

    for boq_id, gidx in df.groupby("BOQ_ID").groups.items():
        sub = df.loc[gidx].sort_values("__adj_price").copy()
        n = len(sub)
        cut = max(0, int(n * cut_ratio)) if n > 5 else 0

        if cut > 0:
            keep_mask = np.zeros(n, dtype=bool)
            keep_mask[cut:n - cut] = True
        else:
            keep_mask = np.ones(n, dtype=bool)

        kept_index = sub.index[keep_mask]
        df.loc[kept_index, "DefaultInclude"] = True
        df.loc[kept_index, "Include"] = True

    # BOQ 결과
    results = []
    for boq_id, sub in df.groupby("BOQ_ID"):
        inc = sub[sub["Include"] == True]
        one = sub.iloc[0]

        if inc.empty:
            final_price = None
            reason_text = "매칭 후보 없음(또는 전부 제외)"
        else:
            final_price = float(pd.to_numeric(inc["__adj_price"], errors="coerce").mean())
            reason_text = f"{len(inc)}개 내역 평균(국내DB)"

        results.append({
            "BOQ_ID": int(boq_id),
            "명칭": one.get("BOQ_명칭", ""),
            "규격": one.get("BOQ_규격", ""),
            "단위": one.get("BOQ_단위", ""),
            "수량": one.get("BOQ_수량", ""),
            "Final Price": f"{final_price:,.2f}" if final_price is not None else None,
            "산출근거": reason_text,
        })

    result_df = pd.DataFrame(results).sort_values("BOQ_ID").reset_index(drop=True)

    log_cols = [
        "BOQ_ID","BOQ_명칭","BOQ_규격","BOQ_단위","BOQ_수량","BOQ_단가",
        "Include","DefaultInclude",
        "현장코드","현장명","현장특성",
        "실행명칭","규격","단위","수량",
        "계약단가","보정단가","계약월",
        "업체코드","업체명",
        "공종Code분류","세부분류",
        "__adj_price","__hyb",
    ]
    for c in log_cols:
        if c not in df.columns:
            df[c] = None
    log_df = df[log_cols].copy()

    return result_df, log_df


# =========================
# 🤖 Include 자동 추천 에이전트(룰 기반)
# =========================
def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def suggest_include_for_one_boq(
    df_boq: pd.DataFrame,
    mode: str = "균형",
    min_keep: int = 3,
    max_keep: int = 50,
):
    d = df_boq.copy()

    hyb = _to_num(d.get("__hyb", 0)).fillna(0.0)
    price = _to_num(d.get("__adj_price", np.nan))

    if mode == "보수적":
        hyb_min = 80
        iqr_k = 1.0
    elif mode == "공격적":
        hyb_min = 60
        iqr_k = 2.0
    else:  # 균형
        hyb_min = 70
        iqr_k = 1.5

    keep = hyb >= hyb_min

    valid = price[price.notna()]
    low = high = None
    if len(valid) >= 5:
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        low = q1 - iqr_k * iqr
        high = q3 + iqr_k * iqr
        keep = keep & (price.between(low, high) | price.isna())

    keep_idx = d.index[keep].tolist()
    if len(keep_idx) < int(min_keep):
        top_idx = hyb.sort_values(ascending=False).head(int(min_keep)).index.tolist()
        keep_idx = sorted(set(keep_idx) | set(top_idx))

    if len(keep_idx) > int(max_keep):
        keep_idx = hyb.loc[keep_idx].sort_values(ascending=False).head(int(max_keep)).index.tolist()

    include = pd.Series(False, index=d.index)
    include.loc[keep_idx] = True

    reasons = []
    for idx in d.index:
        r = []
        if hyb.loc[idx] < hyb_min:
            r.append(f"유사도<{hyb_min}")
        if low is not None and high is not None and pd.notna(price.loc[idx]):
            if price.loc[idx] < low or price.loc[idx] > high:
                r.append("단가이상치(IQR)")
        if include.loc[idx]:
            reasons.append("포함" if not r else "포함(예외보완): " + ", ".join(r))
        else:
            reasons.append("제외" if not r else "제외: " + ", ".join(r))

    summary = {
        "mode": mode,
        "hyb_min": hyb_min,
        "iqr_k": iqr_k,
        "min_keep": int(min_keep),
        "max_keep": int(max_keep),
        "kept": int(include.sum()),
        "total": int(len(d)),
    }
    return include, pd.Series(reasons, index=d.index), summary


def apply_agent_to_log(
    log_all: pd.DataFrame,
    boq_id: int,
    mode: str = "균형",
    min_keep: int = 3,
    max_keep: int = 50,
):
    mask = log_all["BOQ_ID"].astype(int) == int(boq_id)
    sub = log_all.loc[mask].copy()
    if sub.empty:
        return log_all, None

    inc, reason_s, summary = suggest_include_for_one_boq(sub, mode=mode, min_keep=min_keep, max_keep=max_keep)

    if "AI_추천사유" not in log_all.columns:
        log_all["AI_추천사유"] = ""
    if "AI_모드" not in log_all.columns:
        log_all["AI_모드"] = ""

    log_all.loc[mask, "Include"] = inc.values
    log_all.loc[mask, "AI_추천사유"] = reason_s.values
    log_all.loc[mask, "AI_모드"] = mode

    return log_all, summary


def apply_agent_to_all_boqs(
    log_all: pd.DataFrame,
    mode: str = "균형",
    min_keep: int = 3,
    max_keep: int = 50,
):
    rows = []
    for boq_id in sorted(log_all["BOQ_ID"].dropna().astype(int).unique().tolist()):
        log_all, summary = apply_agent_to_log(log_all, boq_id, mode=mode, min_keep=min_keep, max_keep=max_keep)
        if summary:
            rows.append([boq_id, summary["kept"], summary["total"], summary["mode"]])
    sum_df = pd.DataFrame(rows, columns=["BOQ_ID", "포함수", "후보수", "모드"])
    return log_all, sum_df


# =========================
# 📝 근거 보고서 생성(요약/상세)
# =========================

REPORT_SUMMARY_RENAME = {
    "BOQ_내역": "BOQ 내역",
    "BOQ_Unit": "단위",
    "후보수": "후보수",
    "포함수": "포함수",
    "포함국가": "포함국가",
    "포함현장수": "현장수",
    "포함업체수": "업체수",
    "산출단가평균": "산출단가(평균)",
    "산출단가표준편차": "표준편차",
    "산출단가최저": "최저",
    "산출단가최고": "최고",
    "최빈현장": "최빈현장",
    "최빈업체": "최빈업체",
    "리스크": "리스크",
    "Final Price": "Final Price",
    "산출근거": "산출근거",
    "근거공종(최빈)": "근거공종(최빈)",
}

REPORT_SUMMARY_ORDER = [
    "BOQ_ID",
    "BOQ 내역",
    "단위",
    "Final Price",
    "산출근거",
    "근거공종(최빈)",
    "후보수",
    "포함수",
    "포함국가",
    "현장수",
    "업체수",
    "산출단가(평균)",
    "표준편차",
    "최저",
    "최고",
    "최빈현장",
    "최빈업체",
    "리스크",
]

REPORT_DETAIL_RENAME = {
    # BOQ
    "BOQ_내역": "BOQ 내역",
    "BOQ_Unit": "단위",

    # 가격/보정/유사도
    "Unit Price": "원단가",
    "통화": "원통화",
    "계약년월": "계약년월",
    "__adj_price": "산출단가",
    "산출통화": "산출통화",
    "__cpi_ratio": "CPI 지수",
    "__latest_ym": "적용년월",
    "__fx_ratio": "적용환율",
    "__fac_ratio": "Location Factor",
    "__hyb": "유사도",

    # 코드/현장/업체
    "공종코드": "공종코드",
    "공종명": "공종명",
    "현장코드": "현장코드",
    "현장명": "현장명",
    "협력사코드": "협력사코드",
    "협력사명": "협력사명",

    # AI
    "AI_모드": "AI 모드",
    "AI_추천사유": "AI 추천사유",
}

# 상세는 "원하는 것만 남기는" 방식이 가장 깔끔합니다.
REPORT_DETAIL_COLS = [
    "BOQ_ID", "BOQ_내역", "BOQ_Unit",
    "Unit Price", "통화", "계약년월",
    "__adj_price", "산출통화",
    "__cpi_ratio", "__latest_ym", "__fx_ratio", "__fac_ratio", "__hyb",
    "공종코드", "공종명",
    "현장코드", "현장명",
    "협력사코드", "협력사명",
    "AI_모드", "AI_추천사유",
]

REPORT_DETAIL_ORDER = [
    "BOQ_ID",
    "BOQ 내역",
    "단위",
    "원단가",
    "원통화",
    "계약년월",
    "산출단가",
    "산출통화",
    "CPI 지수",
    "적용년월",
    "적용환율",
    "Location Factor",
    "유사도",
    "공종코드",
    "공종명",
    "현장코드",
    "현장명",
    "협력사코드",
    "협력사명",
    "AI 모드",
    "AI 추천사유",
]

def build_report_tables(log_df: pd.DataFrame, result_df: pd.DataFrame):
    if log_df is None or log_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = log_df.copy()
    df["BOQ_ID"] = df["BOQ_ID"].astype(int)

    inc = df[df["Include"] == True].copy()

    # =========================
    # (1) 상세(detail)
    # =========================
    for c in REPORT_DETAIL_COLS:
        if c not in inc.columns:
            inc[c] = None

    detail_df = inc[REPORT_DETAIL_COLS].copy()
    detail_df = detail_df.rename(columns=REPORT_DETAIL_RENAME)

    # 정렬(존재하는 컬럼만)
    exist_detail = [c for c in REPORT_DETAIL_ORDER if c in detail_df.columns]
    remain_detail = [c for c in detail_df.columns if c not in exist_detail]
    detail_df = detail_df[exist_detail + remain_detail].copy()

    # =========================
    # (2) 요약(summary)
    # =========================
    rows = []
    for boq_id, g in df.groupby("BOQ_ID"):
        g_inc = g[g["Include"] == True].copy()
        total_n = len(g)
        inc_n = len(g_inc)

        adj = pd.to_numeric(g_inc.get("__adj_price", np.nan), errors="coerce")
        mean = float(adj.mean()) if inc_n else np.nan
        std = float(adj.std(ddof=0)) if inc_n else np.nan
        vmin = float(adj.min()) if inc_n else np.nan
        vmax = float(adj.max()) if inc_n else np.nan

        countries = sorted(g_inc["통화"].astype(str).str.upper().unique().tolist()) if inc_n else []
        sites = g_inc["현장코드"].astype(str).nunique() if inc_n and "현장코드" in g_inc.columns else 0
        vendors = g_inc["협력사코드"].astype(str).nunique() if inc_n and "협력사코드" in g_inc.columns else 0

        top_site = ""
        top_vendor = ""
        if inc_n and "현장코드" in g_inc.columns:
            vc = g_inc["현장코드"].astype(str).value_counts()
            top_site = f"{vc.index[0]} ({int(vc.iloc[0])}/{inc_n})" if len(vc) else ""
        if inc_n and "협력사코드" in g_inc.columns:
            vc2 = g_inc["협력사코드"].astype(str).value_counts()
            top_vendor = f"{vc2.index[0]} ({int(vc2.iloc[0])}/{inc_n})" if len(vc2) else ""

        risk = []
        if inc_n == 0:
            risk.append("포함후보없음")
        if inc_n and pd.notna(vmax) and pd.notna(vmin) and vmin > 0 and (vmax / vmin > 3):
            risk.append("단가편차큼(>3배)")
        if inc_n and pd.notna(std) and pd.notna(mean) and mean != 0 and (std / mean > 0.5):
            risk.append("변동성큼(CV>0.5)")
        if inc_n and sites == 1 and inc_n >= 3:
            risk.append("현장편향(1개현장)")
        if inc_n and vendors == 1 and inc_n >= 3:
            risk.append("업체편향(1개업체)")

        one = g.iloc[0]
        rows.append({
            "BOQ_ID": int(boq_id),
            "BOQ_내역": one.get("BOQ_내역", ""),
            "BOQ_Unit": one.get("BOQ_Unit", ""),
            "후보수": int(total_n),
            "포함수": int(inc_n),
            "포함국가": ", ".join(countries),
            "포함현장수": int(sites),
            "포함업체수": int(vendors),
            "산출단가평균": mean,
            "산출단가표준편차": std,
            "산출단가최저": vmin,
            "산출단가최고": vmax,
            "최빈현장": top_site,
            "최빈업체": top_vendor,
            "리스크": ", ".join(risk),
        })

    summary_df = pd.DataFrame(rows).sort_values("BOQ_ID").reset_index(drop=True)

    # result_df의 일부 컬럼 병합
    if result_df is not None and not result_df.empty and "BOQ_ID" in result_df.columns:
        tmp = result_df.copy()
        tmp["BOQ_ID"] = tmp["BOQ_ID"].astype(int)
        keep = [c for c in ["BOQ_ID", "Final Price", "산출근거", "근거공종(최빈)"] if c in tmp.columns]
        if keep:
            summary_df = summary_df.merge(tmp[keep], on="BOQ_ID", how="left")

    # 요약 rename + 정렬
    summary_df = summary_df.rename(columns=REPORT_SUMMARY_RENAME)
    exist_sum = [c for c in REPORT_SUMMARY_ORDER if c in summary_df.columns]
    remain_sum = [c for c in summary_df.columns if c not in exist_sum]
    summary_df = summary_df[exist_sum + remain_sum].copy()

    return summary_df, detail_df

def build_report_tables_domestic(log_df: pd.DataFrame, result_df: pd.DataFrame):
    """
    국내 근거 보고서 테이블 생성(요약/상세)
    - 해외 TAB3 형식과 동일한 섹션 구성을 만들기 위한 summary/detail 2개 테이블 반환
    """
    if log_df is None or log_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = log_df.copy()
    df["BOQ_ID"] = df["BOQ_ID"].astype(int)

    # Include=True 상세 후보
    inc = df[df["Include"] == True].copy()

    # -------------------------
    # (1) 상세(detail)
    # -------------------------
    detail_cols = [
        "BOQ_ID", "BOQ_명칭", "BOQ_규격", "BOQ_단위",
        "실행명칭", "규격", "단위",
        "__adj_price", "__hyb",
        "계약월", "보정단가", "계약단가",
        "현장코드", "현장명", "현장특성",
        "업체코드", "업체명",
        "공종Code분류", "세부분류",
        "AI_모드", "AI_추천사유",
    ]
    for c in detail_cols:
        if c not in inc.columns:
            inc[c] = None
    detail_df = inc[detail_cols].copy()

    # ✅ (표시용) 열 제목 변경
    detail_rename = {
        "BOQ_ID": "BOQ 번호",
        "BOQ_명칭": "BOQ 명칭",
        "BOQ_규격": "BOQ 규격",
        "BOQ_단위": "BOQ 단위",
    
        "실행명칭": "실행명칭",
        "규격": "규격",
        "단위": "단위",
    
        "__adj_price": "산출단가",
        "__hyb": "유사도(%)",
    
        "계약월": "계약월",
        "보정단가": "보정단가",
        "계약단가": "계약단가",
    
        "현장코드": "현장코드",
        "현장명": "현장명",
        "현장특성": "현장특성",
    
        "업체코드": "업체코드",
        "업체명": "업체명",
    
        "공종Code분류": "공종분류",
        "세부분류": "세부분류",
    
        "AI_모드": "AI 모드",
        "AI_추천사유": "AI 추천사유",
    }
    
    detail_df = detail_df.rename(columns=detail_rename)

    # -------------------------
    # (2) 요약(summary)
    # -------------------------
    rows = []
    for boq_id, g in df.groupby("BOQ_ID"):
        g_inc = g[g["Include"] == True].copy()
        total_n = len(g)
        inc_n = len(g_inc)

        adj = pd.to_numeric(g_inc.get("__adj_price", np.nan), errors="coerce")
        mean = float(adj.mean()) if inc_n else np.nan
        std = float(adj.std(ddof=0)) if inc_n else np.nan
        vmin = float(adj.min()) if inc_n else np.nan
        vmax = float(adj.max()) if inc_n else np.nan

        sites = g_inc["현장코드"].astype(str).nunique() if inc_n and "현장코드" in g_inc.columns else 0
        vendors = g_inc["업체코드"].astype(str).nunique() if inc_n and "업체코드" in g_inc.columns else 0

        top_site = ""
        top_vendor = ""
        if inc_n and "현장코드" in g_inc.columns:
            vc = g_inc["현장코드"].astype(str).value_counts()
            top_site = f"{vc.index[0]} ({int(vc.iloc[0])}/{inc_n})" if len(vc) else ""
        if inc_n and "업체코드" in g_inc.columns:
            vc2 = g_inc["업체코드"].astype(str).value_counts()
            top_vendor = f"{vc2.index[0]} ({int(vc2.iloc[0])}/{inc_n})" if len(vc2) else ""

        risk = []
        if inc_n == 0:
            risk.append("포함후보없음")
        if inc_n and pd.notna(vmax) and pd.notna(vmin) and vmin > 0 and (vmax / vmin > 3):
            risk.append("단가편차큼(>3배)")
        if inc_n and pd.notna(std) and pd.notna(mean) and mean != 0 and (std / mean > 0.5):
            risk.append("변동성큼(CV>0.5)")
        if inc_n and sites == 1 and inc_n >= 3:
            risk.append("현장편향(1개현장)")
        if inc_n and vendors == 1 and inc_n >= 3:
            risk.append("업체편향(1개업체)")

        one = g.iloc[0]
        rows.append({
            "BOQ_ID": int(boq_id),
            "BOQ_명칭": one.get("BOQ_명칭", ""),
            "BOQ_규격": one.get("BOQ_규격", ""),
            "BOQ_단위": one.get("BOQ_단위", ""),
            "후보수": int(total_n),
            "포함수": int(inc_n),
            "포함현장수": int(sites),
            "포함업체수": int(vendors),
            "산출단가평균": mean,
            "산출단가표준편차": std,
            "산출단가최저": vmin,
            "산출단가최고": vmax,
            "최빈현장": top_site,
            "최빈업체": top_vendor,
            "리스크": ", ".join(risk),
        })

    summary_df = pd.DataFrame(rows).sort_values("BOQ_ID").reset_index(drop=True)

    # 결과(result_df)의 Final Price/산출근거 병합(있으면)
    if result_df is not None and not result_df.empty and "BOQ_ID" in result_df.columns:
        tmp = result_df.copy()
        tmp["BOQ_ID"] = tmp["BOQ_ID"].astype(int)
        keep = [c for c in ["BOQ_ID", "Final Price", "산출근거"] if c in tmp.columns]
        if keep:
            summary_df = summary_df.merge(tmp[keep], on="BOQ_ID", how="left")

    return summary_df, detail_df


# =========================
# 🤖 AI 최종 적용 기준 기록/표시용 (TAB3에서 사용)
# =========================
def record_ai_last_applied(
    scope: str,
    mode: str,
    min_keep: int,
    max_keep: int,
    summary: Optional[dict] = None,
    boq_id: Optional[int] = None,
):
    payload = {
        "scope": str(scope),
        "mode": str(mode),
        "min_keep": int(min_keep),
        "max_keep": int(max_keep),
    }
    if boq_id is not None:
        payload["boq_id"] = int(boq_id)

    if isinstance(summary, dict):
        for k in ["hyb_min", "iqr_k", "kept", "total"]:
            if k in summary:
                payload[k] = summary[k]

    st.session_state["ai_last_applied"] = payload


def get_ai_effective_rule_text() -> str:
    info = st.session_state.get("ai_last_applied", None)
    if not isinstance(info, dict) or not info.get("mode"):
        return "AI 최종기준 기록 없음(수동 편집 또는 기본 컷만 적용)"

    scope = info.get("scope", "")
    mode = info.get("mode", "")
    min_keep = info.get("min_keep", "")
    max_keep = info.get("max_keep", "")
    boq_id = info.get("boq_id", None)
    hyb_min = info.get("hyb_min", None)
    iqr_k = info.get("iqr_k", None)

    parts = []
    if scope == "현재 BOQ" and boq_id is not None:
        parts.append(f"적용범위={scope}(BOQ_ID={boq_id})")
    else:
        parts.append(f"적용범위={scope}")

    parts.append(f"모드={mode}")
    parts.append(f"최소포함={min_keep}")
    parts.append(f"최대포함={max_keep}")

    if hyb_min is not None:
        parts.append(f"유사도최소(hyb_min)={hyb_min}")
    if iqr_k is not None:
        parts.append(f"IQR계수(iqr_k)={iqr_k}")

    return " / ".join(parts)


# =========================
# 🧾 보고서 TAB3 유틸(특성/현장/AI기준/분포 그래프)
# =========================
def build_feature_context_table(feature_master: pd.DataFrame, selected_feature_ids: list) -> pd.DataFrame:
    if not selected_feature_ids:
        return pd.DataFrame(columns=["특성ID", "대공종", "중공종", "소공종", "Cost Driver Method", "Cost Driver Condition"])

    fm = feature_master.copy()
    cols5 = ["대공종", "중공종", "소공종", "Cost Driver Method", "Cost Driver Condition"]
    keep = ["특성ID"] + cols5

    for c in keep:
        if c in fm.columns:
            fm[c] = fm[c].astype(str).fillna("").str.strip()
        else:
            fm[c] = ""

    out = fm[fm["특성ID"].astype(str).isin([str(x) for x in selected_feature_ids])][keep].copy()
    out = out.drop_duplicates(subset=["특성ID"]).reset_index(drop=True)
    return out


def build_site_context_table(cost_db: pd.DataFrame, selected_site_codes: list) -> pd.DataFrame:
    if not selected_site_codes:
        return pd.DataFrame(columns=["현장코드", "현장명"])
    tmp = cost_db[["현장코드", "현장명"]].copy()
    tmp = tmp.dropna(subset=["현장코드"])
    tmp["현장코드"] = tmp["현장코드"].apply(norm_site_code)
    tmp["현장명"] = tmp["현장명"].astype(str).fillna("").str.strip()
    tmp.loc[tmp["현장명"].isin(["", "nan", "None"]), "현장명"] = "(현장명없음)"
    tmp = tmp.drop_duplicates(subset=["현장코드"])
    out = tmp[tmp["현장코드"].isin([norm_site_code(x) for x in selected_site_codes])].copy()
    out = out.sort_values("현장코드").reset_index(drop=True)
    return out


def plot_distribution(series: pd.Series, title: str):
    s = pd.to_numeric(series, errors="coerce").dropna()
    fig = plt.figure()
    plt.title(title)
    if len(s) == 0:
        plt.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
    else:
        plt.hist(s.values, bins=30)
        plt.xlabel("산출단가(__adj_price)")
        plt.ylabel("빈도")
    st.pyplot(fig, clear_figure=True)


# =========================
# 📊 BOQ 내역별 산점도(계약년월 vs 산출단가) - 포함/미포함 표시
# =========================
def _parse_contract_month_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().any():
        s2 = s.astype(str).apply(to_year_month_string)
        dt2 = pd.to_datetime(s2, errors="coerce")
        dt = dt.fillna(dt2)
    return dt


def render_boq_scatter(log_df: pd.DataFrame, base_result: pd.DataFrame):
    if log_df is None or log_df.empty:
        st.info("로그 데이터가 없어 그래프를 표시할 수 없습니다.")
        return

    keyword = st.text_input("내역 키워드(예: REBAR)", value="", key="report_kw")
    cand = base_result.copy() if (base_result is not None and not base_result.empty) else None

    if cand is not None and "내역" in cand.columns and "BOQ_ID" in cand.columns and keyword.strip():
        kw = keyword.strip().lower()
        cand = cand[cand["내역"].astype(str).str.lower().str.contains(kw, na=False)].copy()

    if cand is not None and not cand.empty:
        boq_ids = cand["BOQ_ID"].dropna().astype(int).unique().tolist()
        boq_ids = sorted(boq_ids)
        id_to_text = cand.set_index(cand["BOQ_ID"].astype(int))["내역"].astype(str).to_dict()
    else:
        boq_ids = sorted(log_df["BOQ_ID"].dropna().astype(int).unique().tolist())
        id_to_text = (
            log_df.dropna(subset=["BOQ_ID"])
            .assign(BOQ_ID=lambda d: d["BOQ_ID"].astype(int))
            .groupby("BOQ_ID")["BOQ_내역"].first()
            .astype(str).to_dict()
        )

    if not boq_ids:
        st.info("표시할 BOQ_ID가 없습니다.")
        return

    def fmt(x: int) -> str:
        t = id_to_text.get(int(x), "")
        t = (t[:60] + "…") if len(t) > 60 else t
        return f"{int(x)} | {t}"

    sel = st.selectbox("그래프 볼 BOQ 선택", options=boq_ids, format_func=fmt, key="report_boq_pick")

    sub = log_df[log_df["BOQ_ID"].astype(int) == int(sel)].copy()
    if sub.empty:
        st.info("해당 BOQ 후보가 없습니다.")
        return

    sub["계약월_dt"] = _parse_contract_month_series(sub["계약년월"])
    sub["산출단가"] = pd.to_numeric(sub["__adj_price"], errors="coerce")
    sub["포함여부"] = sub["Include"].fillna(False).astype(bool)
    sub["표시내역"] = sub["내역"].astype(str)

    chart = (
        alt.Chart(sub.dropna(subset=["계약월_dt", "산출단가"]))
        .mark_circle()
        .encode(
            x=alt.X("계약월_dt:T", title="계약년월"),
            y=alt.Y("산출단가:Q", title="산출단가(산출통화 기준)"),
            color=alt.Color("포함여부:N", title="포함"),
            size=alt.Size("포함여부:N", title="포함(크기)", scale=alt.Scale(range=[40, 140])),
            tooltip=[
                alt.Tooltip("표시내역:N", title="내역"),
                alt.Tooltip("산출단가:Q", title="산출단가", format=",.4f"),
                alt.Tooltip("통화:N", title="원통화"),
                alt.Tooltip("계약년월:N", title="계약년월"),
                alt.Tooltip("__hyb:Q", title="유사도", format=".2f"),
                alt.Tooltip("현장코드:N", title="현장코드"),
                alt.Tooltip("협력사코드:N", title="협력사코드"),
            ],
        )
        .properties(height=420)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

def render_boq_scatter_domestic(log_df: pd.DataFrame, base_result: pd.DataFrame):
    if log_df is None or log_df.empty:
        st.info("로그 데이터가 없어 그래프를 표시할 수 없습니다.")
        return

    # 해외 TAB3와 같은 UX: 키워드로 BOQ 후보를 줄일 수 있게
    keyword = st.text_input("명칭 키워드(예: 철근)", value="", key="report_kw_kr")

    cand = base_result.copy() if (base_result is not None and not base_result.empty) else None

    # result_df에 "명칭" 컬럼이 있으므로 그걸로 필터
    if cand is not None and "명칭" in cand.columns and "BOQ_ID" in cand.columns and keyword.strip():
        kw = keyword.strip().lower()
        cand = cand[cand["명칭"].astype(str).str.lower().str.contains(kw, na=False)].copy()

    if cand is not None and not cand.empty:
        boq_ids = sorted(cand["BOQ_ID"].dropna().astype(int).unique().tolist())
        id_to_text = cand.set_index(cand["BOQ_ID"].astype(int))["명칭"].astype(str).to_dict()
    else:
        boq_ids = sorted(log_df["BOQ_ID"].dropna().astype(int).unique().tolist())
        id_to_text = (
            log_df.dropna(subset=["BOQ_ID"])
            .assign(BOQ_ID=lambda d: d["BOQ_ID"].astype(int))
            .groupby("BOQ_ID")
            .apply(lambda g: f'{str(g["BOQ_명칭"].iloc[0])} / {str(g["BOQ_규격"].iloc[0])}')
            .to_dict()
        )

    if not boq_ids:
        st.info("표시할 BOQ_ID가 없습니다.")
        return

    def fmt(x: int) -> str:
        t = id_to_text.get(int(x), "")
        t = (t[:60] + "…") if len(t) > 60 else t
        return f"{int(x)} | {t}"

    sel = st.selectbox("그래프 볼 BOQ 선택(국내)", options=boq_ids, format_func=fmt, key="report_boq_pick_kr")

    sub = log_df[log_df["BOQ_ID"].astype(int) == int(sel)].copy()
    if sub.empty:
        st.info("해당 BOQ 후보가 없습니다.")
        return

    # 계약월 파싱
    sub["계약월_dt"] = pd.to_datetime(sub["계약월"], errors="coerce")
    sub["산출단가"] = pd.to_numeric(sub["__adj_price"], errors="coerce")
    sub["포함여부"] = sub["Include"].fillna(False).astype(bool)
    sub["표시내역"] = sub["실행명칭"].astype(str)

    chart = (
        alt.Chart(sub.dropna(subset=["계약월_dt", "산출단가"]))
        .mark_circle()
        .encode(
            x=alt.X("계약월_dt:T", title="계약월"),
            y=alt.Y("산출단가:Q", title="산출단가(국내)"),
            color=alt.Color("포함여부:N", title="포함"),
            size=alt.Size("포함여부:N", title="포함(크기)", scale=alt.Scale(range=[40, 140])),
            tooltip=[
                alt.Tooltip("표시내역:N", title="실행명칭"),
                alt.Tooltip("산출단가:Q", title="산출단가", format=",.4f"),
                alt.Tooltip("__hyb:Q", title="유사도", format=".2f"),
                alt.Tooltip("현장코드:N", title="현장코드"),
                alt.Tooltip("현장명:N", title="현장명"),
                alt.Tooltip("업체코드:N", title="업체코드"),
                alt.Tooltip("업체명:N", title="업체명"),
                alt.Tooltip("계약월:N", title="계약월"),
            ],
        )
        .properties(height=420)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


# =========================
# 데이터 로드
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def load_excel_from_repo(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        st.error(f"필수 파일을 찾을 수 없습니다: {path.as_posix()}")
        st.stop()
    return pd.read_excel(path, engine="openpyxl")


@st.cache_data(show_spinner=False)
def load_overseas_data():
    cost_db = load_excel_from_repo("cost_db.xlsx")
    price_index = load_excel_from_repo("price_index.xlsx")
    exchange = load_excel_from_repo("exchange.xlsx")
    factor = load_excel_from_repo("Factor.xlsx")
    ppp = load_excel_from_repo("PPP Factor.xlsx")
    project_feature_long = load_excel_from_repo("project_feature_long.xlsx")
    feature_master = load_excel_from_repo("feature_master_FID.xlsx")
    return cost_db, price_index, exchange, factor, ppp, project_feature_long, feature_master

@st.cache_data(show_spinner=False)
def load_domestic_data():
    cost_db_kr = load_excel_from_repo("cost_db (kr).xlsx")
    return cost_db_kr

# =========================
# ✅ 컬럼명 표준화 + alias 매핑 (KeyError 방지)
# =========================
def _std_colname(s: str) -> str:
    s = str(s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_std_colname(c) for c in df.columns]
    return df


def apply_feature_column_alias(df: pd.DataFrame) -> pd.DataFrame:
    """
    feature_master_FID / project_feature_long 컬럼이 조금 달라도
    아래 '표준 컬럼명'으로 강제 맞춤
    """
    df = df.copy()
    col_map = {}

    aliases = {
        "특성ID": ["특성ID", "특성 Id", "FeatureID", "Feature Id", "FID"],
        "대공종": ["대공종", "대 공종", "Major", "Main"],
        "중공종": ["중공종", "중 공종", "Middle"],
        "소공종": ["소공종", "소 공종", "Minor", "Sub"],

        "Cost Driver Type": [
            "Cost Driver Type", "CostDriver Type", "Cost DriverType",
            "Cost Driver_Type", "CostDriver_Type", "Type", "Driver Type"
        ],
        "Cost Driver Method": [
            "Cost Driver Method", "CostDriver Method", "Cost DriverMethod",
            "Cost Driver_Method", "CostDriver_Method", "Method"
        ],
        "Cost Driver Condition": [
            "Cost Driver Condition", "CostDriver Condition", "Cost DriverCondition",
            "Cost Driver_Condition", "CostDriver_Condition", "Condition"
        ],

        "현장코드": ["현장코드", "현장 코드", "Site Code", "SiteCode"],
        "현장명": ["현장명", "현장 명", "Site Name", "SiteName"],
    }

    cols = list(df.columns)

    for std_name, cand_list in aliases.items():
        for cand in cand_list:
            cand_std = _std_colname(cand)
            if cand_std in cols:
                col_map[cand_std] = std_name
                break

    df = df.rename(columns=col_map)

    must_cols = [
        "특성ID", "대공종", "중공종", "소공종",
        "Cost Driver Type", "Cost Driver Method", "Cost Driver Condition"
    ]
    for c in must_cols:
        if c not in df.columns:
            df[c] = ""

    return df

# =========================
# ✅ 데이터 로드 + 표준화/alias 적용 (함수 정의 이후에 1회만)
# =========================
cost_db, price_index, exchange, factor, ppp, project_feature_long, feature_master = load_overseas_data()
cost_db_kr = load_domestic_data()

# (권장) DB도 표준화
cost_db = standardize_columns(cost_db)
cost_db_kr = standardize_columns(cost_db_kr)

# feature 관련 표준화/alias
project_feature_long = standardize_columns(project_feature_long)
feature_master = standardize_columns(feature_master)

project_feature_long = apply_feature_column_alias(project_feature_long)
feature_master = apply_feature_column_alias(feature_master)

ppp = ppp.copy()
ppp["Currency"] = ppp["Currency"].astype(str).str.upper().str.strip()

# 연도 컬럼 정리 (문자열 → int)
year_cols = [c for c in ppp.columns if str(c).isdigit()]
ppp_years = sorted([int(c) for c in year_cols])

# =========================
# Session init
# =========================
if "selected_feature_ids" not in st.session_state:
    st.session_state["selected_feature_ids"] = []
if "auto_sites" not in st.session_state:
    st.session_state["auto_sites"] = []

# ✅ 탭 전환 상태(사이드바 중복 렌더 방지용)
if "active_db" not in st.session_state:
    st.session_state["active_db"] = "overseas"


# ============================================================
# ✅ 국내 탭 (UI skeleton only)
# ============================================================
def render_domestic():
    gs_header("📦 국내 실적단가 DB")

    # -------------------------
    # Sidebar: 설정(국내)
    # -------------------------
    st.sidebar.markdown("<div class='sb-major'>⚙️ 설정</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr class='sb-hr'/>", unsafe_allow_html=True)

    # 1) BOQ 업로드
    with st.container(border=True):
        card_title("📤 BOQ 파일 업로드(국내)")
        dom_boq_file = st.file_uploader(
            label="",
            type=["xlsx"],
            key="dom_boq_uploader",
            label_visibility="collapsed",
        )

    # 2) 국내 필터(현장특성/현장)
    # - "해외 특성 선택"은 그대로 두고, 국내는 "현장특성" 기준으로 필터 UI 제공
    kr = cost_db_kr.copy()

    # 현장특성
    feat_col = "현장특성"
    if feat_col not in kr.columns:
        kr[feat_col] = ""

    feat_options = sorted([x for x in kr[feat_col].astype(str).fillna("").unique().tolist() if x.strip() and x != "nan"])
    sel_feat = st.sidebar.multiselect(
        "🏷️ 현장특성",
        options=feat_options,
        default=st.session_state.get("dom_sel_feat", []),
        key="dom_sel_feat",
    )

    if sel_feat:
        kr_view = kr[kr[feat_col].astype(str).isin(sel_feat)].copy()
    else:
        kr_view = kr

    # 현장 선택(국내)
    if "현장코드" not in kr_view.columns:
        kr_view["현장코드"] = ""
    if "현장명" not in kr_view.columns:
        kr_view["현장명"] = ""

    site_df = kr_view[["현장코드", "현장명"]].copy()
    site_df = site_df.dropna(subset=["현장코드"])
    site_df["현장코드"] = site_df["현장코드"].apply(norm_site_code)
    site_df["현장명"] = site_df["현장명"].astype(str).fillna("").str.strip()
    site_df.loc[site_df["현장명"].isin(["", "nan", "None"]), "현장명"] = "(현장명없음)"
    site_df = site_df.drop_duplicates(subset=["현장코드"]).reset_index(drop=True)

    all_codes = site_df["현장코드"].tolist()
    code_to_name = dict(zip(site_df["현장코드"], site_df["현장명"]))

    def fmt_site(code: str) -> str:
        name = code_to_name.get(code, "").strip()
        return (name[:25] + "…") if len(name) > 25 else name

    st.sidebar.markdown(
        f"""
        <div class="sb-row">
          <div class="sb-title">🏗️ 실적 현장 선택</div>
          <div class="sb-muted">가능 현장: {len(all_codes)}개</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("<hr class='sb-hr'/>", unsafe_allow_html=True)

    dom_selected_sites = st.sidebar.multiselect(
        "국내 실적현장",
        options=all_codes,
        default=st.session_state.get("dom_selected_site_codes", []),
        key="dom_selected_site_codes",
        format_func=fmt_site,
    )

    # 3) 설정값
    st.sidebar.markdown("<div class='sb-title'>🧩 설정값</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr class='sb-hr'/>", unsafe_allow_html=True)

    # 해외랑 비슷하게 유지
    DEFAULT_W_STR = 0.35
    w_str = DEFAULT_W_STR
    w_sem = 1.0 - w_str
    top_k_sem = 200

    dom_sim_threshold = st.sidebar.slider("매칭 유사도 기준값(%)", 0, 100, 65, 5, key="dom_sim_threshold")
    dom_cut_ratio = st.sidebar.slider("상/하위 컷 비율 (%)", 0, 30, 20, 5, key="dom_cut_ratio") / 100.0

    sidebar_hr(thick=True, mt=10, mb=8)

    # -------------------------
    # Run / Auto Recompute(국내)
    # -------------------------
    def dom_boq_file_signature(uploaded_file) -> str:
        if uploaded_file is None:
            return "no_boq"
        try:
            b = uploaded_file.getvalue()
            if len(b) > 2_000_000:
                b = b[:1_000_000] + b[-1_000_000:]
            return hashlib.md5(b).hexdigest()
        except Exception:
            return f"{getattr(uploaded_file, 'name', 'boq')}_{getattr(uploaded_file, 'size', '')}"

    def make_dom_params_signature() -> str:
        payload = {
            "boq": dom_boq_file_signature(dom_boq_file),
            "sel_feat": sorted([str(x) for x in (sel_feat or [])]),
            "sel_sites": sorted([norm_site_code(x) for x in (dom_selected_sites or [])]),
            "sim_threshold": float(dom_sim_threshold),
            "cut_ratio": float(dom_cut_ratio),
            "w_str": float(w_str),
            "w_sem": float(w_sem),
            "top_k_sem": int(top_k_sem),
        }
        s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def run_domestic_and_store(run_sig: str):
        status_box = st.empty()

        if dom_boq_file is None:
            status_box.empty()
            st.warning("국내 BOQ 파일을 업로드해 주세요.")
            return

        progress = st.progress(0.0)
        prog_text = st.empty()

        status_box.markdown("### ⏳ [국내] 산출중... (BOQ 로드)")
        boq_kr = pd.read_excel(dom_boq_file, engine="openpyxl")

        # 국내 BOQ 컬럼 보강(첫 열: 명칭 규격 단위 수량 단가)
        need_boq_cols = ["명칭","규격","단위","수량","단가"]
        for c in need_boq_cols:
            if c not in boq_kr.columns:
                boq_kr[c] = ""

        # 국내 DB 필터 적용(현장특성/현장)
        db_run = cost_db_kr.copy()
        if sel_feat:
            db_run = db_run[db_run["현장특성"].astype(str).isin([str(x) for x in sel_feat])].copy()
        if dom_selected_sites:
            db_run = db_run[db_run["현장코드"].apply(norm_site_code).isin([norm_site_code(x) for x in dom_selected_sites])].copy()

        st.sidebar.caption(f"[국내] 전체 {len(cost_db_kr):,}개 중 {len(db_run):,}개로 산출")

        pool_sig_payload = {
            "boq": dom_boq_file_signature(dom_boq_file),
            "sel_feat": sorted([str(x) for x in (sel_feat or [])]),
            "sel_sites": sorted([norm_site_code(x) for x in (dom_selected_sites or [])]),
            "top_k_sem": int(top_k_sem),
            "w_str": float(w_str),
            "w_sem": float(w_sem),
            "db_rows": int(len(db_run)),
        }
        pool_sig = hashlib.md5(json.dumps(pool_sig_payload, sort_keys=True).encode("utf-8")).hexdigest()

        need_new_pool = (st.session_state.get("dom_candidate_pool_sig") != pool_sig) or ("dom_candidate_pool" not in st.session_state)

        if need_new_pool:
            status_box.markdown("### ⏳ [국내] 산출중... (후보 풀 생성)")
            with st.spinner("[국내] 후보 풀 생성 중..."):
                pool = build_candidate_pool_domestic(
                    cost_db_kr=db_run,
                    boq_kr=boq_kr,
                    sim_w_str=w_str,
                    sim_w_sem=w_sem,
                    top_k_sem=top_k_sem,
                    pool_per_boq=400,
                    progress=progress,
                    prog_text=prog_text,
                )
            st.session_state["dom_candidate_pool"] = pool
            st.session_state["dom_candidate_pool_sig"] = pool_sig
        else:
            pool = st.session_state["dom_candidate_pool"]

        status_box.markdown("### ⏳ [국내] 산출중...")
        with st.spinner("[국내] 빠른 재계산 중..."):
            result_df, log_df = fast_recompute_from_pool_domestic(
                pool=pool,
                sim_threshold=dom_sim_threshold,
                cut_ratio=dom_cut_ratio,
            )
        # ✅ 산출 완료 후 진행 텍스트 제거
        try:
            status_box.empty()
        except Exception:
            pass
        try:
            progress.empty()
        except Exception:
            pass
        try:
            prog_text.empty()
        except Exception:
            pass

        st.session_state["dom_boq_df"] = boq_kr
        st.session_state["dom_result_df_base"] = result_df.copy()
        st.session_state["dom_log_df_base"] = log_df.copy()
        st.session_state["dom_log_df_edited"] = log_df.copy()
        st.session_state["dom_has_results"] = True
        st.session_state["dom_last_run_sig"] = run_sig

    run_dom_btn = st.sidebar.button("🚀 산출 실행", key="dom_run_btn")

    cur_sig = make_dom_params_signature()
    last_sig = st.session_state.get("dom_last_run_sig", None)
    needs_rerun = (last_sig is not None and cur_sig != last_sig)

    # 자동재산출은 국내도 ON(원하면 False로 바꾸면 됩니다)
    auto_recompute = True
    auto_run = st.session_state.get("dom_has_results", False) and needs_rerun and auto_recompute

    if run_dom_btn or auto_run:
        run_domestic_and_store(cur_sig)

    # -------------------------
    # Tabs(국내)
    # -------------------------
    tab1, tab2, tab3 = st.tabs(["📄 BOQ 결과", "🧾 산출 근거(편집 가능)", "📝 근거 보고서"])

    with tab2:
        if not st.session_state.get("dom_has_results", False):
            st.info("산출 실행 후 로그가 표시됩니다.")
        else:
            
    
            # 현재 편집 대상: 전체 로그(edited)
            if "dom_log_df_edited" not in st.session_state:
                st.session_state["dom_log_df_edited"] = st.session_state.get("dom_log_df_base", pd.DataFrame()).copy()
    
            log_all = st.session_state["dom_log_df_edited"]
            if log_all is None or log_all.empty:
                st.warning("로그 데이터가 없습니다.")
            else:
                # --- BOQ 선택 ---
                boq_ids = sorted(log_all["BOQ_ID"].dropna().astype(int).unique().tolist())
    
                # 라벨(BOQ_ID | 명칭/규격) 만들기
                id_to_text = (
                    log_all.dropna(subset=["BOQ_ID"])
                    .assign(BOQ_ID=lambda d: d["BOQ_ID"].astype(int))
                    .groupby("BOQ_ID")
                    .apply(lambda g: f'{str(g["BOQ_명칭"].iloc[0])} / {str(g["BOQ_규격"].iloc[0])}')
                    .to_dict()
                )
    
                def fmt_boq_id(x: int) -> str:
                    t = id_to_text.get(int(x), "")
                    t = (t[:60] + "…") if len(t) > 60 else t
                    return f"{int(x)} | {t}"
    
                sel_id = st.selectbox(
                    "편집할 BOQ 선택(국내)",
                    options=boq_ids,
                    format_func=fmt_boq_id,
                    key="dom_sel_boq_id",
                )
    
                # 현재 BOQ 후보만 보기
                log_view_full = log_all[log_all["BOQ_ID"].astype(int) == int(sel_id)].copy()
                if log_view_full.empty:
                    st.info("해당 BOQ 후보가 없습니다.")
                else:
                    # --- 백업(되돌리기) 저장소 ---
                    if "dom_include_backup" not in st.session_state:
                        st.session_state["dom_include_backup"] = {}
                    if "dom_include_backup_all" not in st.session_state:
                        st.session_state["dom_include_backup_all"] = None
    
                    # --- AI 파라미터(UI) ---
                    cA, cB, cC, cD = st.columns([1.2, 1.0, 1.0, 1.8])
                    with cA:
                        agent_mode = st.selectbox("AI 추천 모드(국내)", ["보수적", "균형", "공격적"], index=1, key="dom_agent_mode")
                    with cB:
                        min_keep = st.number_input("최소 포함", min_value=1, max_value=20, value=3, step=1, key="dom_agent_min_keep")
                    with cC:
                        max_keep = st.number_input("최대 포함", min_value=3, max_value=200, value=50, step=1, key="dom_agent_max_keep")
                    with cD:
                        st.caption("※ 적용 후 화면이 자동 갱신됩니다.")
    
                    # --- 버튼(해외와 동일) ---
                    b1, b2, b3, b4 = st.columns([1.2, 1.2, 1.2, 2.4])
                    with b1:
                        btn_ai_one = st.button("🤖 AI 적용(현재 BOQ)", key="dom_btn_ai_one")
                    with b2:
                        btn_undo_one = st.button("↩️ 되돌리기(현재 BOQ)", key="dom_btn_undo_one")
                    with b3:
                        btn_ai_all = st.button("🤖 AI 적용(전체 BOQ)", key="dom_btn_ai_all")
                    with b4:
                        btn_undo_all = st.button("↩️ 되돌리기(전체 BOQ)", key="dom_btn_undo_all")
                    
                    # --- 되돌리기(현재 BOQ) ---
                    if btn_undo_one:
                        backup = st.session_state["dom_include_backup"].get(int(sel_id))
                        if backup is not None and len(backup) == len(log_view_full.index):
                            st.session_state["dom_log_df_edited"].loc[log_view_full.index, "Include"] = backup.values
                            st.session_state["dom_result_df_adjusted"] = recompute_dom_result_from_log(st.session_state["dom_log_df_edited"])
                            st.success("되돌리기 완료(현재 BOQ)")
                            st.rerun()
                        else:
                            st.warning("되돌릴 백업이 없습니다(또는 후보행이 변경됨).")
                    
                    # --- AI 적용(현재 BOQ) ---
                    if btn_ai_one:
                        # 현재 BOQ Include 백업
                        st.session_state["dom_include_backup"][int(sel_id)] = st.session_state["dom_log_df_edited"].loc[log_view_full.index, "Include"].copy()
                    
                        updated, summary = apply_agent_to_log(
                            log_all=st.session_state["dom_log_df_edited"].copy(),
                            boq_id=int(sel_id),
                            mode=agent_mode,
                            min_keep=int(min_keep),
                            max_keep=int(max_keep),
                        )
                        st.session_state["dom_log_df_edited"] = updated
                        st.session_state["dom_result_df_adjusted"] = recompute_dom_result_from_log(st.session_state["dom_log_df_edited"])
                        if summary:
                            st.success(f"AI 적용 완료(현재 BOQ): {summary['kept']}/{summary['total']} 포함, 모드={summary['mode']}")
                    
                        # <-- 안전하게 summary 전달
                        record_ai_last_applied("현재 BOQ", agent_mode, int(min_keep), int(max_keep), summary, boq_id=int(sel_id))
                        st.rerun()
                    
                    # --- AI 적용(전체 BOQ) ---
                    if btn_ai_all:
                        st.session_state["dom_include_backup_all"] = st.session_state["dom_log_df_edited"][["BOQ_ID", "Include"]].copy()
                    
                        updated, sum_df = apply_agent_to_all_boqs(
                            log_all=st.session_state["dom_log_df_edited"].copy(),
                            mode=agent_mode,
                            min_keep=int(min_keep),
                            max_keep=int(max_keep),
                        )
                        st.session_state["dom_log_df_edited"] = updated
                        st.session_state["dom_result_df_adjusted"] = recompute_dom_result_from_log(st.session_state["dom_log_df_edited"])
                        st.success("AI 적용 완료(전체 BOQ)")
                        if sum_df is not None and not sum_df.empty:
                            st.dataframe(sum_df, use_container_width=True)
                    
                        # 전체 적용은 summary가 없으므로 None 전달
                        record_ai_last_applied("전체 BOQ", agent_mode, int(min_keep), int(max_keep), None)
                        st.rerun()
    
                    # --- 결과 재계산(Include 기반) 함수 ---
                    def recompute_dom_result_from_log(cur_log: pd.DataFrame) -> pd.DataFrame:
                        rows = []
                        for boq_id, g in cur_log.groupby("BOQ_ID"):
                            g2 = g[g["Include"] == True]
                            one = g.iloc[0]
                            if g2.empty:
                                price = None
                                reason = "매칭 후보 없음(또는 전부 제외)"
                            else:
                                price = float(pd.to_numeric(g2["__adj_price"], errors="coerce").mean())
                                reason = f"{len(g2)}개 내역 평균(국내DB)"
                            rows.append({
                                "BOQ_ID": int(boq_id),
                                "명칭": one.get("BOQ_명칭", ""),
                                "규격": one.get("BOQ_규격", ""),
                                "단위": one.get("BOQ_단위", ""),
                                "수량": one.get("BOQ_수량", ""),
                                "Final Price": f"{price:,.2f}" if price is not None else None,
                                "산출근거": reason,
                            })
                        return pd.DataFrame(rows).sort_values("BOQ_ID").reset_index(drop=True)
    
                    # --- 되돌리기(현재 BOQ) ---
                    if btn_undo_one:
                        backup = st.session_state["dom_include_backup"].get(int(sel_id))
                        if backup is not None and len(backup) == len(log_view_full.index):
                            st.session_state["dom_log_df_edited"].loc[log_view_full.index, "Include"] = backup.values
                            st.session_state["dom_result_df_adjusted"] = recompute_dom_result_from_log(st.session_state["dom_log_df_edited"])
                            st.success("되돌리기 완료(현재 BOQ)")
                            st.rerun()
                        else:
                            st.warning("되돌릴 백업이 없습니다(또는 후보행이 변경됨).")
    
                    # --- AI 적용(현재 BOQ) ---
                    if btn_ai_one:
                        # 현재 BOQ Include 백업
                        st.session_state["dom_include_backup"][int(sel_id)] = st.session_state["dom_log_df_edited"].loc[log_view_full.index, "Include"].copy()
    
                        updated, summary = apply_agent_to_log(
                            log_all=st.session_state["dom_log_df_edited"].copy(),
                            boq_id=int(sel_id),
                            mode=agent_mode,
                            min_keep=int(min_keep),
                            max_keep=int(max_keep),
                        )
                        st.session_state["dom_log_df_edited"] = updated
                        st.session_state["dom_result_df_adjusted"] = recompute_dom_result_from_log(st.session_state["dom_log_df_edited"])
                        if summary:
                            st.success(f"AI 적용 완료(현재 BOQ): {summary['kept']}/{summary['total']} 포함, 모드={summary['mode']}")
                        st.rerun()
    
                    # --- AI 적용(전체 BOQ) ---
                    if btn_ai_all:
                        st.session_state["dom_include_backup_all"] = st.session_state["dom_log_df_edited"][["BOQ_ID", "Include"]].copy()
    
                        updated, sum_df = apply_agent_to_all_boqs(
                            log_all=st.session_state["dom_log_df_edited"].copy(),
                            mode=agent_mode,
                            min_keep=int(min_keep),
                            max_keep=int(max_keep),
                        )
                        st.session_state["dom_log_df_edited"] = updated
                        st.session_state["dom_result_df_adjusted"] = recompute_dom_result_from_log(st.session_state["dom_log_df_edited"])
                        st.success("AI 적용 완료(전체 BOQ)")
                        if sum_df is not None and not sum_df.empty:
                            st.dataframe(sum_df, use_container_width=True)
                        st.rerun()
    
                    # --- 되돌리기(전체 BOQ) ---
                    if btn_undo_all:
                        backup_all = st.session_state.get("dom_include_backup_all")
                        if backup_all is None or backup_all.empty:
                            st.warning("되돌릴 전체 백업이 없습니다.")
                        else:
                            cur = st.session_state["dom_log_df_edited"].copy()
                            b = backup_all.copy()
                            b["BOQ_ID"] = b["BOQ_ID"].astype(int)
                            cur["BOQ_ID"] = cur["BOQ_ID"].astype(int)
    
                            cur = cur.drop(columns=["Include"], errors="ignore").merge(b, on="BOQ_ID", how="left")
                            cur["Include"] = cur["Include"].fillna(False).astype(bool)
    
                            st.session_state["dom_log_df_edited"] = cur
                            st.session_state["dom_result_df_adjusted"] = recompute_dom_result_from_log(st.session_state["dom_log_df_edited"])
                            st.success("되돌리기 완료(전체 BOQ)")
                            st.rerun()

                    # =========================
                    # (국내) 필터/컷 조정 UI (현재 BOQ)
                    # =========================
                    # 필터 대상 컬럼 보강
                    for c in ["현장명", "세부분류", "__hyb", "__adj_price"]:
                        if c not in log_view_full.columns:
                            log_view_full[c] = None
                    
                    # 숫자형 정리
                    log_view_full["__hyb_num"] = pd.to_numeric(log_view_full["__hyb"], errors="coerce").fillna(0.0)
                    log_view_full["__price_num"] = pd.to_numeric(log_view_full["__adj_price"], errors="coerce").fillna(np.nan)
                    
                    with st.expander("🔎 필터(현장명/세부분류/유사도) + 컷 비율 조정", expanded=True):
                        # 1) 현장명 필터
                        site_opts = sorted([
                            x for x in log_view_full["현장명"].astype(str).fillna("").unique().tolist()
                            if x.strip() and x not in ["nan", "None"]
                        ])
                        sel_sites_nm = st.multiselect(
                            "현장명 필터(선택 시 해당 현장만 표시/적용)",
                            options=site_opts,
                            default=st.session_state.get("dom_f_site_nm", []),
                            key="dom_f_site_nm",
                        )
                    
                        # 2) 세부분류 필터
                        sub_opts = sorted([
                            x for x in log_view_full["세부분류"].astype(str).fillna("").unique().tolist()
                            if x.strip() and x not in ["nan", "None"]
                        ])
                        sel_sub = st.multiselect(
                            "세부분류 필터(선택 시 해당 분류만 표시/적용)",
                            options=sub_opts,
                            default=st.session_state.get("dom_f_sub", []),
                            key="dom_f_sub",
                        )
                    
                        # 3) 유사도 기준(현재 BOQ 전용: 사이드바 값과 분리)
                        hyb_thr_tab2 = st.slider(
                            "매칭 유사도 기준값(현재 내역 기준, %)",
                            min_value=0,
                            max_value=100,
                            value=int(st.session_state.get("dom_sim_threshold_tab2", st.session_state.get("dom_sim_threshold", 80))),
                            step=5,
                            key="dom_sim_threshold_tab2",
                        )
                    
                        # 4) 상/하위 컷 비율(현재 BOQ 전용)
                        cut_pct = st.slider(
                            "상/하위 컷 비율(현재 BOQ, %)",
                            min_value=0,
                            max_value=30,
                            value=int(st.session_state.get("dom_cut_pct_tab2", 20)),
                            step=5,
                            key="dom_cut_pct_tab2",
                        )
                        cut_ratio_local = float(cut_pct) / 100.0
                    
                        cbtn1, cbtn2, cbtn3 = st.columns([1.4, 1.2, 1.4])
                        with cbtn1:
                            btn_apply_filter_cut = st.button("✂️ 필터+컷 적용(Include 자동 재설정)", key="dom_btn_apply_filter_cut")
                        with cbtn2:
                            btn_reset_to_default = st.button("↩️ DefaultInclude로 초기화(현재 BOQ)", key="dom_btn_reset_default")
                        with cbtn3:
                            st.caption("※ ‘필터+컷 적용’은 현재 BOQ의 Include를 필터 결과 기준으로 다시 세팅합니다.")
                    
                    # --- 필터 마스크 생성(표시 + 컷 적용에 공통 사용) ---
                    mask = pd.Series(True, index=log_view_full.index)
                    
                    if sel_sites_nm:
                        mask &= log_view_full["현장명"].astype(str).isin([str(x) for x in sel_sites_nm])
                    
                    if sel_sub:
                        mask &= log_view_full["세부분류"].astype(str).isin([str(x) for x in sel_sub])
                    
                    thr = float(st.session_state.get(
                        "dom_sim_threshold_tab2",
                        st.session_state.get("dom_sim_threshold", 65)
                    ))
                    
                    mask &= (log_view_full["__hyb_num"] >= hyb_thr_tab2)
                    
                    # 표시용(필터 적용된 후보만 보여줌)
                    log_view_full_filtered = log_view_full.loc[mask].copy()
                    
                    # --- DefaultInclude 초기화(현재 BOQ) ---
                    if btn_reset_to_default:
                        # 백업 저장(현재 BOQ)
                        st.session_state["dom_include_backup"][int(sel_id)] = st.session_state["dom_log_df_edited"].loc[log_view_full.index, "Include"].copy()
                    
                        # DefaultInclude 기준으로 Include 복원
                        base_inc = st.session_state["dom_log_df_edited"].loc[log_view_full.index, "DefaultInclude"].fillna(False).astype(bool)
                        st.session_state["dom_log_df_edited"].loc[log_view_full.index, "Include"] = base_inc.values
                    
                        st.session_state["dom_result_df_adjusted"] = recompute_dom_result_from_log(st.session_state["dom_log_df_edited"])
                        st.success("현재 BOQ를 DefaultInclude 기준으로 초기화했습니다.")
                        st.rerun()
                    
                    # --- 필터+컷 적용(현재 BOQ) ---
                    if btn_apply_filter_cut:
                        # 백업 저장(현재 BOQ)
                        st.session_state["dom_include_backup"][int(sel_id)] = st.session_state["dom_log_df_edited"].loc[log_view_full.index, "Include"].copy()
                    
                        # 1) 현재 BOQ 전체 Include를 우선 False로
                        st.session_state["dom_log_df_edited"].loc[log_view_full.index, "Include"] = False
                    
                        # 2) 필터 통과 후보만 가지고 컷 적용
                        sub = log_view_full.loc[mask].copy()
                        sub["__price_num"] = pd.to_numeric(sub["__adj_price"], errors="coerce")
                    
                        sub = sub.dropna(subset=["__price_num"]).sort_values("__price_num").copy()
                        n = len(sub)
                        cut = max(0, int(n * cut_ratio_local)) if n > 5 else 0
                    
                        if n == 0:
                            st.warning("필터 조건을 만족하는 후보가 없습니다.")
                        else:
                            if cut > 0:
                                keep_mask = np.zeros(n, dtype=bool)
                                keep_mask[cut:n - cut] = True
                            else:
                                keep_mask = np.ones(n, dtype=bool)
                    
                            kept_index = sub.index[keep_mask]
                            st.session_state["dom_log_df_edited"].loc[kept_index, "Include"] = True
                    
                            # DefaultInclude도 같이 갱신(원하면 제거 가능)
                            st.session_state["dom_log_df_edited"].loc[log_view_full.index, "DefaultInclude"] = False
                            st.session_state["dom_log_df_edited"].loc[kept_index, "DefaultInclude"] = True
                    
                            st.session_state["dom_result_df_adjusted"] = recompute_dom_result_from_log(st.session_state["dom_log_df_edited"])
                            st.success(f"필터+컷 적용 완료: {len(kept_index)}/{n} 포함")
                        st.rerun()
                        
                    # 이후 편집 화면은 '필터된 후보'를 보여주도록 교체
                    log_view_full = log_view_full_filtered
                    
    
                    # --- 화면에 보여줄 컬럼(국내) ---
                    display_cols = [
                        "Include", "DefaultInclude",
                        "실행명칭", "규격", "단위", "수량",
                        "보정단가", "계약단가", "계약월",
                        "__adj_price", "__hyb",
                        "현장코드", "현장명", "현장특성",
                        "업체코드", "업체명",
                        "공종Code분류", "세부분류",
                    ]
                    for c in display_cols:
                        if c not in log_view_full.columns:
                            log_view_full[c] = None
    
                    log_view = log_view_full[display_cols].copy()
    
                    edited_view = st.data_editor(
                        log_view,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Include": st.column_config.CheckboxColumn("포함", help="평균단가 산출 포함/제외"),
                            "DefaultInclude": st.column_config.CheckboxColumn("기본포함", help="초기 자동 포함 여부(컷 로직)"),
                            "__adj_price": st.column_config.NumberColumn("산출단가", format="%.2f"),
                            "__hyb": st.column_config.NumberColumn("유사도", format="%.2f"),
                            "보정단가": st.column_config.NumberColumn("보정단가", format="%.2f"),
                            "계약단가": st.column_config.NumberColumn("계약단가", format="%.2f"),
                        },
                        disabled=[c for c in log_view.columns if c not in ["Include"]],
                        key="dom_log_editor_oneboq",
                    )
    
                    # --- 편집 반영(현재 BOQ rows만) ---
                    st.session_state["dom_log_df_edited"].loc[log_view_full.index, "Include"] = edited_view["Include"].values
    
                    # --- 결과 재계산(Include 반영) ---
                    st.session_state["dom_result_df_adjusted"] = recompute_dom_result_from_log(st.session_state["dom_log_df_edited"])
    
                    # 참고용: 현재 BOQ 포함 후보 수
                    inc_n = int(pd.Series(edited_view["Include"]).sum())
                    st.caption(f"현재 BOQ 포함 후보: {inc_n}개")
    
    
    
    with tab1:
        if not st.session_state.get("dom_has_results", False):
            st.info("국내 BOQ 업로드 후 '산출 실행(국내)'을 눌러주세요.")
        else:
            # ✅ 해외 TAB1과 동일 패턴: adjusted 우선, 없으면 base
            show_df = st.session_state.get(
                "dom_result_df_adjusted",
                st.session_state.get("dom_result_df_base", pd.DataFrame())
            ).copy()
    
            # (선택) 표시용 정리: BOQ_ID 기준 정렬/컬럼 순서 정리 등
            if "BOQ_ID" in show_df.columns:
                try:
                    show_df["BOQ_ID"] = show_df["BOQ_ID"].astype(int)
                    show_df = show_df.sort_values("BOQ_ID").reset_index(drop=True)
                except Exception:
                    pass
    
            st.dataframe(show_df, use_container_width=True)

    
    with tab3:
        if not st.session_state.get("dom_has_results", False):
            st.info("산출 실행 후 보고서/다운로드가 가능합니다.")
        else:
    
            base_result = st.session_state.get(
                "dom_result_df_adjusted",
                st.session_state.get("dom_result_df_base", pd.DataFrame())
            ).copy()
    
            log_for_report = st.session_state.get(
                "dom_log_df_edited",
                st.session_state.get("dom_log_df_base", pd.DataFrame())
            ).copy()
    
            # 1) 공종 특성(국내에는 해외처럼 feature_master 연동이 없으므로, 동일 섹션은 "현장특성 선택값"으로 대체)
            st.markdown("### 1) 공종 특성")
            _sel_feat = st.session_state.get("dom_sel_feat", [])
            if not _sel_feat:
                st.info("선택된 현장특성이 없습니다.")
            else:
                st.dataframe(pd.DataFrame({"현장특성(선택)": list(_sel_feat)}), use_container_width=True)
    
            # 2) 실적 현장 리스트
            st.markdown("### 2) 실적 현장 리스트")
            
            # (1) 현재 사이드바의 현장특성 필터를 동일하게 적용해서 "가능 현장" 풀을 만든다
            _sel_feat = st.session_state.get("dom_sel_feat", [])
            kr_view = cost_db_kr.copy()
            
            if "현장특성" not in kr_view.columns:
                kr_view["현장특성"] = ""
            
            if _sel_feat:
                kr_view = kr_view[kr_view["현장특성"].astype(str).isin([str(x) for x in _sel_feat])].copy()
            
            # (2) 가능 현장 목록(=사이드바에 표시되는 '가능 현장 n개'의 근거)
            if "현장코드" not in kr_view.columns:
                kr_view["현장코드"] = ""
            if "현장명" not in kr_view.columns:
                kr_view["현장명"] = ""
            
            site_pool = kr_view[["현장코드", "현장명"]].copy()
            site_pool = site_pool.dropna(subset=["현장코드"])
            site_pool["현장코드"] = site_pool["현장코드"].apply(norm_site_code)
            site_pool["현장명"] = site_pool["현장명"].astype(str).fillna("").str.strip()
            site_pool.loc[site_pool["현장명"].isin(["", "nan", "None"]), "현장명"] = "(현장명없음)"
            site_pool = site_pool.drop_duplicates(subset=["현장코드"]).reset_index(drop=True)
            
            all_codes = site_pool["현장코드"].tolist()
            
            # (3) 사용자가 '국내 실적현장'을 선택했는지 확인
            selected_codes = st.session_state.get("dom_selected_site_codes", [])
            selected_codes = [norm_site_code(x) for x in (selected_codes or []) if norm_site_code(x)]
            
            # (4) 선택이 없으면: 가능한 전체(all_codes) 표시
            #     선택이 있으면: 선택한 현장만 표시
            if len(selected_codes) == 0:
                show_codes = all_codes
                st.caption(f"선택된 현장이 없어 가능한 전체 현장 {len(show_codes)}개를 표시합니다.")
            else:
                # 혹시 선택값에 풀에 없는 코드가 섞이면 제거
                show_codes = [c for c in selected_codes if c in set(all_codes)]
                st.caption(f"선택된 현장 {len(show_codes)}개만 표시합니다.")
            
            st_sites = build_site_context_table(site_pool, show_codes)
            
            if st_sites.empty:
                st.info("표시할 현장이 없습니다.")
            else:
                # ✅ 인덱스를 1부터 시작하도록 변경
                st_sites = st_sites.reset_index(drop=True)
                st_sites.index = st_sites.index + 1
            
                st.dataframe(st_sites, use_container_width=True)
    
            # 3) 단가 추출 근거(조건)
            st.markdown("### 3) 단가 추출 근거(조건)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("매칭 유사도, (%)", f"{float(st.session_state.get('dom_sim_threshold', 0.0)):.0f}")
            with c2:
                st.metric("상/하위 컷 비율(%)", f"{float(st.session_state.get('dom_cut_ratio', 0.0)):.0f}")
            with c3:
                st.metric("DB", "국내DB")
    
            # 4) AI 적용 시 최종 기준(해외와 동일 문구)
            st.markdown("### 4) AI 적용 시 최종 기준")
            st.write(get_ai_effective_rule_text())
    
            # 5) 실적 단가 BOQ(결과)
            st.markdown("### 5) 실적 단가 BOQ(결과)")
            if base_result is None or base_result.empty:
                st.warning("결과 데이터가 없습니다. 먼저 산출 실행 후 다시 시도하세요.")
            else:
                st.dataframe(base_result, use_container_width=True)
    
            # 6~7 테이블 자동 생성/갱신 (버튼 제거)
            # - tab3를 열면 항상 최신 log_for_report/base_result 기반으로 갱신
            summary_df, detail_df = build_report_tables_domestic(log_for_report, base_result)
            st.session_state["dom_report_summary_df"] = summary_df
            st.session_state["dom_report_detail_df"] = detail_df
    
            summary_df = st.session_state.get("dom_report_summary_df", pd.DataFrame())
            detail_df = st.session_state.get("dom_report_detail_df", pd.DataFrame())
    
            st.markdown("### 6) 각 내역별 단가 근거(평균)")
            if summary_df is None or summary_df.empty:
                st.info("보고서를 보려면 '보고서 생성/갱신(국내)'을 눌러주세요.")
            else:
                st.dataframe(summary_df, use_container_width=True)
    
            st.markdown("### 7) 각 내역별 단가 근거(선택된 내역)")
            if detail_df is not None and not detail_df.empty:
                st.dataframe(detail_df, use_container_width=True)
            else:
                st.info("Include=True 상세 후보가 없습니다(전부 제외되었거나 후보가 없음).")
    
            # 8) 분포 그래프
            st.markdown("### 8) 내역별 단가 분포")
            render_boq_scatter_domestic(log_for_report, base_result)
    
            # =========================
            # (TAB3) ✅ Excel 다운로드(근거보고서 기준) - 국내
            # =========================
            st.markdown("---")
            st.markdown("### ⬇️ Excel 다운로드(근거보고서 기준)")
            
            # 1) 보고서(표시용으로 이미 rename/정렬된 결과)를 그대로 사용
            rep_sum = st.session_state.get("dom_report_summary_df", pd.DataFrame()).copy()
            rep_det = st.session_state.get("dom_report_detail_df", pd.DataFrame()).copy()
            
            # 2) 아직 생성 전이면(안전장치) 즉시 생성
            if rep_sum.empty or rep_det.empty:
                base_result = st.session_state.get(
                    "dom_result_df_adjusted",
                    st.session_state.get("dom_result_df_base", pd.DataFrame())
                )
                log_for_report = st.session_state.get(
                    "dom_log_df_edited",
                    st.session_state.get("dom_log_df_base", pd.DataFrame())
                )
            
                rep_sum, rep_det = build_report_tables_domestic(log_for_report, base_result)
                st.session_state["dom_report_summary_df"] = rep_sum
                st.session_state["dom_report_detail_df"] = rep_det
            
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                # ✅ 근거보고서 기준으로 저장 (열이름/순서 그대로)
                if not rep_sum.empty:
                    rep_sum.to_excel(writer, index=False, sheet_name="report_summary")
                if not rep_det.empty:
                    rep_det.to_excel(writer, index=False, sheet_name="report_detail")
            
            bio.seek(0)
            
            st.download_button(
                "⬇️ 리포트 다운로드(근거보고서 기준)",
                data=bio.read(),
                file_name="report_domestic.xlsx",
                key="download_report_domestic",
            )

# ============================================================
# ✅ 해외 탭 (기존 코드 전체를 함수로 감싼 버전)
# ============================================================
def render_overseas():
    gs_header("📦 해외 실적단가 DB")

    # =========================
    # Sidebar: 설정
    # =========================
    st.sidebar.markdown("<div class='sb-major'>⚙️ 설정</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr class='sb-hr'/>", unsafe_allow_html=True)

    use_site_filter = True

    DEFAULT_W_STR = 0.3
    DEFAULT_TOP_K_SEM = 200
    w_str = DEFAULT_W_STR
    w_sem = 1.0 - w_str
    top_k_sem = DEFAULT_TOP_K_SEM

    boq_file = None

    # =========================
    # (1) BOQ 업로드 (먼저!)
    # =========================
    with st.container(border=True):
        card_title("📤 BOQ 파일 업로드")
    
        boq_file = st.file_uploader(
            label="",
            type=["xlsx"],
            label_visibility="collapsed",
            key="boq_uploader_overseas",
        )
    # =========================
    # (2) 메인: BOQ 업로드 아래 특성 선택 UI
    # =========================
    auto_sites = []
    
    if boq_file is not None:
        with st.container(border=True):
            card_title("🏷️ 프로젝트 특성 선택", "")
            st.markdown(
                "<div class='dash-muted'>프로젝트 특성을 선택하면 관련 현장이 자동으로 추천됩니다.</div>",
                unsafe_allow_html=True
            )
    
            fm = feature_master.copy()
    
            cols6 = ["대공종", "중공종", "소공종", "Cost Driver Type", "Cost Driver Method", "Cost Driver Condition"]
            need_cols = ["특성ID"] + cols6
    
            for c in need_cols:
                if c not in fm.columns:
                    fm[c] = ""
                fm[c] = fm[c].astype(str).fillna("").str.strip()
    
            if ("특성ID" in project_feature_long.columns) and ("현장코드" in project_feature_long.columns):
                site_cnt = project_feature_long.groupby("특성ID")["현장코드"].nunique().astype(int).to_dict()
            else:
                site_cnt = {}
    
            fm["현장수"] = fm["특성ID"].map(site_cnt).fillna(0).astype(int)
    
            fm["라벨"] = fm.apply(
                lambda r: f'{r["특성ID"]} | {r["대공종"]}/{r["중공종"]}/{r["소공종"]} | '
                          f'{r["Cost Driver Method"]}/{r["Cost Driver Condition"]} | '
                          f'현장 {r["현장수"]}개',
                axis=1
            )
    
            keyword = st.text_input(
                "특성 목록 필터(키워드)",
                value="",
                placeholder="예: DCM, Jet, 지반개량, 도심 ...",
                key="feature_keyword_overseas",
            )
    
            fm_view = fm
            if keyword.strip():
                kw = keyword.strip().lower()
                fm_view = fm[fm["라벨"].str.lower().str.contains(kw, na=False)].copy()
    
            options = fm_view["라벨"].tolist()
            label_to_id = dict(zip(fm_view["라벨"], fm_view["특성ID"]))
    
            # ✅ 필터 바꿔도 기존 선택 유지
            master_label_to_id = dict(zip(fm["라벨"], fm["특성ID"]))
            master_id_to_label = {}
            for lab, fid in master_label_to_id.items():
                master_id_to_label.setdefault(fid, lab)
    
            current_selected_ids = st.session_state.get("selected_feature_ids", [])
            current_labels = [master_id_to_label[fid] for fid in current_selected_ids if fid in master_id_to_label]
    
            new_selected_labels = st.multiselect(
                "특성 선택(다중 선택 가능)",
                options=options,
                default=[lab for lab in current_labels if lab in options],
                key="selected_features_labels_overseas",
            )
    
            new_ids = [label_to_id[lab] for lab in new_selected_labels]
            kept_ids = [
                fid for fid in current_selected_ids
                if (fid in master_id_to_label and master_id_to_label[fid] not in options)
            ]
            merged_ids = sorted(list(dict.fromkeys(kept_ids + new_ids)))
            st.session_state["selected_feature_ids"] = merged_ids
    
            # ✅ auto_sites 계산/저장(기능 유지)
            if merged_ids:
                auto_sites = (
                    project_feature_long[
                        project_feature_long["특성ID"].astype(str).isin([str(x) for x in merged_ids])
                    ]["현장코드"].astype(str).unique().tolist()
                )
            else:
                auto_sites = []
    
            new_auto_sites = sorted({
                norm_site_code(x)
                for x in (auto_sites or [])
                if norm_site_code(x)
            })
            st.session_state["auto_sites"] = new_auto_sites
    
    else:
        st.info("BOQ 업로드 후 프로젝트 특성을 선택할 수 있습니다.")

    # =========================
    # (3) 사이드바: 실적 현장 선택
    # =========================
    selected_site_codes = None

    if use_site_filter:
        _sel_cnt = len(set(
            st.session_state.get("selected_auto_codes", [])
            + st.session_state.get("selected_extra_codes", [])
        ))
        
        st.sidebar.markdown(
            f"""
            <div class="sb-row">
              <div class="sb-title">🏗️ 실적 현장 선택</div>
              <div class="sb-muted">선택 현장: {_sel_cnt}개</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.sidebar.markdown("<hr class='sb-hr'/>", unsafe_allow_html=True)

        auto_sites = st.session_state.get("auto_sites", [])

        # 1) cost_db에서 전체 현장 목록 만들기
        site_df = cost_db[["현장코드", "현장명"]].copy()
        site_df = site_df.dropna(subset=["현장코드"])

        site_df["현장코드"] = site_df["현장코드"].apply(norm_site_code)
        site_df["현장명"] = site_df["현장명"].astype(str).fillna("").str.strip()
        site_df.loc[site_df["현장명"].isin(["", "nan", "None"]), "현장명"] = "(현장명없음)"
        site_df = site_df.drop_duplicates(subset=["현장코드"]).reset_index(drop=True)

        all_codes = site_df["현장코드"].tolist()
        code_to_name = dict(zip(site_df["현장코드"], site_df["현장명"]))

        auto_codes_raw = [norm_site_code(x) for x in (auto_sites or [])]
        auto_codes = [c for c in auto_codes_raw if c in code_to_name]
        other_codes = [c for c in all_codes if c not in set(auto_codes)]

        def fmt_site_code(code: str) -> str:
            name = code_to_name.get(code, "")
            name = name.strip()
            if len(name) > 25:
                return name[:25] + "…"
            return name

        # ✅ auto 후보가 바뀌면 즉시 전체 선택 상태
        auto_sig = "|".join(auto_codes)
        if st.session_state.get("_auto_sig") != auto_sig:
            st.session_state["_auto_sig"] = auto_sig
            st.session_state["selected_auto_codes"] = list(auto_codes)

        if "selected_auto_codes" not in st.session_state:
            st.session_state["selected_auto_codes"] = list(auto_codes)
        if "selected_extra_codes" not in st.session_state:
            st.session_state["selected_extra_codes"] = []

        selected_auto_codes = st.sidebar.multiselect(
            "실적현장",
            options=auto_codes,
            key="selected_auto_codes",
            format_func=fmt_site_code,
        )

        selected_extra_codes = st.sidebar.multiselect(
            "추가 실적현장",
            options=other_codes,
            key="selected_extra_codes",
            format_func=fmt_site_code,
        )

        selected_site_codes = sorted(set(selected_auto_codes + selected_extra_codes))

    # =========================
    # 기타 슬라이더/통화 선택
    # =========================
    st.sidebar.markdown("<div class='sb-title'>🧩 설정값</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr class='sb-hr'/>", unsafe_allow_html=True)

    sim_threshold = st.sidebar.slider("매칭 유사도 기준값(%)", 0, 100, 60, 5)
    cut_ratio = st.sidebar.slider("상/하위 컷 비율 (%)", 0, 30, 20, 5) / 100.0

    target_options = sorted(factor["국가"].astype(str).str.upper().unique().tolist())
    default_idx = target_options.index("KRW") if "KRW" in target_options else 0
    target_currency = st.sidebar.selectbox("산출통화", options=target_options, index=default_idx)

    missing_exchange = exchange[exchange["통화"].astype(str).str.upper() == target_currency].empty
    missing_factor = factor[factor["국가"].astype(str).str.upper() == target_currency].empty

    if missing_exchange:
        st.sidebar.error(f"선택한 산출통화 '{target_currency}'에 대한 환율 정보가 exchange.xlsx에 없습니다.")
    if missing_factor:
        st.sidebar.error(f"선택한 산출통화 '{target_currency}'에 대한 지수 정보가 Factor.xlsx에 없습니다.")

    sidebar_hr(thick=True, mt=10, mb=8)

    # =========================
    # 보정 방식 선택 (여기에 추가)
    # =========================
    st.sidebar.markdown("<div class='sb-title'>💱 보정 방식</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr class='sb-hr'/>", unsafe_allow_html=True)
    
    adjust_method = st.sidebar.radio(
        "보정 방식 선택",
        options=["Location Factor", "PPP Factor", "혼합 방식"],
        index=0,
        key="adjust_method",
    )
    
    # 기본값
    ppp_weight = 0.0
    loc_weight = 1.0
    
    # 혼합 선택 시
    if adjust_method == "혼합 방식":
        loc_weight = st.sidebar.slider(
            "Location 비율 (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            key="loc_weight_slider",
        ) / 100.0
    
        ppp_weight = 1.0 - loc_weight
    
        st.sidebar.caption(f"PPP 비율: {int(ppp_weight * 100)}%")

    # =========================
    # Run / Auto Recompute
    # =========================
    auto_recompute = True  # UI는 숨기지만 기능은 항상 ON

    def boq_file_signature(uploaded_file) -> str:
        if uploaded_file is None:
            return "no_boq"
        try:
            b = uploaded_file.getvalue()
            if len(b) > 2_000_000:
                b = b[:1_000_000] + b[-1_000_000:]
            return hashlib.md5(b).hexdigest()
        except Exception:
            return f"{getattr(uploaded_file, 'name', 'boq')}_{getattr(uploaded_file, 'size', '')}"

    def make_params_signature() -> str:
        payload = {
            "boq": boq_file_signature(boq_file),
            "use_site_filter": bool(use_site_filter),
            "selected_site_codes": sorted([norm_site_code(x) for x in (selected_site_codes or [])]),
            "sim_threshold": float(sim_threshold),
            "cut_ratio": float(cut_ratio),
            "target_currency": str(target_currency),
            "w_str": float(w_str),
            "w_sem": float(w_sem),
            "top_k_sem": int(top_k_sem),
        }
        s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def run_calculation_and_store(run_sig: str):
        status_box = st.empty()

        if boq_file is None:
            status_box.empty()
            st.warning("BOQ 파일을 업로드해 주세요.")
            return
        if missing_exchange or missing_factor:
            status_box.empty()
            st.error("산출통화에 필요한 환율/지수 정보가 없습니다.")
            return

        progress = st.progress(0.0)
        prog_text = st.empty()

        status_box.markdown("### ⏳ 산출중... (BOQ 로드/필터링)")
        boq = pd.read_excel(boq_file, engine="openpyxl")

        if use_site_filter and selected_site_codes is not None:
            cost_db_run = cost_db[
                cost_db["현장코드"].apply(norm_site_code).isin([norm_site_code(x) for x in selected_site_codes])
            ].copy()
        else:
            cost_db_run = cost_db.copy()

        st.sidebar.caption(f"전체 {len(cost_db):,}개 내역 중 {len(cost_db_run):,}개 내역으로 산출 실행")

        pool_sig_payload = {
            "boq": boq_file_signature(boq_file),
            "use_site_filter": bool(use_site_filter),
            "selected_site_codes": sorted([norm_site_code(x) for x in (selected_site_codes or [])]),
            "top_k_sem": int(top_k_sem),
            "w_str": float(w_str),
            "w_sem": float(w_sem),
            "cost_db_rows": int(len(cost_db_run)),
        }
        pool_sig = hashlib.md5(json.dumps(pool_sig_payload, sort_keys=True).encode("utf-8")).hexdigest()

        need_new_pool = (st.session_state.get("candidate_pool_sig") != pool_sig) or ("candidate_pool" not in st.session_state)

        if need_new_pool:
            status_box.markdown("### ⏳ 산출중... (후보 풀 생성)")
            with st.spinner("후보 풀 생성(최초/현장변경 시 오래 걸릴 수 있음)..."):
                pool = build_candidate_pool(
                    cost_db=cost_db_run,
                    boq=boq,
                    price_index=price_index,
                    sim_w_str=w_str,
                    sim_w_sem=w_sem,
                    top_k_sem=top_k_sem,
                    pool_per_boq=400,
                    progress=progress,
                    prog_text=prog_text,
                )
            st.session_state["candidate_pool"] = pool
            st.session_state["candidate_pool_sig"] = pool_sig
        else:
            pool = st.session_state["candidate_pool"]

        status_box.markdown("### ⏳ 산출중...")
        with st.spinner("빠른 재계산 중)..."):
            result_df, log_df = fast_recompute_from_pool(
                pool=pool,
                exchange=exchange,
                factor=factor,
                sim_threshold=sim_threshold,
                cut_ratio=cut_ratio,
                target_currency=target_currency,
            )
        # ✅ 산출 완료 후 진행 문구/프로그레스 제거 (남는 문구 방지)
        try:
            status_box.empty()
        except Exception:
            pass
        try:
            progress.empty()
        except Exception:
            pass
        try:
            prog_text.empty()
        except Exception:
            pass

        st.session_state["boq_df"] = boq
        st.session_state["result_df_base"] = result_df.copy()
        st.session_state["log_df_base"] = log_df.copy()
        st.session_state["log_df_edited"] = log_df.copy()
        st.session_state.pop("result_df_adjusted", None)
        st.session_state["has_results"] = True
        st.session_state["last_run_sig"] = run_sig
        st.session_state["adjust_method"] = adjust_method
        st.session_state["loc_weight"] = loc_weight
        st.session_state["ppp_weight"] = ppp_weight

    run_btn = st.sidebar.button("🚀 산출 실행")
    current_sig = make_params_signature()
    last_sig = st.session_state.get("last_run_sig", None)
    needs_rerun = (last_sig is not None and current_sig != last_sig)

    if st.session_state.get("has_results", False) and needs_rerun and not auto_recompute:
        st.sidebar.warning("⚠️ 조건이 변경되었습니다. 다시 산출 실행이 필요합니다.")

    auto_run = st.session_state.get("has_results", False) and needs_rerun and auto_recompute

    if run_btn or auto_run:
        if auto_run:
            st.sidebar.info("ℹ️ 조건 변경 감지 → 자동 재산출 중 (로그 편집값은 초기화됩니다)")
        run_calculation_and_store(current_sig)

    if st.session_state.get("has_results", False):
        boq = st.session_state["boq_df"]
        result_df = st.session_state["result_df_base"]
        log_df = st.session_state["log_df_base"]

        def recompute_result_from_log(edited_log: pd.DataFrame) -> pd.DataFrame:
            base = st.session_state["result_df_base"].copy()

            out_prices = []
            for boq_id, g in edited_log.groupby("BOQ_ID"):
                g2 = g[g["Include"] == True].copy()
                if g2.empty:
                    out_prices.append((int(boq_id), None, target_currency, "매칭 후보 없음(또는 전부 제외)", ""))
                    continue

                final_price = float(pd.to_numeric(g2["__adj_price"], errors="coerce").mean())

                currencies = sorted(g2["통화"].astype(str).str.upper().unique().tolist())
                reason_text = f"{len(currencies)}개국({', '.join(currencies)}) {len(g2)}개 내역 근거"

                vc = g2["공종코드"].astype(str).value_counts()
                top_code = vc.index[0] if len(vc) else ""
                top_cnt = int(vc.iloc[0]) if len(vc) else 0
                top_work = f"{top_code} ({top_cnt}/{len(g2)})" if top_code else ""

                out_prices.append((int(boq_id), f"{final_price:,.2f}", target_currency, reason_text, top_work))

            upd = pd.DataFrame(out_prices, columns=["BOQ_ID", "Final Price", "산출통화", "산출근거", "근거공종(최빈)"])

            base = base.drop(
                columns=[c for c in ["Final Price", "산출통화", "산출근거", "근거공종(최빈)"] if c in base.columns],
                errors="ignore"
            )
            base = base.merge(upd, on="BOQ_ID", how="left")
            return base

        tab1, tab2, tab3 = st.tabs(["📄 BOQ 결과", "🧾 산출 근거(편집 가능)", "📝 근거 보고서"])

        with tab2:
            st.caption("✅ 체크 해제하면 평균단가 산출에서 제외됩니다. 체크하면 포함됩니다.")

            if "log_df_edited" not in st.session_state:
                st.session_state["log_df_edited"] = log_df.copy()

            log_all = st.session_state["log_df_edited"]
          
            boq_ids = sorted(log_all["BOQ_ID"].dropna().astype(int).unique().tolist())

            base_for_label = st.session_state.get("result_df_base", pd.DataFrame()).copy()
            boq_text_col = "내역" if ("내역" in base_for_label.columns) else None

            if boq_text_col and ("BOQ_ID" in base_for_label.columns):
                id_to_text = (
                    base_for_label.dropna(subset=["BOQ_ID"])
                    .assign(BOQ_ID=lambda d: d["BOQ_ID"].astype(int))
                    .set_index("BOQ_ID")[boq_text_col]
                    .astype(str)
                    .to_dict()
                )
            else:
                id_to_text = (
                    log_all.dropna(subset=["BOQ_ID"])
                    .assign(BOQ_ID=lambda d: d["BOQ_ID"].astype(int))
                    .groupby("BOQ_ID")["BOQ_내역"].first()
                    .astype(str)
                    .to_dict()
                )

            def fmt_boq_id(x: int) -> str:
                t = id_to_text.get(int(x), "")
                t = (t[:60] + "…") if len(t) > 60 else t
                return f"{int(x)} | {t}"

            sel_id = st.selectbox(
                "편집할 BOQ 선택",
                options=boq_ids,
                format_func=fmt_boq_id,
                key="sel_boq_id"
            )

            log_view_full = log_all[log_all["BOQ_ID"].astype(int) == int(sel_id)].copy()

            if "_include_backup" not in st.session_state:
                st.session_state["_include_backup"] = {}
            if "_include_backup_all" not in st.session_state:
                st.session_state["_include_backup_all"] = None

            cA, cB, cC, cD = st.columns([1.2, 1.0, 1.0, 1.8])
            with cA:
                agent_mode = st.selectbox("AI 추천 모드", ["보수적", "균형", "공격적"], index=1, key="agent_mode")
            with cB:
                min_keep = st.number_input("최소 포함", min_value=1, max_value=20, value=3, step=1, key="agent_min_keep")
            with cC:
                max_keep = st.number_input("최대 포함", min_value=3, max_value=200, value=50, step=1, key="agent_max_keep")
            with cD:
                st.caption("※ 적용 후 화면이 자동 갱신됩니다.")

            b1, b2, b3, b4 = st.columns([1.2, 1.2, 1.2, 2.4])
            with b1:
                btn_ai_one = st.button("🤖 AI 적용(현재 BOQ)", key="btn_ai_one")
            with b2:
                btn_undo_one = st.button("↩️ 되돌리기(현재 BOQ)", key="btn_undo_one")
            with b3:
                btn_ai_all = st.button("🤖 AI 적용(전체 BOQ)", key="btn_ai_all")
            with b4:
                btn_undo_all = st.button("↩️ 되돌리기(전체 BOQ)", key="btn_undo_all")

            if btn_undo_one:
                backup = st.session_state["_include_backup"].get(int(sel_id))
                if backup is not None and len(backup) == len(log_view_full.index):
                    st.session_state["log_df_edited"].loc[log_view_full.index, "Include"] = backup.values
                    st.session_state["result_df_adjusted"] = recompute_result_from_log(st.session_state["log_df_edited"])
                    st.success("되돌리기 완료(현재 BOQ)")
                    st.rerun()
                else:
                    st.warning("되돌릴 백업이 없습니다(또는 후보행이 변경됨).")

            if btn_ai_one:
                st.session_state["_include_backup"][int(sel_id)] = st.session_state["log_df_edited"].loc[log_view_full.index, "Include"].copy()
                updated, summary = apply_agent_to_log(
                    log_all=st.session_state["log_df_edited"].copy(),
                    boq_id=int(sel_id),
                    mode=agent_mode,
                    min_keep=int(min_keep),
                    max_keep=int(max_keep),
                )
                st.session_state["log_df_edited"] = updated
                st.session_state["result_df_adjusted"] = recompute_result_from_log(st.session_state["log_df_edited"])
                if summary:
                    st.success(f"AI 적용 완료(현재 BOQ): {summary['kept']}/{summary['total']} 포함, 모드={summary['mode']}")
                record_ai_last_applied("현재 BOQ", agent_mode, int(min_keep), int(max_keep), summary, boq_id=int(sel_id))
                st.rerun()

            if btn_ai_all:
                st.session_state["_include_backup_all"] = st.session_state["log_df_edited"][["BOQ_ID", "Include"]].copy()
                updated, sum_df = apply_agent_to_all_boqs(
                    log_all=st.session_state["log_df_edited"].copy(),
                    mode=agent_mode,
                    min_keep=int(min_keep),
                    max_keep=int(max_keep),
                )
                st.session_state["log_df_edited"] = updated
                st.session_state["result_df_adjusted"] = recompute_result_from_log(st.session_state["log_df_edited"])
                st.success("AI 적용 완료(전체 BOQ)")
                if sum_df is not None and not sum_df.empty:
                    st.dataframe(sum_df, use_container_width=True)
                record_ai_last_applied("전체 BOQ", agent_mode, int(min_keep), int(max_keep), None)
                st.rerun()

            if btn_undo_all:
                backup_all = st.session_state.get("_include_backup_all")
                if backup_all is None or backup_all.empty:
                    st.warning("되돌릴 전체 백업이 없습니다.")
                else:
                    cur = st.session_state["log_df_edited"].copy()
                    b = backup_all.copy()
                    b["BOQ_ID"] = b["BOQ_ID"].astype(int)
                    cur["BOQ_ID"] = cur["BOQ_ID"].astype(int)

                    cur = cur.drop(columns=["Include"], errors="ignore").merge(b, on="BOQ_ID", how="left")
                    cur["Include"] = cur["Include"].fillna(False).astype(bool)

                    st.session_state["log_df_edited"] = cur
                    st.session_state["result_df_adjusted"] = recompute_result_from_log(st.session_state["log_df_edited"])
                    st.success("되돌리기 완료(전체 BOQ)")
                    st.rerun()

            display_cols = [
                "Include", "DefaultInclude",
                "내역", "Unit",
                "Unit Price", "통화", "계약년월",
                "__adj_price", "산출통화",
                "__cpi_ratio", "__latest_ym",
                "__fx_ratio",
                "__fac_ratio",
                "__hyb",
                "공종코드", "공종명",
                "현장코드", "현장명",
                "협력사코드", "협력사명",
            ]

            for c in display_cols:
                if c not in log_view_full.columns:
                    log_view_full[c] = None

            log_view = log_view_full[display_cols].copy()

            edited_view = st.data_editor(
                log_view,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Include": st.column_config.CheckboxColumn("포함", help="평균단가 산출 포함/제외"),
                    "DefaultInclude": st.column_config.CheckboxColumn("기본포함", help="초기 자동 포함 여부(컷 로직)"),
                    "내역": st.column_config.TextColumn("내역", width="large"),
                    "Unit": st.column_config.TextColumn("단위(Unit)"),
                    "Unit Price": st.column_config.NumberColumn("원단가", format="%.4f"),
                    "통화": st.column_config.TextColumn("원통화"),
                    "계약년월": st.column_config.TextColumn("계약년월"),
                    "__adj_price": st.column_config.NumberColumn("산출단가(산출통화 기준)", format="%.4f"),
                    "산출통화": st.column_config.TextColumn("산출통화"),
                    "__cpi_ratio": st.column_config.NumberColumn("물가보정계수(CPI)", format="%.6f"),
                    "__latest_ym": st.column_config.TextColumn("물가지수 최신월"),
                    "__fx_ratio": st.column_config.NumberColumn("환율보정계수", format="%.6f"),
                    "__fac_ratio": st.column_config.NumberColumn("국가보정계수(Factor)", format="%.6f"),
                    "__hyb": st.column_config.NumberColumn("유사도점수", format="%.2f"),
                    "공종코드": st.column_config.TextColumn("공종코드"),
                    "공종명": st.column_config.TextColumn("공종명"),
                    "현장코드": st.column_config.TextColumn("현장코드"),
                    "현장명": st.column_config.TextColumn("현장명"),
                    "협력사코드": st.column_config.TextColumn("협력사코드"),
                    "협력사명": st.column_config.TextColumn("협력사명"),
                },
                disabled=[c for c in log_view.columns if c not in ["Include"]],
                key="log_editor",
            )

            st.session_state["log_df_edited"].loc[log_view_full.index, "Include"] = edited_view["Include"].values
            st.session_state["result_df_adjusted"] = recompute_result_from_log(st.session_state["log_df_edited"])

        with tab1:
            show_df = st.session_state.get("result_df_adjusted", result_df).copy()

            if "통화" in show_df.columns:
                show_df = show_df.drop(columns=["통화"])

            if "Final Price" in show_df.columns:
                if "산출통화" not in show_df.columns:
                    show_df["산출통화"] = target_currency

                cols = show_df.columns.tolist()
                cols.remove("산출통화")
                fp_idx = cols.index("Final Price")
                cols.insert(fp_idx + 1, "산출통화")
                show_df = show_df[cols]

            st.dataframe(show_df, use_container_width=True)

        with tab3:
          
            base_result = st.session_state.get("result_df_adjusted", st.session_state.get("result_df_base", pd.DataFrame()))
            log_for_report = st.session_state.get("log_df_edited", st.session_state.get("log_df_base", pd.DataFrame()))

            st.markdown("### 1) 공종 특성")
            sel_features = st.session_state.get("selected_feature_ids", [])
            ft = build_feature_context_table(feature_master, sel_features)
            if ft.empty:
                st.info("선택된 특성ID가 없습니다.")
            else:
                st.dataframe(ft, use_container_width=True)

            st.markdown("### 2) 실적 현장 리스트")
            try:
                _sel_sites = selected_site_codes if (selected_site_codes is not None) else []
            except Exception:
                _sel_sites = []
            st_sites = build_site_context_table(cost_db, _sel_sites)
            if st_sites.empty:
                st.info("표시할 현장이 없습니다.")
            else:
                # ✅ 인덱스를 1부터 시작하도록 변경
                st_sites = st_sites.reset_index(drop=True)
                st_sites.index = st_sites.index + 1
            
                st.dataframe(st_sites, use_container_width=True)

            st.markdown("### 3) 단가 추출 근거(조건)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("매칭 유사도, (%)", f"{float(sim_threshold):.0f}")
            with c2:
                st.metric("상/하위 컷 비율(%)", f"{float(cut_ratio) * 100 :.0f}")
            with c3:
                st.metric("산출통화", str(target_currency))

            st.markdown("### 4) AI 적용 시 최종 기준")
            st.write(get_ai_effective_rule_text())

            st.markdown("### 5) 실적 단가 BOQ(결과)")
            if base_result is None or base_result.empty:
                st.warning("결과 데이터가 없습니다. 먼저 산출 실행 후 다시 시도하세요.")
            else:
                st.dataframe(base_result, use_container_width=True)

            # ✅ (TAB3) 보고서 자동 생성/갱신 (변경 감지)
            def _report_sig(log_df: pd.DataFrame, result_df: pd.DataFrame) -> str:
                # Include/BOQ_ID/__adj_price만 바뀌어도 요약/상세가 바뀌므로 이 정도만 서명으로 사용
                cols = [c for c in ["BOQ_ID", "Include", "__adj_price", "__hyb"] if c in log_df.columns]
                base = log_df[cols].copy() if cols else log_df.copy()
                s1 = file_fingerprint(base, cols) if cols else hashlib.md5(str(base.shape).encode()).hexdigest()
            
                s2 = ""
                if result_df is not None and not result_df.empty:
                    cols2 = [c for c in ["BOQ_ID", "Final Price"] if c in result_df.columns]
                    if cols2:
                        s2 = file_fingerprint(result_df[cols2].copy(), cols2)
                    else:
                        s2 = hashlib.md5(str(result_df.shape).encode()).hexdigest()
            
                return hashlib.md5((s1 + "|" + s2).encode("utf-8")).hexdigest()
            
            cur_rep_sig = _report_sig(log_for_report, base_result)
            prev_rep_sig = st.session_state.get("report_sig", None)
            
            if (prev_rep_sig != cur_rep_sig) or ("report_summary_df" not in st.session_state) or ("report_detail_df" not in st.session_state):
                summary_df, detail_df = build_report_tables(log_for_report, base_result)
                st.session_state["report_summary_df"] = summary_df
                st.session_state["report_detail_df"] = detail_df
                st.session_state["report_sig"] = cur_rep_sig
            
            summary_df = st.session_state.get("report_summary_df", pd.DataFrame())
            detail_df = st.session_state.get("report_detail_df", pd.DataFrame())

            st.markdown("### 6) 각 내역별 단가 근거(평균)")
            if summary_df is None or summary_df.empty:
                st.info("보고서를 보려면 '보고서 생성/갱신'을 눌러주세요.")
            else:
                st.dataframe(summary_df, use_container_width=True)

            st.markdown("### 7) 각 내역별 단가 근거(선택된 내역)")
            if detail_df is not None and not detail_df.empty:
                st.dataframe(detail_df, use_container_width=True)
            else:
                st.info("Include=True 상세 후보가 없습니다(전부 제외되었거나 후보가 없음).")

            st.markdown("### 8) 내역별 단가 분포")
            render_boq_scatter(log_for_report, base_result)

            out_result = st.session_state.get("result_df_adjusted", result_df).copy()
            out_log = st.session_state.get("log_df_edited", log_df).copy()

            # =========================
            # (TAB3) ✅ Excel 다운로드(근거보고서)
            # =========================
            st.markdown("---")
            st.markdown("### ⬇️ Excel 다운로드(근거보고서 기준)")
            
            # 1) 보고서(표시용으로 이미 rename/정렬된 결과)를 그대로 사용
            rep_sum = st.session_state.get("report_summary_df", pd.DataFrame()).copy()
            rep_det = st.session_state.get("report_detail_df", pd.DataFrame()).copy()
            
            # 2) 혹시 아직 생성 전이면(안전장치) 즉시 생성
            if rep_sum.empty or rep_det.empty:
                base_result = st.session_state.get("result_df_adjusted", st.session_state.get("result_df_base", pd.DataFrame()))
                log_for_report = st.session_state.get("log_df_edited", st.session_state.get("log_df_base", pd.DataFrame()))
                rep_sum, rep_det = build_report_tables(log_for_report, base_result)
                st.session_state["report_summary_df"] = rep_sum
                st.session_state["report_detail_df"] = rep_det
            
            # 3) 결과/로그도 “표시용(열 이름 바뀐 것)”을 쓰고 싶으면 여기서 rename해서 맞추면 되지만,
            #    질문 요지(근거보고서 기반)라서 report_summary/report_detail만 기준으로 저장합니다.
            
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                # ✅ 근거보고서 기준으로 저장 (열이름/순서 그대로)
                if not rep_sum.empty:
                    rep_sum.to_excel(writer, index=False, sheet_name="report_summary")
                if not rep_det.empty:
                    rep_det.to_excel(writer, index=False, sheet_name="report_detail")
            
            bio.seek(0)
            
            st.download_button(
                "⬇️ 리포트 다운로드(근거보고서 기준)",
                data=bio.read(),
                file_name="report_overseas.xlsx",
                key="download_report_overseas",
            )

# ============================================================
# ✅ 상단 탭(해외/국내) + 사이드바 중복 렌더 방지 로직
# - Streamlit은 탭이 있어도 코드가 둘 다 실행되는 경우가 많아서,
#   active_db 상태로 "한쪽만" 실제 렌더하도록 구성
# ============================================================
tab_over, tab_dom = st.tabs(["🌍 해외 실적단가 DB", "🇰🇷 실적단가 DB"])

with tab_over:
    if st.session_state["active_db"] != "overseas":
        if st.button("이 탭으로 전환", key="switch_to_overseas"):
            st.session_state["active_db"] = "overseas"
            st.rerun()
        st.info("현재 활성 화면은 국내 탭입니다. 전환 버튼을 눌러 활성화하세요.")
    else:
        render_overseas()

with tab_dom:
    if st.session_state["active_db"] != "domestic":
        if st.button("이 탭으로 전환", key="switch_to_domestic"):
            st.session_state["active_db"] = "domestic"
            st.rerun()
        st.info("현재 활성 화면은 해외 탭입니다. 전환 버튼을 눌러 활성화하세요.")
    else:
        render_domestic()


















































