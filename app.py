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

st.markdown(f"""
<style>
  .main {{ background-color: {BG_LIGHT}; }}

  /* ====== 사이드바에서만 강제 적용(우선순위 ↑) ====== */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
  background-color: #ffffff !important;
  border: 1px solid #005EB8 !important;
  border-radius: 6px !important;
}

/* ✅ 산출통화(Selectbox) 선택 텍스트를 검정으로 강제 */
section[data-testid="stSidebar"] div[data-baseweb="select"] input{
  color:#000000 !important;
  -webkit-text-fill-color:#000000 !important;
}

/* ====== 멀티셀렉트 칩(tag) : BaseWeb DOM이 케이스별로 div/span 둘 다 나올 수 있어 둘 다 잡음 ====== */
section[data-testid="stSidebar"] div[data-baseweb="tag"],
section[data-testid="stSidebar"] span[data-baseweb="tag"]{
  background-color:#4DA3FF !important;   /* 밝은 파란색 */
  border:1px solid #2F80ED !important;
  border-radius:8px !important;

  min-height:30px !important;
  height:30px !important;
  display:inline-flex !important;
  align-items:center !important;

  padding:0 10px !important;
  box-sizing:border-box !important;

  /* ✅ 폭을 넓혀서 15자 잘림 완화 */
  max-width: 280px !important;
}

/* ✅ tag 내부 텍스트: ellipsis 유지 + 폭 확대 */
section[data-testid="stSidebar"] div[data-baseweb="tag"] span,
section[data-testid="stSidebar"] span[data-baseweb="tag"] span{
  color:#ffffff !important;
  white-space:nowrap !important;
  overflow:hidden !important;
  text-overflow:ellipsis !important;
  max-width: 230px !important;  /* 25자 정도 노출 목표 */
  display:inline-block !important;
}

/* ✅ tag의 X 아이콘/버튼 색 */
section[data-testid="stSidebar"] div[data-baseweb="tag"] svg,
section[data-testid="stSidebar"] span[data-baseweb="tag"] svg,
section[data-testid="stSidebar"] div[data-baseweb="tag"] path,
section[data-testid="stSidebar"] span[data-baseweb="tag"] path{
  fill:#ffffff !important;
}

/* hover */
section[data-testid="stSidebar"] div[data-baseweb="tag"]:hover,
section[data-testid="stSidebar"] span[data-baseweb="tag"]:hover{
  background-color:#2F80ED !important;
  border:1px solid #1C6DD0 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='gs-header'>📦 해외 실적단가 DB</div>", unsafe_allow_html=True)
st.write("")


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
    hasher.update(head.encode()); hasher.update(tail.encode())
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
        now  = df.loc[df["년월_std"] == latest_ym, "Index"].values
        if len(base) and len(now) and base[0] not in (0, None):
            return float(now[0]) / float(base[0]), float(base[0]), float(now[0]), latest_ym
    except Exception:
        pass
    return 1.0, None, None, None

def get_exchange_rate(exchange: pd.DataFrame, from_currency: str, to_currency: str) -> float:
    try:
        usd_from = exchange.loc[exchange["통화"].astype(str).str.upper()==str(from_currency).upper(), "USD당환율"].values
        usd_to   = exchange.loc[exchange["통화"].astype(str).str.upper()==str(to_currency).upper(), "USD당환율"].values
        if len(usd_from) and len(usd_to) and float(usd_from[0]) != 0:
            return float(usd_to[0]) / float(usd_from[0])
    except Exception:
        pass
    return 1.0

def get_factor_ratio(factor: pd.DataFrame, from_currency: str, to_currency: str) -> float:
    try:
        f_from = factor.loc[factor["국가"].astype(str).str.upper()==str(from_currency).upper(), "지수"].values
        f_to   = factor.loc[factor["국가"].astype(str).str.upper()==str(to_currency).upper(), "지수"].values
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
            cand_idx = I[0]; sem_scores = D[0]
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
        hyb = hybrid_scores(boq_text_norm, unit_df["__내역_norm"], unit_df["__sem"].to_numpy(), sim_w_str, sim_w_sem)
        unit_df["__hyb"] = hyb

        # 너무 큰 풀 방지: hyb 상위 N개만 보관
        unit_df = unit_df.sort_values("__hyb", ascending=False).head(pool_per_boq).copy()

        # CPI는 통화+계약월 기준으로 미리 계산 (산출통화 바뀌어도 재사용 가능)
        # contract_ym을 문자열로
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

    # 풀에서 앞으로 필요한 최소 컬럼만 유지(가벼운 재계산용)
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
    df = df[pd.to_numeric(df["__hyb"], errors="coerce").fillna(0) >= float(sim_threshold)].copy()
    if df.empty:
        # BOQ 결과도 BOQ_ID 기반으로 만들기 어렵기 때문에, 빈 결과 반환
        return pd.DataFrame(), pd.DataFrame()

    # 2) FX/Factor 맵(통화별) 만들어 vectorized 계산
    currencies = df["통화"].astype(str).str.upper().unique().tolist()

    fx_map = {}
    fac_map = {}
    for c in currencies:
        fx_map[c] = get_exchange_rate(exchange, c, target_currency)
        fac_map[c] = get_factor_ratio(factor, c, target_currency)

    df["통화_std"] = df["통화"].astype(str).str.upper()
    df["__fx_ratio"] = df["통화_std"].map(fx_map).fillna(1.0)
    df["__fac_ratio"] = df["통화_std"].map(fac_map).fillna(1.0)
    df["산출통화"] = target_currency

    # 3) __adj_price 계산
    unit_price = pd.to_numeric(df["Unit Price"], errors="coerce").fillna(0.0)
    cpi_ratio = pd.to_numeric(df["__cpi_ratio"], errors="coerce").fillna(1.0)

    df["__adj_price"] = unit_price * cpi_ratio * df["__fx_ratio"] * df["__fac_ratio"]

    # 4) BOQ별 컷 + Include/DefaultInclude 설정
    df["Include"] = False
    df["DefaultInclude"] = False

    for boq_id, gidx in df.groupby("BOQ_ID").groups.items():
        sub = df.loc[gidx].sort_values("__adj_price").copy()
        n = len(sub)
        cut = max(0, int(n * cut_ratio)) if n > 5 else 0

        if cut > 0:
            keep_mask = np.zeros(n, dtype=bool)
            keep_mask[cut:n-cut] = True
        else:
            keep_mask = np.ones(n, dtype=bool)

        kept_index = sub.index[keep_mask]
        df.loc[kept_index, "DefaultInclude"] = True
        df.loc[kept_index, "Include"] = True

    # 5) BOQ 결과(result_df) 생성
    results = []
    for boq_id, sub in df.groupby("BOQ_ID"):
        inc = sub[sub["Include"] == True]
        if inc.empty:
            final_price = None
            reason_text = "매칭 후보 없음(또는 전부 제외)"
            top_work = ""
        else:
            final_price = float(inc["__adj_price"].mean())
            currencies2 = sorted(inc["통화_std"].unique().tolist())
            reason_text = f"{len(currencies2)}개국({', '.join(currencies2)}) {len(inc)}개 내역 근거"

            vc = inc["공종코드"].astype(str).value_counts()
            top_code = vc.index[0] if len(vc) else ""
            top_cnt = int(vc.iloc[0]) if len(vc) else 0
            top_work = f"{top_code} ({top_cnt}/{len(inc)})" if top_code else ""

        # BOQ 메타는 pool에 있음
        one = sub.iloc[0]
        results.append({
            "BOQ_ID": int(boq_id),
            "내역": one.get("BOQ_내역", ""),
            "Unit": one.get("BOQ_Unit", ""),
            "Final Price": f"{final_price:,.2f}" if final_price is not None else None,
            "산출근거": reason_text,
            "근거공종(최빈)": top_work,
        })

    result_df = pd.DataFrame(results).sort_values("BOQ_ID").reset_index(drop=True)

    # 6) 산출 로그(log_df) 반환(Include 편집 가능하도록 필요한 컬럼 포함)
    log_cols = [
        "BOQ_ID","BOQ_내역","BOQ_Unit",
        "Include","DefaultInclude",
        "공종코드","공종명",
        "내역","Unit",
        "Unit Price","통화","계약년월",
        "__adj_price","산출통화",
        "__cpi_ratio","__latest_ym",
        "__fx_ratio","__fac_ratio",
        "__hyb",
        "현장코드","현장명",
        "협력사코드","협력사명",
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
def build_report_tables(log_df: pd.DataFrame, result_df: pd.DataFrame):
    if log_df is None or log_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = log_df.copy()
    df["BOQ_ID"] = df["BOQ_ID"].astype(int)

    inc = df[df["Include"] == True].copy()

    detail_cols = [
        "BOQ_ID","BOQ_내역","BOQ_Unit",
        "내역","Unit","Unit Price","통화","계약년월",
        "__adj_price","산출통화",
        "__cpi_ratio","__latest_ym","__fx_ratio","__fac_ratio","__hyb",
        "공종코드","공종명",
        "현장코드","현장명","협력사코드","협력사명",
        "AI_모드","AI_추천사유",
    ]
    for c in detail_cols:
        if c not in inc.columns:
            inc[c] = None
    detail_df = inc[detail_cols].copy()

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
            "BOQ_내역": one.get("BOQ_내역",""),
            "BOQ_Unit": one.get("BOQ_Unit",""),
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

    if result_df is not None and not result_df.empty and "BOQ_ID" in result_df.columns:
        tmp = result_df.copy()
        tmp["BOQ_ID"] = tmp["BOQ_ID"].astype(int)
        keep = [c for c in ["BOQ_ID","Final Price","산출근거","근거공종(최빈)"] if c in tmp.columns]
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
    """
    scope: "현재 BOQ" or "전체 BOQ"
    summary: suggest_include_for_one_boq()에서 반환한 summary(있으면 hyb_min, iqr_k 포함)
    boq_id: scope가 "현재 BOQ"일 때 어떤 BOQ에 적용했는지 기록용
    """
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
import matplotlib.pyplot as plt

def build_feature_context_table(feature_master: pd.DataFrame, selected_feature_ids: list) -> pd.DataFrame:
    if not selected_feature_ids:
        return pd.DataFrame(columns=["특성ID","대공종","중공종","소공종","Cost Driver Type","Cost Driver Method","Cost Driver Condition"])
    fm = feature_master.copy()
    cols6 = ["대공종","중공종","소공종","Cost Driver Type","Cost Driver Method","Cost Driver Condition"]
    keep = ["특성ID"] + cols6
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
        return pd.DataFrame(columns=["현장코드","현장명"])
    tmp = cost_db[["현장코드","현장명"]].copy()
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
    # "2019-11" / 날짜 / 기타가 섞여 있어도 최대한 datetime으로
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().any():
        # fallback: YYYY-MM 형태로 정규화 후 재파싱
        s2 = s.astype(str).apply(to_year_month_string)
        dt2 = pd.to_datetime(s2, errors="coerce")
        dt = dt.fillna(dt2)
    return dt

def render_boq_scatter(log_df: pd.DataFrame, base_result: pd.DataFrame):
    if log_df is None or log_df.empty:
        st.info("로그 데이터가 없어 그래프를 표시할 수 없습니다.")
        return

    # 검색(예: REBAR)
    keyword = st.text_input("내역 키워드(예: REBAR)", value="", key="report_kw")
    cand = base_result.copy() if (base_result is not None and not base_result.empty) else None

    if cand is not None and "내역" in cand.columns and "BOQ_ID" in cand.columns and keyword.strip():
        kw = keyword.strip().lower()
        cand = cand[cand["내역"].astype(str).str.lower().str.contains(kw, na=False)].copy()

    # 선택 후보 BOQ_ID 목록
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

    # x, y 준비
    sub["계약월_dt"] = _parse_contract_month_series(sub["계약년월"])
    sub["산출단가"] = pd.to_numeric(sub["__adj_price"], errors="coerce")
    sub["포함여부"] = sub["Include"].fillna(False).astype(bool)

    # 표시용 최소 컬럼만
    sub["표시내역"] = sub["내역"].astype(str)

    # Altair 산점도
    # - 색: 포함여부(자동 스킴)
    # - 크기: 포함=True면 크게
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

cost_db     = load_excel_from_repo("cost_db.xlsx")
price_index = load_excel_from_repo("price_index.xlsx")
exchange    = load_excel_from_repo("exchange.xlsx")
factor      = load_excel_from_repo("Factor.xlsx")
project_feature_long = load_excel_from_repo("project_feature_long.xlsx")
feature_master = load_excel_from_repo("feature_master_FID.xlsx")


# =========================
# Session init
# =========================
if "selected_feature_ids" not in st.session_state:
    st.session_state["selected_feature_ids"] = []
if "auto_sites" not in st.session_state:
    st.session_state["auto_sites"] = []


# =========================
# Sidebar: 설정
# =========================
st.sidebar.header("⚙️ 설정")

# ✅ 현장필터는 기능적으로 계속 사용(항상 True)하되, 화면에는 노출하지 않음
use_site_filter = True

DEFAULT_W_STR = 0.3
DEFAULT_TOP_K_SEM = 200
w_str = DEFAULT_W_STR
w_sem = 1.0 - w_str
top_k_sem = DEFAULT_TOP_K_SEM


# =========================
# (1) BOQ 업로드 (먼저!)
# =========================
with st.container():
    st.markdown("<div class='gs-card'>", unsafe_allow_html=True)
    boq_file = st.file_uploader("📤 BOQ 파일 업로드", type=["xlsx"])
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# (2) 메인: BOQ 업로드 아래 특성 선택 UI
# =========================
auto_sites = []

if boq_file is not None:
    st.markdown("<div class='gs-card'>", unsafe_allow_html=True)
    st.markdown("### 🏷️ 프로젝트 특성 선택 (176개 전체)")

    fm = feature_master.copy()
    cols6 = ["대공종","중공종","소공종","Cost Driver Type","Cost Driver Method","Cost Driver Condition"]
    for c in ["특성ID"] + cols6:
        fm[c] = fm[c].astype(str).fillna("").str.strip()

    site_cnt = project_feature_long.groupby("특성ID")["현장코드"].nunique().astype(int).to_dict()
    fm["현장수"] = fm["특성ID"].map(site_cnt).fillna(0).astype(int)

    fm["라벨"] = fm.apply(
        lambda r: f'{r["특성ID"]} | {r["대공종"]}/{r["중공종"]}/{r["소공종"]} | '
                  f'{r["Cost Driver Type"]}/{r["Cost Driver Method"]}/{r["Cost Driver Condition"]} | '
                  f'현장 {r["현장수"]}개',
        axis=1
    )

    keyword = st.text_input("특성 목록 필터(키워드)", value="", placeholder="예: DCM, Jet, 지반개량, 도심 ...")
    fm_view = fm
    if keyword.strip():
        kw = keyword.strip().lower()
        fm_view = fm[fm["라벨"].str.lower().str.contains(kw, na=False)].copy()

    options = fm_view["라벨"].tolist()
    label_to_id = dict(zip(fm_view["라벨"], fm_view["특성ID"]))

    # 기존 선택 복원(필터링 시에도 유지)
    master_label_to_id = dict(zip(fm["라벨"], fm["특성ID"]))
    master_id_to_label = {}
    for lab, fid in master_label_to_id.items():
        master_id_to_label.setdefault(fid, lab)

    current_selected_ids = st.session_state.get("selected_feature_ids", [])
    current_labels = [master_id_to_label[fid] for fid in current_selected_ids if fid in master_id_to_label]

    new_selected_labels = st.multiselect(
        "특성 선택(다중 선택 가능)",
        options=options,
        default=[lab for lab in current_labels if lab in options]
    )

    new_ids = [label_to_id[lab] for lab in new_selected_labels]
    kept_ids = [fid for fid in current_selected_ids if (fid in master_id_to_label and master_id_to_label[fid] not in options)]
    merged_ids = sorted(list(dict.fromkeys(kept_ids + new_ids)))
    st.session_state["selected_feature_ids"] = merged_ids

    st.markdown("#### ✅ 선택된 특성ID")
    if merged_ids:
        st.write(merged_ids)
        del_ids = st.multiselect("제거할 특성ID 선택", options=merged_ids, default=[])
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ 선택 제거"):
                st.session_state["selected_feature_ids"] = [x for x in merged_ids if x not in del_ids]
        with c2:
            if st.button("🧹 전체 초기화"):
                st.session_state["selected_feature_ids"] = []
    else:
        st.info("선택된 특성이 없습니다.")

    # auto_sites 계산
    if st.session_state["selected_feature_ids"]:
        auto_sites = (
            project_feature_long[
                project_feature_long["특성ID"].astype(str).isin([str(x) for x in st.session_state["selected_feature_ids"]])
            ]["현장코드"].astype(str).unique().tolist()
        )
    else:
        auto_sites = []

    # 표준화 + 정렬해서 session 저장
    new_auto_sites = sorted({
        norm_site_code(x)
        for x in (auto_sites or [])
        if norm_site_code(x)
    })
    st.session_state["auto_sites"] = new_auto_sites

    st.success(f"자동 후보 현장: {len(new_auto_sites)}개")
    if len(new_auto_sites) <= 30:
        st.write(new_auto_sites)

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("BOQ 업로드 후 프로젝트 특성을 선택할 수 있습니다.")


# =========================
# (3) 사이드바: 실적 현장 선택
# =========================
selected_site_codes = None

if use_site_filter:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏗️ 실적 현장 선택")

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

    # 2) auto_sites -> auto_codes (존재하는 코드만)
    auto_codes_raw = [norm_site_code(x) for x in (auto_sites or [])]
    auto_codes = [c for c in auto_codes_raw if c in code_to_name]

    other_codes = [c for c in all_codes if c not in set(auto_codes)]

    # ✅ 표시용(현장명만, 최대 25자)
    def fmt_site_code(code: str) -> str:
        name = code_to_name.get(code, "")
        name = name.strip()
        if len(name) > 25:
            return name[:25] + "…"
        return name

    # =========================
    # ✅ auto 후보가 바뀌면: 자동 후보를 "즉시 전체 선택" 상태로 세팅
    # =========================
    auto_sig = "|".join(auto_codes)

    if st.session_state.get("_auto_sig") != auto_sig:
        st.session_state["_auto_sig"] = auto_sig
        st.session_state["selected_auto_codes"] = list(auto_codes)

    if "selected_auto_codes" not in st.session_state:
        st.session_state["selected_auto_codes"] = list(auto_codes)
    if "selected_extra_codes" not in st.session_state:
        st.session_state["selected_extra_codes"] = []

    # ✅ 코드로 선택하되, 화면에는 현장명만 보이게(format_func)
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
    st.sidebar.caption(f"선택 현장: {len(selected_site_codes)}개")


# =========================
# 기타 슬라이더/통화 선택
# =========================
sim_threshold = st.sidebar.slider("Threshold (컷 기준, %)", 0, 100, 60, 5)
cut_ratio = st.sidebar.slider("상/하위 컷 비율 (%)", 0, 30, 20, 5) / 100.0

target_options = sorted(factor["국가"].astype(str).str.upper().unique().tolist())
default_idx = target_options.index("KRW") if "KRW" in target_options else 0
target_currency = st.sidebar.selectbox("산출통화", options=target_options, index=default_idx)

missing_exchange = exchange[exchange["통화"].astype(str).str.upper()==target_currency].empty
missing_factor   = factor[factor["국가"].astype(str).str.upper()==target_currency].empty
if missing_exchange:
    st.sidebar.error(f"선택한 산출통화 '{target_currency}'에 대한 환율 정보가 exchange.xlsx에 없습니다.")
if missing_factor:
    st.sidebar.error(f"선택한 산출통화 '{target_currency}'에 대한 지수 정보가 Factor.xlsx에 없습니다.")


# =========================
# Run / Auto Recompute
# =========================
# ✅ 자동 재산출 토글(사이드바)
auto_recompute = True  # ✅ UI는 숨기지만 기능은 항상 ON

def boq_file_signature(uploaded_file) -> str:
    """BOQ 파일이 바뀌었는지 감지하기 위한 간단 서명(해시)."""
    if uploaded_file is None:
        return "no_boq"
    try:
        b = uploaded_file.getvalue()
        # 너무 크면 앞/뒤 일부만 해시
        if len(b) > 2_000_000:
            b = b[:1_000_000] + b[-1_000_000:]
        return hashlib.md5(b).hexdigest()
    except Exception:
        # fallback
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
    """'산출 실행'과 동일한 효과: 계산 → session_state 저장 → 편집값 초기화"""
    if boq_file is None:
        st.warning("BOQ 파일을 업로드해 주세요.")
        return
    if missing_exchange or missing_factor:
        st.error("산출통화에 필요한 환율/지수 정보가 없습니다.")
        return

    boq = pd.read_excel(boq_file, engine="openpyxl")

    if use_site_filter and selected_site_codes is not None:
        cost_db_run = cost_db[
            cost_db["현장코드"].apply(norm_site_code).isin([norm_site_code(x) for x in selected_site_codes])
        ].copy()
    else:
        cost_db_run = cost_db.copy()

    st.sidebar.caption(
    f"전체 {len(cost_db):,}개 내역 중 {len(cost_db_run):,}개 내역으로 산출 실행"
    )

    progress = st.progress(0.0)
    prog_text = st.empty()

    # ✅ 후보풀 재사용을 위한 시그니처(현장필터/BOQ/DB가 바뀔 때만 새로 생성)
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

    # 1) 후보풀 생성(무거움) - 필요한 경우에만
    need_new_pool = (st.session_state.get("candidate_pool_sig") != pool_sig) or ("candidate_pool" not in st.session_state)

    if need_new_pool:
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

    # 2) 빠른 재계산(가벼움) - Threshold/컷/산출통화는 여기서만 반영
    with st.spinner("빠른 재계산(Threshold/컷/산출통화 반영 중)..."):
        result_df, log_df = fast_recompute_from_pool(
            pool=pool,
            exchange=exchange,
            factor=factor,
            sim_threshold=sim_threshold,
            cut_ratio=cut_ratio,
            target_currency=target_currency,
        )

    progress.progress(1.0)
    prog_text.text("산출 진행률: 완료")
    st.success("✅ 완료! 결과 확인 및 다운로드 가능")

    # ✅ 계산 결과 저장 (rerun 되어도 유지)
    st.session_state["boq_df"] = boq
    st.session_state["result_df_base"] = result_df.copy()
    st.session_state["log_df_base"] = log_df.copy()
    st.session_state["log_df_edited"] = log_df.copy()   # ✅ 편집값 초기화(재산출=실행과 동일효과)
    st.session_state.pop("result_df_adjusted", None)    # ✅ 조정 결과 초기화
    st.session_state["has_results"] = True

    # ✅ 이번 실행 조건 서명 저장
    st.session_state["last_run_sig"] = run_sig


# =========================
# (1) 실행 트리거 결정
# =========================
run_btn = st.sidebar.button("🚀 산출 실행")

current_sig = make_params_signature()
last_sig = st.session_state.get("last_run_sig", None)

needs_rerun = (last_sig is not None and current_sig != last_sig)

# 자동 재산출 OFF인데 조건 바뀐 경우 → 경고
if st.session_state.get("has_results", False) and needs_rerun and not auto_recompute:
    st.sidebar.warning("⚠️ 조건이 변경되었습니다. 다시 산출 실행이 필요합니다.")

# 자동 재산출 ON이고, 결과가 이미 있고, 조건 바뀌면 → 자동 실행
auto_run = st.session_state.get("has_results", False) and needs_rerun and auto_recompute

# 최초 실행(결과 없음)인데 auto_recompute 켜져 있어도, 버튼 없이 자동 실행은 부담될 수 있어 기본은 안 함
# 원하면 아래 조건을 확장해서 'BOQ 업로드 시 자동 1회 실행'도 가능

# =========================
# (2) 버튼 실행 또는 자동 실행
# =========================
if run_btn or auto_run:
    # 자동 재산출이면 사용자 편집이 초기화될 수 있으니 안내
    if auto_run:
        st.sidebar.info("ℹ️ 조건 변경 감지 → 자동 재산출 중 (로그 편집값은 초기화됩니다)")
    run_calculation_and_store(current_sig)


# =========================
# (3) 결과 화면: 결과가 있으면 항상 표시
# =========================
if st.session_state.get("has_results", False):
    boq = st.session_state["boq_df"]
    result_df = st.session_state["result_df_base"]
    log_df = st.session_state["log_df_base"]

    # -------------------------
    # 로그 Include 기준으로 결과 재계산 함수
    # -------------------------
    def recompute_result_from_log(edited_log: pd.DataFrame) -> pd.DataFrame:
        base = st.session_state["result_df_base"].copy()

        out_prices = []
        for boq_id, g in edited_log.groupby("BOQ_ID"):
            g2 = g[g["Include"] == True].copy()
            if g2.empty:
                out_prices.append((int(boq_id), None, "매칭 후보 없음(또는 전부 제외)", ""))
                continue

            final_price = float(pd.to_numeric(g2["__adj_price"], errors="coerce").mean())

            currencies = sorted(g2["통화"].astype(str).str.upper().unique().tolist())
            reason_text = f"{len(currencies)}개국({', '.join(currencies)}) {len(g2)}개 내역 근거"

            vc = g2["공종코드"].astype(str).value_counts()
            top_code = vc.index[0] if len(vc) else ""
            top_cnt = int(vc.iloc[0]) if len(vc) else 0
            top_work = f"{top_code} ({top_cnt}/{len(g2)})" if top_code else ""

            out_prices.append((int(boq_id), f"{final_price:,.2f}", reason_text, top_work))

        upd = pd.DataFrame(out_prices, columns=["BOQ_ID", "Final Price", "산출근거", "근거공종(최빈)"])
        base = base.drop(columns=[c for c in ["Final Price", "산출근거", "근거공종(최빈)"] if c in base.columns], errors="ignore")
        base = base.merge(upd, on="BOQ_ID", how="left")
        return base

    tab1, tab2, tab3 = st.tabs(["📄 BOQ 결과", "🧾 산출 로그(편집 가능)", "📝 근거 보고서"])

    with tab2:
        st.caption("✅ 체크 해제하면 평균단가 산출에서 제외됩니다. 체크하면 포함됩니다.")

        if "log_df_edited" not in st.session_state:
            st.session_state["log_df_edited"] = log_df.copy()

        log_all = st.session_state["log_df_edited"]

        # ✅ BOQ 선택을 "ID | 내역"으로 표시
        boq_ids = sorted(log_all["BOQ_ID"].dropna().astype(int).unique().tolist())

        # result_df_base에서 BOQ_ID별 내역 텍스트 가져오기(있으면 더 정확)
        base_for_label = st.session_state.get("result_df_base", pd.DataFrame()).copy()
        boq_text_col = "내역" if ("내역" in base_for_label.columns) else None

        id_to_text = {}
        if boq_text_col and ("BOQ_ID" in base_for_label.columns):
            id_to_text = (
                base_for_label.dropna(subset=["BOQ_ID"])
                .assign(BOQ_ID=lambda d: d["BOQ_ID"].astype(int))
                .set_index("BOQ_ID")[boq_text_col]
                .astype(str)
                .to_dict()
            )
        else:
            # fallback: log_df의 BOQ_내역 사용
            tmp_map = (
                log_all.dropna(subset=["BOQ_ID"])
                .assign(BOQ_ID=lambda d: d["BOQ_ID"].astype(int))
                .groupby("BOQ_ID")["BOQ_내역"].first()
                .astype(str)
                .to_dict()
            )
            id_to_text = tmp_map

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

        # ✅ 선택된 BOQ 후보만
        log_view_full = log_all[log_all["BOQ_ID"].astype(int) == int(sel_id)].copy()
        # =========================
        # 🤖 AI 추천 컨트롤 (현재 BOQ / 전체 BOQ)
        # =========================
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

                # BOQ_ID 기준 Include 복원
                cur = cur.drop(columns=["Include"], errors="ignore").merge(b, on="BOQ_ID", how="left")
                cur["Include"] = cur["Include"].fillna(False).astype(bool)

                st.session_state["log_df_edited"] = cur
                st.session_state["result_df_adjusted"] = recompute_result_from_log(st.session_state["log_df_edited"])
                st.success("되돌리기 완료(전체 BOQ)")
                st.rerun()

        # -------------------------
        # ✅ 화면 표시용 컬럼(열 순서 고정)
        # - 산출단가(__adj_price)은 앞쪽 유지
        # - 산출근거: 물가 → 환율 → 국가 순
        # - BOQ_ID/BOQ_내역/BOQ_Unit은 화면에서 숨김(다운로드에는 남아있음)
        # -------------------------
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

        # ✅ 편집(Include만)
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

        # ✅ 가장 안전한 반영 방식: 원본 인덱스로 Include만 업데이트
        st.session_state["log_df_edited"].loc[log_view_full.index, "Include"] = edited_view["Include"].values

        # ✅ 즉시 BOQ 결과 재계산
        st.session_state["result_df_adjusted"] = recompute_result_from_log(st.session_state["log_df_edited"])

    with tab1:
        show_df = st.session_state.get("result_df_adjusted", result_df).copy()
        if "통화" in show_df.columns:
            show_df = show_df.drop(columns=["통화"])
        st.dataframe(show_df, use_container_width=True)
   
    with tab3:
        st.markdown("## 📝 근거 보고서(자동 생성)")
        st.caption("현재 Include(포함) 상태 + 조건/선택 현장/특성 + (AI 적용 시) 최종 기준을 포함합니다.")
    
        base_result = st.session_state.get("result_df_adjusted", st.session_state.get("result_df_base", pd.DataFrame()))
        log_for_report = st.session_state.get("log_df_edited", st.session_state.get("log_df_base", pd.DataFrame()))
    
        # 1) 찾아야 할 공종 특성(선택된 프로젝트 특성)
        st.markdown("### 1) 찾아야 할 공종 특성(선택된 프로젝트 특성)")
        sel_features = st.session_state.get("selected_feature_ids", [])
        ft = build_feature_context_table(feature_master, sel_features)
        if ft.empty:
            st.info("선택된 특성ID가 없습니다.")
        else:
            st.dataframe(ft, use_container_width=True)
    
        # 2) 찾은 실적 현장 리스트(최종 선택 현장)
        st.markdown("### 2) 찾은 실적 현장 리스트(최종 선택 현장)")
        try:
            _sel_sites = selected_site_codes if (selected_site_codes is not None) else []
        except Exception:
            _sel_sites = []
        st_sites = build_site_context_table(cost_db, _sel_sites)
        if st_sites.empty:
            st.info("선택된 현장이 없습니다(또는 현장 필터 미사용).")
        else:
            st.dataframe(st_sites, use_container_width=True)
    
        # 3) 단가 추출 근거(조건)
        st.markdown("### 3) 단가 추출 근거(조건)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Threshold(컷 기준, %)", f"{float(sim_threshold):.0f}")
        with c2:
            st.metric("상/하위 컷 비율(%)", f"{float(cut_ratio)*100:.0f}")
        with c3:
            st.metric("산출통화", str(target_currency))
    
        # 4) AI 적용 시 최종 기준
        st.markdown("### 4) AI 적용 시 최종 기준")
        st.write(get_ai_effective_rule_text())
    
        # 5) 실적 단가 BOQ(결과)
        st.markdown("### 5) 실적 단가 BOQ(결과)")
        if base_result is None or base_result.empty:
            st.warning("결과 데이터가 없습니다. 먼저 산출 실행 후 다시 시도하세요.")
        else:
            st.dataframe(base_result, use_container_width=True)
    
        # 6) 보고서 테이블 생성/갱신
        if st.button("📝 보고서 생성/갱신", key="btn_build_report"):
            summary_df, detail_df = build_report_tables(log_for_report, base_result)
            st.session_state["report_summary_df"] = summary_df
            st.session_state["report_detail_df"] = detail_df
    
        summary_df = st.session_state.get("report_summary_df", pd.DataFrame())
        detail_df = st.session_state.get("report_detail_df", pd.DataFrame())
    
        st.markdown("### 6) 각 내역별 단가 근거(요약)")
        if summary_df is None or summary_df.empty:
            st.info("보고서를 보려면 '보고서 생성/갱신'을 눌러주세요.")
        else:
            st.dataframe(summary_df, use_container_width=True)
    
        st.markdown("### 7) 각 내역별 단가 근거(상세: Include=True 후보)")
        if detail_df is not None and not detail_df.empty:
            st.dataframe(detail_df, use_container_width=True)
        else:
            st.info("Include=True 상세 후보가 없습니다(전부 제외되었거나 후보가 없음).")
    
        # 8) 분포 그래프(전체/선택)
        st.markdown("### 8) 내역별 단가 점분포(계약년월 vs 단가) - 포함/미포함")
        render_boq_scatter(log_for_report, base_result)
    
        # 다운로드도 조정값 기준
        out_result = st.session_state.get("result_df_adjusted", result_df).copy()
        out_log = st.session_state.get("log_df_edited", log_df).copy()

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        out_result.to_excel(writer, index=False, sheet_name="boq_with_price")
        out_log.to_excel(writer, index=False, sheet_name="calculation_log")
        rep_sum = st.session_state.get("report_summary_df", pd.DataFrame())
        rep_det = st.session_state.get("report_detail_df", pd.DataFrame())
        if rep_sum is not None and not rep_sum.empty:
            rep_sum.to_excel(writer, index=False, sheet_name="report_summary")
        if rep_det is not None and not rep_det.empty:
            rep_det.to_excel(writer, index=False, sheet_name="report_detail")
    bio.seek(0)
    st.download_button("⬇️ Excel 다운로드", data=bio.read(), file_name="result_unitrate.xlsx")






























