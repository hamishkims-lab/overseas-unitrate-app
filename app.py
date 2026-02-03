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

# =========================
# UI Labels / Constants
# =========================
LABEL_SIM_THRESHOLD = "매칭 유사도 기준값(%)"
LABEL_CUT_RATIO     = "상/하위 컷 비율 (%)"
LABEL_TARGET_CURR   = "산출통화"

CI_BLUE   = "#005EB8"
CI_TEAL   = "#00BFB3"
BG_LIGHT  = "#F6FAFC"

# =========================
# Session Init (안전장치)
# =========================
def init_session():
    defaults = {
        "selected_feature_ids": [],
        "auto_sites": [],
        "selected_auto_codes": [],
        "selected_extra_codes": [],
        "has_results": False,

        "candidate_pool": None,
        "candidate_pool_sig": None,
        "last_run_sig": None,

        "boq_df": None,
        "result_df_base": None,
        "log_df_base": None,
        "log_df_edited": None,
        "result_df_adjusted": None,

        "ai_last_applied": None,
        "_include_backup": {},
        "_include_backup_all": None,

        "report_summary_df": pd.DataFrame(),
        "report_detail_df": pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

st.markdown("""
<style>
  /* 전체 배경 */
  .main { background-color: #F6FAFC; }

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
    caret-color:#000000 !important;
  }

  /* ✅ placeholder/비활성 텍스트도 검정 계열로 */
  section[data-testid="stSidebar"] div[data-baseweb="select"] input::placeholder{
    color:#000000 !important;
    -webkit-text-fill-color:#000000 !important;
    opacity: 0.7 !important;
  }

  /* ✅ tag(칩) 자체는 한 줄 꽉 차게. X가 잘리지 않게 */
section[data-testid="stSidebar"] div[data-baseweb="tag"],
section[data-testid="stSidebar"] span[data-baseweb="tag"]{
  width: 100% !important;
  max-width: 100% !important;
  overflow: visible !important;

  display: inline-flex !important;
  align-items: center !important;
  gap: 8px !important;

  background-color:#4DA3FF !important;
  border:1px solid #2F80ED !important;
  color:#ffffff !important;

  padding: 0 10px !important;
  box-sizing: border-box !important;

  height: 30px !important;
  min-height: 30px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="tag"] > span:first-child,
section[data-testid="stSidebar"] span[data-baseweb="tag"] > span:first-child{
  flex: 1 1 auto !important;
  min-width: 0 !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;

  font-size: 12px !important;
  line-height: 1 !important;
  color:#ffffff !important;
}

section[data-testid="stSidebar"] div[data-baseweb="tag"] > span:last-child,
section[data-testid="stSidebar"] span[data-baseweb="tag"] > span:last-child{
  flex: 0 0 26px !important;
  width: 26px !important;
  min-width: 26px !important;

  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}

section[data-testid="stSidebar"] div[data-baseweb="tag"] svg,
section[data-testid="stSidebar"] span[data-baseweb="tag"] svg,
section[data-testid="stSidebar"] div[data-baseweb="tag"] path,
section[data-testid="stSidebar"] span[data-baseweb="tag"] path{
  fill:#ffffff !important;
}

/* ✅ (닫힌 상태) 선택된 값/아이콘 검정 */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div *{
  color:#000000 !important;
  -webkit-text-fill-color:#000000 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg,
section[data-testid="stSidebar"] div[data-baseweb="select"] svg path{
  fill:#000000 !important;
}
</style>
""", unsafe_allow_html=True)

def sidebar_hr(thick: bool = False, mt: int = 6, mb: int = 6):
    color = "#D9DDE3"
    h = "3px" if thick else "1px"
    st.sidebar.markdown(
        f"<hr style='margin:{mt}px 0 {mb}px 0; border:none; border-top:{h} solid {color};' />",
        unsafe_allow_html=True
    )

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

# (이하 build_candidate_pool / fast_recompute_from_pool / 에이전트 / 보고서 / 그래프 함수들은
#   사용자가 주신 원문 그대로여도 무방하므로 생략 없이 그대로 유지하시면 됩니다.)
# -------------------------------------------------------------------
# 여기부터는 사용자가 주신 원문 함수들을 그대로 두셔도 되고,
# 이미 붙여주신 코드 그대로 이어붙이셔도 됩니다.
# -------------------------------------------------------------------


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
        "특성ID","대공종","중공종","소공종",
        "Cost Driver Type","Cost Driver Method","Cost Driver Condition"
    ]
    for c in must_cols:
        if c not in df.columns:
            df[c] = ""

    return df

def ensure_columns(df: pd.DataFrame, must_cols: list, fill_value=None) -> pd.DataFrame:
    df = df.copy()
    for c in must_cols:
        if c not in df.columns:
            df[c] = fill_value
    return df

def normalize_loaded_tables():
    """
    로드 직후 표준화 + 필수 컬럼 보장.
    """
    global cost_db, price_index, exchange, factor, project_feature_long, feature_master

    def _safe_df(x):
        return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

    cost_db = _safe_df(cost_db)
    price_index = _safe_df(price_index)
    exchange = _safe_df(exchange)
    factor = _safe_df(factor)
    project_feature_long = _safe_df(project_feature_long)
    feature_master = _safe_df(feature_master)

    cost_db = standardize_columns(cost_db)
    price_index = standardize_columns(price_index)
    exchange = standardize_columns(exchange)
    factor = standardize_columns(factor)
    project_feature_long = standardize_columns(project_feature_long)
    feature_master = standardize_columns(feature_master)

    project_feature_long = apply_feature_column_alias(project_feature_long)
    feature_master = apply_feature_column_alias(feature_master)

    cost_db = ensure_columns(cost_db, [
        "내역", "Unit", "Unit Price", "통화", "계약년월",
        "현장코드", "현장명", "협력사코드", "협력사명", "공종코드", "공종명"
    ], fill_value="")

    price_index = ensure_columns(price_index, ["국가", "년월", "Index"], fill_value=np.nan)
    exchange = ensure_columns(exchange, ["통화", "USD당환율"], fill_value=np.nan)
    factor = ensure_columns(factor, ["국가", "지수"], fill_value=np.nan)

normalize_loaded_tables()

# =========================
# Sidebar: 설정
# =========================
st.sidebar.header("⚙️ 설정")
sidebar_hr(thick=True, mt=6, mb=6)

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
with st.container():
    st.markdown("<div class='gs-card'>", unsafe_allow_html=True)
    boq_file = st.file_uploader("📤 BOQ 파일 업로드", type=["xlsx"])
    st.markdown("</div>", unsafe_allow_html=True)

# (이하 특성 선택/현장 선택 로직은 사용자 원문 그대로 유지)

# =========================
# 기타 슬라이더/통화 선택  ✅ 여기만 “완전 교체”
# =========================
sidebar_hr(thick=True, mt=10, mb=6)
st.sidebar.subheader("🧩 설정값")
sidebar_hr(thick=False, mt=6, mb=8)

sim_threshold = st.sidebar.slider(LABEL_SIM_THRESHOLD, 0, 100, 60, 5)
cut_ratio = st.sidebar.slider(LABEL_CUT_RATIO, 0, 30, 20, 5) / 100.0

# ✅ target_options / default_idx를 반드시 먼저 만든 뒤 selectbox 호출
def build_target_options(exchange: pd.DataFrame, factor: pd.DataFrame) -> list:
    opts = set()
    if isinstance(exchange, pd.DataFrame) and "통화" in exchange.columns:
        opts |= set(exchange["통화"].astype(str).str.upper().dropna().tolist())
    if isinstance(factor, pd.DataFrame) and "국가" in factor.columns:
        opts |= set(factor["국가"].astype(str).str.upper().dropna().tolist())
    opts = sorted([x.strip() for x in opts if x and x.strip()])
    if not opts:
        opts = ["KRW"]
    return opts

target_options = build_target_options(exchange, factor)
default_idx = target_options.index("KRW") if "KRW" in target_options else 0

target_currency = st.sidebar.selectbox(
    LABEL_TARGET_CURR,
    options=target_options,
    index=default_idx
)

missing_exchange = exchange[exchange["통화"].astype(str).str.upper()==target_currency].empty
missing_factor   = factor[factor["국가"].astype(str).str.upper()==target_currency].empty

if missing_exchange:
    st.sidebar.error(f"선택한 산출통화 '{target_currency}'에 대한 환율 정보가 exchange.xlsx에 없습니다.")
if missing_factor:
    st.sidebar.error(f"선택한 산출통화 '{target_currency}'에 대한 지수 정보가 Factor.xlsx에 없습니다.")

sidebar_hr(thick=True, mt=10, mb=8)


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

    status_box = st.empty()
    progress = st.progress(0.0)
    prog_text = st.empty()

    try:
        if boq_file is None:
            status_box.empty()
            st.warning("BOQ 파일을 업로드해 주세요.")
            return
        if missing_exchange or missing_factor:
            status_box.empty()
            st.error("산출통화에 필요한 환율/지수 정보가 없습니다.")
            return

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

        # 1) 후보풀 생성
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

        # 2) 빠른 재계산
        status_box.markdown("### ⏳ 산출중... (조건 반영/산출통화 반영)")
        with st.spinner("빠른 재계산(조건 반영 중)..."):
            result_df, log_df = fast_recompute_from_pool(
                pool=pool,
                exchange=exchange,
                factor=factor,
                sim_threshold=sim_threshold,
                cut_ratio=cut_ratio,
                target_currency=target_currency,
            )

        st.session_state["boq_df"] = boq
        st.session_state["result_df_base"] = result_df.copy()
        st.session_state["log_df_base"] = log_df.copy()
        st.session_state["log_df_edited"] = log_df.copy()
        st.session_state.pop("result_df_adjusted", None)
        st.session_state["has_results"] = True
        st.session_state["last_run_sig"] = run_sig

    finally:
        # ✅ 어떤 상황이든 산출중 UI 제거
        try:
            prog_text.empty()
            progress.empty()
            status_box.empty()
        except Exception:
            pass

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
            columns=[c for c in ["Final Price","산출통화","산출근거","근거공종(최빈)"] if c in base.columns],
            errors="ignore"
        )
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
    
        # (원래 있던 통화 컬럼 제거 로직은 유지)
        if "통화" in show_df.columns:
            show_df = show_df.drop(columns=["통화"])
    
        # ✅ Final Price 바로 다음에 산출통화 위치시키기
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
            st.metric("유사도 컷오프 기준(%)", f"{float(sim_threshold):.0f}")
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












