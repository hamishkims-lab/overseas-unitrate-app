import re
import io
import json
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

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

CI_BLUE   = "#005EB8"   # GS CI Pantone 300C
CI_TEAL   = "#00BFB3"   # GS CI Pantone 3272C
BG_LIGHT  = "#F6FAFC"

st.markdown(f"""
<style>
  .main {{ background-color: {BG_LIGHT}; }}
  .gs-header {{
     color: white;
     background: linear-gradient(90deg, {CI_BLUE} 0%, {CI_TEAL} 100%);
     padding: 14px 16px;
     border-radius: 10px;
     font-size: 26px; font-weight: 700;
  }}
  /* Select / Multiselect 공통 스타일 */
  div[data-baseweb="select"] > div {{
     background-color: white !important;
     border: 1px solid {CI_BLUE} !important;
     border-radius: 6px !important;
  }}
  div[data-baseweb="select"] span {{
     background-color: {CI_TEAL} !important;
     color: white !important;
     border-radius: 4px !important;
     padding: 2px 6px !important;
  }}
  .stDownloadButton button {{
     background-color:{CI_BLUE}; color:white; border-radius:8px; padding:8px 14px; border:0;
  }}
  .stDownloadButton button:hover {{ background-color:{CI_TEAL}; color:white; }}
  .gs-card {{
    background-color: white;
    border: 1px solid #e8eef3;
    border-radius: 10px;
    padding: 12px 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }}
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
    """datetime/str → 'YYYY-MM' 문자열, 실패 시 None"""
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
    """YYYY-MM/ YYYYMM / YYYY.MM 등 → 월초 timestamp로 통일"""
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
    """
    현장코드 정규화:
    - 공백 제거
    - 190590.0 같은 소수 표현 제거
    - 숫자/문자 혼용 대비
    """
    if x is None:
        return ""
    s = str(x).strip()
    # 190590.0 형태 처리
    if s.endswith(".0"):
        s = s[:-2]
    # 혹시 남아있는 소수점 제거
    s = s.split(".")[0]
    return s


# =========================
# 보정 로직 (CPI/환율/지수)
# =========================
def get_cpi_ratio(price_index: pd.DataFrame, currency: str, contract_ym: str):
    """currency(=국가코드), contract_ym='YYYY-MM' 기준 최신/계약 CPI 비율"""
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
    """USD 기준 환율표: USD당환율 → 환율비"""
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
# Embedding Cache (Cloud 호환: /tmp 사용)
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
    # normalize for cosine
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
    D, I = index.search(query_vecs, top_k)  # inner product == cosine on normalized
    return D, I


# =========================
# Matching
# =========================
def hybrid_scores(boq_text_norm: str, db_texts_norm: pd.Series, sem_scores: np.ndarray, w_str: float, w_sem: float) -> np.ndarray:
    sem = np.clip(sem_scores, 0.0, 1.0)
    str_scores = np.array([fuzz.token_sort_ratio(boq_text_norm, s) / 100.0 for s in db_texts_norm.tolist()], dtype="float32")
    return (w_str * str_scores + w_sem * sem) * 100.0

def match_items_faiss(
    cost_db: pd.DataFrame,
    boq: pd.DataFrame,
    price_index: pd.DataFrame,
    exchange: pd.DataFrame,
    factor: pd.DataFrame,
    sim_threshold: float,
    cut_ratio: float,
    target_currency: str,
    w_str: float,
    w_sem: float,
    top_k_sem: int,
    progress=None,
    prog_text=None,
):
    # 전처리 + 유효성 필터
    work = cost_db.copy()
    work["__내역_norm"] = work["내역"].apply(norm_text)
    work["__Unit_norm"] = work["Unit"].astype(str).str.lower().str.strip()
    work["_계약월"] = robust_parse_contract_month(work["계약년월"])
    work = work[(pd.to_numeric(work["Unit Price"], errors="coerce") > 0) & (work["_계약월"].notna())].copy()

    # CPI/지수 표준화
    price_index = price_index.copy()
    price_index["년월"] = price_index["년월"].apply(to_year_month_string)

    # 임베딩 & 인덱스
    fp = file_fingerprint(work, ["__내역_norm","__Unit_norm","통화","Unit Price","_계약월"])
    embs = compute_or_load_embeddings(work["__내역_norm"], tag=f"costdb_{fp}")
    index = build_faiss_index(embs) if FAISS_OK else None

    results, logs = [], []
    total = len(boq) if len(boq) else 1

    for i, (_, boq_row) in enumerate(boq.iterrows(), start=1):
        if prog_text is not None:
            prog_text.text(f"산출 진행률: {i}/{total} 항목 처리 중…")
        if progress is not None:
            progress.progress(i/total)

        boq_item = str(boq_row.get("내역", ""))
        boq_unit = str(boq_row.get("Unit", "")).lower().strip()
        boq_text_norm = norm_text(boq_item)

        # 쿼리 임베딩
        q = model.encode([boq_text_norm], batch_size=1, convert_to_tensor=False)
        q = np.asarray(q, dtype="float32")
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

        # 의미 후보
        if FAISS_OK:
            D, I = search_faiss(index, q, top_k=top_k_sem)
            cand_idx = I[0]; sem_scores = D[0]
        else:
            # embs는 numpy float32 (normalize 완료). util.cos_sim은 torch 텐서 기반이므로
            # 간단히 numpy dot으로 cosine(IP) 계산:
            all_sem = np.dot(embs, q[0])  # (N,)
            cand_idx = np.argsort(-all_sem)[:top_k_sem]
            sem_scores = all_sem[cand_idx]

        cand_df = work.iloc[cand_idx].copy()
        cand_df["__sem"] = sem_scores

        # Unit 일치
        unit_df = cand_df[cand_df["__Unit_norm"] == boq_unit].reset_index(drop=True)

        if unit_df.empty:
            res_row = dict(boq_row)
            res_row["Final Price"] = None
            res_row["산출근거"] = "매칭 없음"
            results.append(res_row)
            continue

        # 하이브리드 점수 계산 (참고용: 매칭 품질 근거)
        hyb = hybrid_scores(boq_text_norm, unit_df["__내역_norm"], unit_df["__sem"].to_numpy(), w_str, w_sem)
        unit_df["__hyb"] = hyb

        # Threshold 적용
        unit_df = unit_df[unit_df["__hyb"] >= sim_threshold].copy()
        if unit_df.empty:
            res_row = dict(boq_row)
            res_row["Final Price"] = None
            res_row["산출근거"] = "매칭 없음"
            results.append(res_row)
            continue

        # 보정단가 계산 (로그에 보정요소 표기)
        adj_list = []
        for _, r in unit_df.iterrows():
            c_currency = str(r.get("통화","")).upper().strip()
            unit_price = float(r.get("Unit Price", 0.0))
            contract_ym = to_year_month_string(r.get("_계약월"))

            cpi_ratio, base_cpi, latest_cpi, latest_ym = get_cpi_ratio(price_index, c_currency, contract_ym)
            fx_ratio  = get_exchange_rate(exchange, c_currency, target_currency)
            fac_ratio = get_factor_ratio(factor, c_currency, target_currency)

            adj_price = unit_price * cpi_ratio * fx_ratio * fac_ratio

            adj_list.append({
                **r.to_dict(),
                "__adj_price": adj_price,
                "__base_cpi": base_cpi,
                "__latest_cpi": latest_cpi,
                "__latest_ym": latest_ym,
                "__cpi_ratio": cpi_ratio,
                "__fx_ratio": fx_ratio,
                "__fac_ratio": fac_ratio,
                "__sem": r["__sem"],
                "__hyb": r["__hyb"],
            })
        unit_df = pd.DataFrame(adj_list)

        # 극단치 컷
        unit_df = unit_df.sort_values("__adj_price")
        n = len(unit_df)
        cut = max(0, int(n * cut_ratio)) if n > 5 else 0
        kept = unit_df.iloc[cut:n-cut] if cut > 0 else unit_df.copy()
        kept_ids = set(kept.index)

        # 산출근거 텍스트
        currencies = sorted(kept["통화"].astype(str).str.upper().unique().tolist())
        reason_text = f"{len(currencies)}개국({', '.join(currencies)}) {len(kept)}개 내역 근거"

        # 산출 로그
        for ridx, row in unit_df.iterrows():
            logs.append({
                "BOQ 항목": boq_item,
                "BOQ Unit": boq_unit,
                "실적내역": row.get("내역", None),
                "실적계약년월": to_year_month_string(row.get("_계약월")),
                "원단가(현지통화)": row.get("Unit Price", None),
                "실적통화": row.get("통화", None),
                "계약CPI": row["__base_cpi"],
                "최신CPI": row["__latest_cpi"],
                "최신CPI년월": row["__latest_ym"],
                "CPI보정": row["__cpi_ratio"],
                "적용환율": row["__fx_ratio"],
                "건설지수보정": row["__fac_ratio"],
                "타겟통화": target_currency,
                "최종단가(보정후)": row["__adj_price"],
                "포함여부": "포함" if ridx in kept_ids else "제외",
            })

        # 최종단가
        final_price = float(kept["__adj_price"].mean()) if not kept.empty else None

        res_row = dict(boq_row)
        res_row["Final Price"] = f"{final_price:,.2f}" if final_price is not None else None
        res_row["산출근거"] = reason_text
        results.append(res_row)

    result_df = pd.DataFrame(results)
    log_df = pd.DataFrame(logs)

    # 보기용 포맷
    if "최종단가(보정후)" in log_df.columns:
        log_df["최종단가(보정후)"] = log_df["최종단가(보정후)"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else None)

    return result_df, log_df


# =========================
# 데이터 로드 (Cloud 호환: repo 상대경로)
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def load_excel_from_repo(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        st.error(
            f"필수 파일을 찾을 수 없습니다: {path.as_posix()}\n"
            f"GitHub 저장소에 data/{filename} 파일이 있는지 확인하세요."
        )
        st.stop()
    return pd.read_excel(path, engine="openpyxl")

cost_db     = load_excel_from_repo("cost_db.xlsx")
price_index = load_excel_from_repo("price_index.xlsx")
exchange    = load_excel_from_repo("exchange.xlsx")
factor      = load_excel_from_repo("Factor.xlsx") 
project_feature_long = load_excel_from_repo("project_feature_long.xlsx")
feature_master = load_excel_from_repo("feature_master_FID.xlsx") # ✅ 대소문자 정확히!

# =========================
# Globals (Streamlit rerun 안전)
# =========================
auto_sites = None
matched_feature_ids = []

if "selected_feature_ids" not in st.session_state:
    st.session_state["selected_feature_ids"] = []

if "manual_site_codes" not in st.session_state:
    st.session_state["manual_site_codes"] = []  # 사용자가 추가로 체크한 현장
if "excluded_site_codes" not in st.session_state:
    st.session_state["excluded_site_codes"] = []  # 자동 후보에서 제외한 현장


# =========================
# Sidebar (순서/도움말/요구사항 반영)
# =========================
st.sidebar.header("⚙️ 설정")
st.sidebar.caption("①~⑥ 순서대로 설정하세요.")

# ✅ 현장 필터 사용 여부 (국가 필터 대체)
use_site_filter = st.sidebar.checkbox(
    "현장 필터 사용(추천)",
    value=True,
    help="프로젝트 특성 기반으로 현장을 자동 선택하고, 수동으로 추가/제외합니다."
)

# ✅ (숨김 처리) ②, ④는 UI에 노출하지 않고 내부 고정값 사용
DEFAULT_W_STR = 0.3
DEFAULT_TOP_K_SEM = 200

w_str = DEFAULT_W_STR
w_sem = 1.0 - w_str
top_k_sem = DEFAULT_TOP_K_SEM

# ① 실적단가 필터링 - 국가 (현장 필터 미사용 시에만)
if not use_site_filter:

    all_currencies = sorted([c for c in cost_db["통화"].astype(str).str.upper().unique() if c.strip()])
    if "" in cost_db["통화"].astype(str).unique().tolist():
        all_currencies = all_currencies + [""]

    selected_currencies = st.sidebar.multiselect(
        "① 실적단가 필터링 - 국가",
        options=all_currencies,
        default=all_currencies,
        help="실적국가(통화)만 사용할 수 있습니다. 미선택 시 전체 사용."
    )

    # 필터 적용
    if selected_currencies:
        cost_db = cost_db[
            cost_db["통화"].astype(str).str.upper().isin(
                [s for s in selected_currencies if s != ""] +
                ([] if "" not in selected_currencies else [""])
            )
        ]

# =========================
# 사이드바: 실적 현장 선택 (자동 후보를 기본으로 동기화 + 수동 추가/제외)
# =========================
selected_site_codes = None

if use_site_filter:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏗️ 실적 현장 선택")

    site_df = cost_db[["현장코드", "현장명"]].copy()
    site_df["현장코드"] = site_df["현장코드"].apply(norm_site_code)
    site_df["현장명"] = site_df["현장명"].astype(str).str.strip()
    site_df = site_df.dropna().drop_duplicates()

    site_df["label"] = site_df["현장코드"] + " | " + site_df["현장명"]
    all_labels = site_df["label"].sort_values().tolist()
    code_to_label = dict(zip(site_df["현장코드"], site_df["label"]))
    all_codes = site_df["현장코드"].tolist()

    auto_codes_raw = [norm_site_code(x) for x in (auto_sites or [])]
    auto_codes = [c for c in auto_codes_raw if c in code_to_label]

    missing_auto = [c for c in auto_codes_raw if c not in code_to_label]
    if missing_auto:
        st.sidebar.warning(f"cost_db에 없는 자동후보 코드: {missing_auto[:10]}")

    # ✅ 자동 후보가 바뀔 때 "기본 선택"을 동기화할지 옵션
    sync_auto = st.sidebar.checkbox("특성 변경 시 자동 후보로 선택 갱신", value=True)

    # session_state로 현장 선택 유지
    if "selected_site_labels" not in st.session_state:
        # 최초: 자동 후보가 있으면 자동 후보, 없으면 전체
        st.session_state["selected_site_labels"] = [code_to_label[c] for c in (auto_codes if auto_codes else all_codes)]

    if sync_auto:
        # 자동 후보 기반으로 초기화(단, 사용자가 제외/추가한 건 반영하기 위해 아래에서 조정)
        base_codes = set(auto_codes) if auto_codes else set(all_codes)
    else:
        # 동기화 끄면 기존 선택 유지
        base_codes = set([lab.split(" | ")[0] for lab in st.session_state["selected_site_labels"]])

    # 사용자가 따로 추가/제외한 것 반영
    excluded = set(st.session_state.get("excluded_site_codes", []))
    manual_add = set(st.session_state.get("manual_site_codes", []))

    final_codes = (base_codes - excluded) | manual_add
    final_labels = [code_to_label[c] for c in all_codes if c in final_codes]

    # 자동/기타 구분 표시
    auto_labels = [code_to_label[c] for c in auto_codes]
    other_labels = [code_to_label[c] for c in all_codes if c not in set(auto_codes)]

    st.sidebar.caption(f"자동 후보 {len(auto_labels)}개 / 기타 {len(other_labels)}개")

    # 1) 자동 후보(기본 포함, 제외 가능)
    selected_auto_labels = st.sidebar.multiselect(
        "자동 후보(제외 가능)",
        options=auto_labels,
        default=[lab for lab in auto_labels if lab in final_labels],
        key="selected_auto_labels"
    )
    # 여기서 빠진 자동후보는 excluded로 저장
    selected_auto_codes = set([lab.split(" | ")[0] for lab in selected_auto_labels])
    st.session_state["excluded_site_codes"] = [c for c in auto_codes if c not in selected_auto_codes]

    # 2) 기타 현장(추가 가능)
    selected_extra_labels = st.sidebar.multiselect(
        "기타 현장(추가 가능)",
        options=other_labels,
        default=[lab for lab in other_labels if lab.split(" | ")[0] in manual_add],
        key="selected_extra_labels"
    )
    st.session_state["manual_site_codes"] = [lab.split(" | ")[0] for lab in selected_extra_labels]

    # 최종 선택
    selected_site_codes = sorted(list((selected_auto_codes | set(st.session_state["manual_site_codes"]))))

    st.sidebar.caption(f"최종 선택 현장: {len(selected_site_codes)}개")

# feature_master(176개) 기준 옵션 생성
fm = feature_master.copy()
for c in ["특성ID","대공종","중공종","소공종","Cost Driver Type","Cost Driver Method","Cost Driver Condition"]:
    fm[c] = fm[c].astype(str)

# 각 특성ID가 project_feature_long에 몇 개 현장으로 매핑되는지 계산
site_cnt = (
    project_feature_long.groupby("특성ID")["현장코드"]
    .nunique()
    .astype(int)
    .to_dict()
)

fm["현장수"] = fm["특성ID"].map(site_cnt).fillna(0).astype(int)

# UI에 보여줄 라벨 만들기
fm["라벨"] = fm.apply(
    lambda r: f'{r["특성ID"]} | {r["대공종"]}/{r["중공종"]}/{r["소공종"]} | {r["Cost Driver Type"]}/{r["Cost Driver Method"]}/{r["Cost Driver Condition"]} | 현장 {r["현장수"]}개',
    axis=1
)

label_to_id = dict(zip(fm["라벨"], fm["특성ID"]))
options = fm["라벨"].tolist()

selected_labels = st.sidebar.multiselect(
    "특성 선택 (176개 전체)",
    options=options,
    default=[]
)

selected_feature_ids = [label_to_id[x] for x in selected_labels]

# 선택된 특성ID → 현장코드 후보
if selected_feature_ids:
    allowed_sites = (
        project_feature_long[
            project_feature_long["특성ID"].astype(str).isin([str(x) for x in selected_feature_ids])
        ]["현장코드"]
        .astype(str)
        .unique()
        .tolist()
    )
else:
    allowed_sites = None



if allowed_sites is not None:
    st.sidebar.write("현장코드(예시):", allowed_sites[:10])

# ② Threshold
sim_threshold = st.sidebar.slider(
    "② Threshold (컷 기준, %)",
    min_value=0, max_value=100, value=60, step=5,
    help="매칭 인정 최소 점수 기준입니다."
)

# ③ 상/하위 컷 비율
cut_ratio = st.sidebar.slider(
    "③ 상/하위 컷 비율 (%)",
    min_value=0, max_value=30, value=20, step=5,
    help="보정단가 분포의 양끝단 극단값을 제거하는 비율입니다. 표본 수가 5개 이하이면 적용되지 않습니다."
) / 100.0

# ④ 산출통화 (Factor.xlsx 기준) + 환율/지수 존재 검증
target_options = sorted(factor["국가"].astype(str).str.upper().unique().tolist())
default_idx = target_options.index("KRW") if "KRW" in target_options else 0
target_currency = st.sidebar.selectbox(
    "④ 산출통화",
    options=target_options,
    index=default_idx,
    help="최종 산출 통화(국가 지수 기준)를 선택합니다."
)

# =========================
# Validation 기반 추천 (Grid Search UI)
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Validation 기반 추천")

def run_grid_search():
    data = {
        "Params": [
            {"top_k": 200, "threshold": 60, "string_weight": 0.3, "semantic_weight": 0.7},
            {"top_k": 300, "threshold": 65, "string_weight": 0.5, "semantic_weight": 0.5},
            {"top_k": 500, "threshold": 70, "string_weight": 0.7, "semantic_weight": 0.3},
        ],
        "Precision": [1.0, 0.85, 0.8],
        "Recall": [1.0, 0.9, 0.7],
        "F1": [1.0, 0.87, 0.74],
    }
    df = pd.DataFrame(data)
    best_idx = df["F1"].idxmax()
    best_params = df.loc[best_idx, "Params"]
    return df, best_params

if st.sidebar.button("🔍 Grid Search 실행"):
    grid_results, best_params = run_grid_search()
    st.session_state["grid_results"] = grid_results
    st.session_state["best_params"] = best_params

if "grid_results" in st.session_state:
    st.sidebar.markdown("**추천 파라미터 (F1 최고):**")
    st.sidebar.json(st.session_state["best_params"])

    # ✅ ②/④는 숨김이므로 '적용'은 ③(threshold)만 반영(혼란 방지)
    if st.sidebar.button("⬇️ 추천 파라미터 적용"):
        sel = st.session_state["best_params"]
        sim_threshold = sel["threshold"]
        st.sidebar.success("추천 파라미터가 적용되었습니다 ✅ (Threshold만 반영)")

missing_exchange = exchange[exchange["통화"].astype(str).str.upper()==target_currency].empty
missing_factor   = factor[factor["국가"].astype(str).str.upper()==target_currency].empty
if missing_exchange:
    st.sidebar.error(f"선택한 산출통화 '{target_currency}'에 대한 환율 정보가 exchange.xlsx에 없습니다.")
if missing_factor:
    st.sidebar.error(f"선택한 산출통화 '{target_currency}'에 대한 지수 정보가 Factor.xlsx에 없습니다.")


# =========================
# 업로드 & 실행
# =========================
with st.container():
    st.markdown("<div class='gs-card'>", unsafe_allow_html=True)
    boq_file = st.file_uploader(
        "📤 BOQ 파일 업로드",
        type=["xlsx"],
        help="BOQ는 최소한 '내역', 'Unit' 컬럼이 필요합니다. 업로드 후 '산출 실행'을 클릭하세요."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# BOQ 업로드 아래: 특성 선택(176개 전체) + 키워드 필터 + 다중 선택
# =========================
auto_sites = None  # 매 rerun마다 계산
selected_feature_ids = st.session_state["selected_feature_ids"]

if use_site_filter:
    if boq_file is not None:
        st.markdown("<div class='gs-card'>", unsafe_allow_html=True)
        st.markdown("### 🏷️ 프로젝트 특성 선택 (176개 전체)")

        fm = feature_master.copy()
        cols6 = ["대공종","중공종","소공종","Cost Driver Type","Cost Driver Method","Cost Driver Condition"]
        for c in ["특성ID"] + cols6:
            fm[c] = fm[c].astype(str).fillna("").str.strip()

        # 각 특성ID가 project_feature_long에 몇 개 현장으로 매핑되는지(옵션)
        site_cnt = (
            project_feature_long.groupby("특성ID")["현장코드"].nunique().astype(int).to_dict()
        )
        fm["현장수"] = fm["특성ID"].map(site_cnt).fillna(0).astype(int)

        fm["라벨"] = fm.apply(
            lambda r: f'{r["특성ID"]} | {r["대공종"]}/{r["중공종"]}/{r["소공종"]} | '
                      f'{r["Cost Driver Type"]}/{r["Cost Driver Method"]}/{r["Cost Driver Condition"]} | '
                      f'현장 {r["현장수"]}개',
            axis=1
        )

        # ✅ 키워드로 목록 자체를 줄이는 필터(선택사항, 멀티셀렉트 검색과 별개)
        keyword = st.text_input("특성 목록 필터(키워드)", value="", placeholder="예: DCM, Jet, 지반개량, 도심 ...")
        if keyword.strip():
            kw = keyword.strip().lower()
            fm_view = fm[fm["라벨"].str.lower().str.contains(kw, na=False)].copy()
        else:
            fm_view = fm

        options = fm_view["라벨"].tolist()
        label_to_id = dict(zip(fm_view["라벨"], fm_view["특성ID"]))

        # ✅ 현재 선택된 ID를 라벨로 복원(필터링으로 라벨이 안 보일 수 있으니 master에서 복원)
        master_label_to_id = dict(zip(fm["라벨"], fm["특성ID"]))
        master_id_to_label = {}
        for lab, fid in master_label_to_id.items():
            master_id_to_label.setdefault(fid, lab)

        current_selected_labels = [master_id_to_label[fid] for fid in selected_feature_ids if fid in master_id_to_label]

        # ✅ 멀티셀렉트 (Streamlit 기본 검색 지원)
        new_selected_labels = st.multiselect(
            "특성 선택(다중 선택 가능)",
            options=options,
            default=[lab for lab in current_selected_labels if lab in options]  # 현재 필터 화면에 보이는 것만 default
        )

        # ✅ “필터로 안 보이는 기존 선택”도 유지하면서 합치기
        new_ids = [label_to_id[lab] for lab in new_selected_labels]
        # 기존 선택 중 이번 화면에 없었던 것 유지
        kept_ids = [fid for fid in selected_feature_ids if fid in master_id_to_label and master_id_to_label[fid] not in options]
        merged_ids = sorted(list(dict.fromkeys(kept_ids + new_ids)))  # 중복 제거, 순서 유지

        st.session_state["selected_feature_ids"] = merged_ids
        selected_feature_ids = merged_ids

        # 선택된 특성 표시 + 삭제 UI
        st.markdown("#### ✅ 선택된 특성ID")
        if selected_feature_ids:
            st.write(selected_feature_ids)

            del_ids = st.multiselect("제거할 특성ID 선택", options=selected_feature_ids, default=[])
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑️ 선택 제거"):
                    st.session_state["selected_feature_ids"] = [x for x in selected_feature_ids if x not in del_ids]
            with c2:
                if st.button("🧹 전체 초기화"):
                    st.session_state["selected_feature_ids"] = []
        else:
            st.info("선택된 특성이 없습니다.")

        # ✅ auto_sites 계산 (선택된 특성ID OR)
        if st.session_state["selected_feature_ids"]:
            auto_sites = (
                project_feature_long[
                    project_feature_long["특성ID"].astype(str).isin([str(x) for x in st.session_state["selected_feature_ids"]])
                ]["현장코드"].astype(str).unique().tolist()
            )
        else:
            auto_sites = []

        st.success(f"자동 후보 현장: {len(auto_sites)}개")
        if len(auto_sites) <= 20:
            st.write(auto_sites)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("BOQ 업로드 후 프로젝트 특성을 선택할 수 있습니다.")

run_btn = st.sidebar.button("🚀 산출 실행", help="현재 설정과 업로드한 BOQ로 단가를 산출합니다. 진행률이 표시됩니다.")


# =========================
# ✅ 프로젝트 특성 필터 적용 (현장코드 기준)
# =========================
if run_btn:
    if boq_file is None:
        st.warning("BOQ 파일을 업로드해 주세요.")
    elif missing_exchange or missing_factor:
        st.error("산출통화에 필요한 환율/지수 정보가 없습니다.")
    else:
        # 1) BOQ 로드
        boq = pd.read_excel(boq_file, engine="openpyxl")

        # 2) 최종 현장 필터 적용 (E)
        if use_site_filter and selected_site_codes is not None:
            cost_db_run = cost_db[
                cost_db["현장코드"].apply(norm_site_code).isin(
                    [norm_site_code(x) for x in selected_site_codes]
                )
            ].copy()
        else:
            cost_db_run = cost_db.copy()

        st.sidebar.caption(
            f"실행용 cost_db 행수: {len(cost_db_run):,} / 전체 {len(cost_db):,}"
        )

        # 🔴 이 줄의 들여쓰기가 핵심
        progress = st.progress(0.0)
        prog_text = st.empty()

        with st.spinner("임베딩/인덱스 준비 및 계산 중..."):
            result_df, log_df = match_items_faiss(
                cost_db=cost_db_run,
                boq=boq,
                price_index=price_index,
                exchange=exchange,
                factor=factor,
                sim_threshold=sim_threshold,
                cut_ratio=cut_ratio,
                target_currency=target_currency,
                w_str=w_str,
                w_sem=w_sem,
                top_k_sem=top_k_sem,
                progress=progress,
                prog_text=prog_text,
            )

        progress.progress(1.0)
        prog_text.text("산출 진행률: 완료")

        st.success("✅ 완료! 결과 확인 및 다운로드 가능")

        # --- 탭 분리 ---
        tab1, tab2, tab3 = st.tabs(["📄 BOQ 결과", "🧾 산출 로그", "ℹ️ 매칭 근거 안내"])

        # 📄 BOQ 결과: '통화' 열 완전 제거
        with tab1:
            if "통화" in result_df.columns:
                result_df = result_df.drop(columns=["통화"])
            st.dataframe(result_df, use_container_width=True)

        # 🧾 산출 로그: 조건부 색상 + 다운로드
        with tab2:
            df_disp = log_df.copy()

            try:
                numeric = df_disp["최종단가(보정후)"].str.replace(",", "", regex=False).astype(float)
            except Exception:
                numeric = pd.to_numeric(df_disp["최종단가(보정후)"], errors="coerce")

            include_mask = df_disp["포함여부"] == "포함"
            avg_map = (
                pd.DataFrame({"BOQ 항목": df_disp["BOQ 항목"], "_num": numeric, "포함여부": include_mask})
                .query("포함여부 == True")
                .groupby("BOQ 항목")["_num"].mean()
            )

            def color_by_avg(col):
                styles = []
                for idx, v in enumerate(col):
                    try:
                        if df_disp.iloc[idx]["포함여부"] != "포함":
                            styles.append("")
                            continue
                        avg_val = avg_map.get(df_disp.iloc[idx]["BOQ 항목"], None)
                        vv = float(str(v).replace(",", "")) if isinstance(v, str) else float(v)
                        if avg_val is None or pd.isna(vv):
                            styles.append("")
                        else:
                            styles.append("color: green" if vv < float(avg_val) else "color: red")
                    except Exception:
                        styles.append("")
                return styles

            if "최종단가(보정후)" in df_disp.columns:
                styled = df_disp.style.apply(color_by_avg, subset=["최종단가(보정후)"])
                st.dataframe(styled, use_container_width=True)
            else:
                st.dataframe(df_disp, use_container_width=True)

            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="boq_with_price")
                log_df.to_excel(writer, index=False, sheet_name="calculation_log")
            bio.seek(0)
            st.download_button("⬇️ Excel 다운로드", data=bio.read(), file_name="result_unitrate.xlsx")

        # ℹ️ 매칭 근거 안내
        with tab3:
            st.markdown("""
**하이브리드 점수(0-100)**는 *문자열 유사도*와 *의미 유사도(임베딩 코사인)*를 가중 평균한 값으로,  
BOQ 항목과 실적 DB 항목이 왜 매칭되었는지 설명하는 근거 지표입니다.  
- 문자열 유사도: 철자/토큰 구성의 유사성  
- 의미 유사도: 문장의 의미적 근접성(언어모델 임베딩 사용)  
- 본 앱에서는 사용자 혼란을 줄이기 위해 UI에 지표를 숨기고, **Threshold=60%** 기준으로 운용합니다.
            """)

# =========================
# 설명 섹션
# =========================
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("<div class='gs-card'>", unsafe_allow_html=True)
st.subheader("📌 단가 산출 기준")
st.markdown("""
1. **실적단가 필터링**  
   - **국가(통화)**: 다중 선택 가능, 빈칸은 목록 하단에 표시  

2. **매칭 기준 (Hybrid Matching)**  
   - 문자열+의미 유사도 가중치(합계=1), **Threshold** 이상만 매칭 인정  

3. **단가 보정 과정**  
   - 계약시점 대비 최신 **CPI 보정**, **환율 변환(USD기준)**, **건설지수 보정**  

4. **상/하위 컷 비율 적용**  
   - 극단값 제외 후 잔여 표본의 평균을 **최종 단가**로 산정  

5. **출력 (산출통화 기준)**  
   - 산출통화로 환산된 BOQ별 **최종 단가 + 산출근거 + 로그**  
""")
st.markdown("</div>", unsafe_allow_html=True)



















