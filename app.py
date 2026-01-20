import re
import io
import json
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
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
  .gs-header {{
     color: white;
     background: linear-gradient(90deg, {CI_BLUE} 0%, {CI_TEAL} 100%);
     padding: 14px 16px;
     border-radius: 10px;
     font-size: 26px; font-weight: 700;
  }}
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
    work = cost_db.copy()
    work["__내역_norm"] = work["내역"].apply(norm_text)
    work["__Unit_norm"] = work["Unit"].astype(str).str.lower().str.strip()
    work["_계약월"] = robust_parse_contract_month(work["계약년월"])
    work = work[(pd.to_numeric(work["Unit Price"], errors="coerce") > 0) & (work["_계약월"].notna())].copy()

    price_index = price_index.copy()
    price_index["년월"] = price_index["년월"].apply(to_year_month_string)

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

        unit_df = cand_df[cand_df["__Unit_norm"] == boq_unit].reset_index(drop=True)
        if unit_df.empty:
            res_row = dict(boq_row)
            res_row["Final Price"] = None
            res_row["산출근거"] = "매칭 없음"
            results.append(res_row)
            continue

        hyb = hybrid_scores(boq_text_norm, unit_df["__내역_norm"], unit_df["__sem"].to_numpy(), w_str, w_sem)
        unit_df["__hyb"] = hyb

        unit_df = unit_df[unit_df["__hyb"] >= sim_threshold].copy()
        if unit_df.empty:
            res_row = dict(boq_row)
            res_row["Final Price"] = None
            res_row["산출근거"] = "매칭 없음"
            results.append(res_row)
            continue

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
                "__hyb": r["__hyb"],
            })
        unit_df = pd.DataFrame(adj_list)

        unit_df = unit_df.sort_values("__adj_price")
        n = len(unit_df)
        cut = max(0, int(n * cut_ratio)) if n > 5 else 0
        kept = unit_df.iloc[cut:n-cut] if cut > 0 else unit_df.copy()

        currencies = sorted(kept["통화"].astype(str).str.upper().unique().tolist())
        reason_text = f"{len(currencies)}개국({', '.join(currencies)}) {len(kept)}개 내역 근거"

        final_price = float(kept["__adj_price"].mean()) if not kept.empty else None

        res_row = dict(boq_row)
        res_row["Final Price"] = f"{final_price:,.2f}" if final_price is not None else None
        res_row["산출근거"] = reason_text
        results.append(res_row)

    return pd.DataFrame(results), pd.DataFrame(logs)


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
st.sidebar.caption("①~⑥ 순서대로 설정하세요.")

use_site_filter = st.sidebar.checkbox(
    "현장 필터 사용(추천)",
    value=True
)

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
    boq_file = st.file_uploader(
        "📤 BOQ 파일 업로드",
        type=["xlsx"],
        help="BOQ는 최소한 '내역', 'Unit' 컬럼이 필요합니다."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# (2) 메인: BOQ 업로드 아래 특성 선택 UI
# =========================
auto_sites = []

if use_site_filter:
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

        current_selected_ids = st.session_state["selected_feature_ids"]
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
    else:
    st.info("BOQ 업로드 후 프로젝트 특성을 선택할 수 있습니다.")

        # =========================
        # BOQ 업로드 아래: auto_sites 계산
        # =========================
        if st.session_state["selected_feature_ids"]:
            auto_sites = (
                project_feature_long[
                    project_feature_long["특성ID"].astype(str).isin(
                        [str(x) for x in st.session_state["selected_feature_ids"]]
                    )
                ]["현장코드"].astype(str).unique().tolist()
            )
        else:
            auto_sites = []
        
        st.session_state["auto_sites"] = auto_sites
        
        # =========================
        # 사이드바 자동후보 즉시 선택 반영(자동선택)
        # =========================
        site_df = cost_db[["현장코드", "현장명"]].copy().dropna(subset=["현장코드"])
        site_df["현장코드"] = site_df["현장코드"].apply(norm_site_code)
        site_df["현장명"] = site_df["현장명"].astype(str).fillna("").str.strip()
        site_df.loc[site_df["현장명"].isin(["", "nan", "None"]), "현장명"] = "(현장명없음)"
        site_df = site_df.drop_duplicates(subset=["현장코드"])
        site_df["label"] = site_df["현장코드"] + " | " + site_df["현장명"]
        
        code_to_label = dict(zip(site_df["현장코드"], site_df["label"]))
        auto_codes = [norm_site_code(x) for x in auto_sites]
        auto_labels = [code_to_label[c] for c in auto_codes if c in code_to_label]
        
        # ✅ 사이드바 multiselect의 default를 “자동후보로 강제”
        st.session_state["selected_auto_labels"] = auto_labels
        
        st.success(f"자동 후보 현장: {len(auto_sites)}개")
        if len(auto_sites) <= 30:
            st.write(auto_sites)

       
# =========================
# (3) 사이드바: 실적 현장 선택 (auto_sites가 session에 저장된 이후에!)
# =========================
selected_site_codes = None

if use_site_filter:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏗️ 실적 현장 선택")

    # ✅ (선택) 디버그 초기화 버튼: 누르면 상태만 지우고 rerun
    if st.sidebar.button("🧹 강제 초기화(디버그)"):
        for k in ["selected_auto_labels", "selected_extra_labels", "auto_sites", "selected_feature_ids"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # ✅ 항상 auto_sites를 읽고 UI를 그려야 함 (버튼 if 밖!)
    auto_sites = st.session_state.get("auto_sites", [])

    # 1) cost_db에서 전체 현장 목록 만들기
    site_df = cost_db[["현장코드", "현장명"]].copy()
    site_df = site_df.dropna(subset=["현장코드"])

    site_df["현장코드"] = site_df["현장코드"].apply(norm_site_code)
    site_df["현장명"] = site_df["현장명"].astype(str).fillna("").str.strip()
    site_df.loc[site_df["현장명"].isin(["", "nan", "None"]), "현장명"] = "(현장명없음)"

    site_df = site_df.drop_duplicates(subset=["현장코드"])
    site_df["label"] = site_df["현장코드"] + " | " + site_df["현장명"]

    all_codes = site_df["현장코드"].tolist()
    code_to_label = dict(zip(site_df["현장코드"], site_df["label"]))

    # 2) auto_sites -> auto_codes (존재하는 코드만)
    auto_codes_raw = [norm_site_code(x) for x in (auto_sites or [])]
    auto_codes = [c for c in auto_codes_raw if c in code_to_label]

    auto_labels = [code_to_label[c] for c in auto_codes]
    other_labels = [code_to_label[c] for c in all_codes if c not in set(auto_codes)]

    st.sidebar.caption(f"자동 후보 {len(auto_labels)}개 / 기타 {len(other_labels)}개")

    # default는 session_state에 있으면 그걸 쓰고, 없으면 auto_labels
    default_auto = st.session_state.get("selected_auto_labels", auto_labels)
    
    selected_auto_labels = st.sidebar.multiselect(
        "자동 후보(제외 가능)",
        options=auto_labels,
        default=[x for x in default_auto if x in auto_labels],  # 옵션에 없는 건 제거
        key="selected_auto_labels"
    )
    selected_auto_codes = [x.split(" | ")[0] for x in selected_auto_labels]

    selected_extra_labels = st.sidebar.multiselect(
        "기타 현장(추가 가능)",
        options=other_labels,
        default=[],
        key="selected_extra_labels"
    )
    selected_extra_codes = [x.split(" | ")[0] for x in selected_extra_labels]

    selected_site_codes = sorted(list(set(selected_auto_codes + selected_extra_codes)))
    st.sidebar.caption(f"최종 선택 현장: {len(selected_site_codes)}개")


# =========================
# 기타 슬라이더/통화 선택
# =========================
sim_threshold = st.sidebar.slider("② Threshold (컷 기준, %)", 0, 100, 60, 5)
cut_ratio = st.sidebar.slider("③ 상/하위 컷 비율 (%)", 0, 30, 20, 5) / 100.0

target_options = sorted(factor["국가"].astype(str).str.upper().unique().tolist())
default_idx = target_options.index("KRW") if "KRW" in target_options else 0
target_currency = st.sidebar.selectbox("④ 산출통화", options=target_options, index=default_idx)

missing_exchange = exchange[exchange["통화"].astype(str).str.upper()==target_currency].empty
missing_factor   = factor[factor["국가"].astype(str).str.upper()==target_currency].empty
if missing_exchange:
    st.sidebar.error(f"선택한 산출통화 '{target_currency}'에 대한 환율 정보가 exchange.xlsx에 없습니다.")
if missing_factor:
    st.sidebar.error(f"선택한 산출통화 '{target_currency}'에 대한 지수 정보가 Factor.xlsx에 없습니다.")


# =========================
# Run 버튼
# =========================
run_btn = st.sidebar.button("🚀 산출 실행")

if run_btn:
    if boq_file is None:
        st.warning("BOQ 파일을 업로드해 주세요.")
    elif missing_exchange or missing_factor:
        st.error("산출통화에 필요한 환율/지수 정보가 없습니다.")
    else:
        boq = pd.read_excel(boq_file, engine="openpyxl")

        if use_site_filter and selected_site_codes is not None:
            cost_db_run = cost_db[
                cost_db["현장코드"].apply(norm_site_code).isin([norm_site_code(x) for x in selected_site_codes])
            ].copy()
        else:
            cost_db_run = cost_db.copy()

        st.sidebar.caption(f"실행용 cost_db 행수: {len(cost_db_run):,} / 전체 {len(cost_db):,}")

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

        tab1, tab2 = st.tabs(["📄 BOQ 결과", "🧾 산출 로그"])

        with tab1:
            if "통화" in result_df.columns:
                result_df = result_df.drop(columns=["통화"])
            st.dataframe(result_df, use_container_width=True)

        with tab2:
            st.dataframe(log_df, use_container_width=True)

        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False, sheet_name="boq_with_price")
            log_df.to_excel(writer, index=False, sheet_name="calculation_log")
        bio.seek(0)
        st.download_button("⬇️ Excel 다운로드", data=bio.read(), file_name="result_unitrate.xlsx")











