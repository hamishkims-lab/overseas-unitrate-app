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

        # -------------------------
        # (A) 컷 계산 + Include 기본값 지정
        # -------------------------
        unit_df = unit_df.sort_values("__adj_price").reset_index(drop=True)
        n = len(unit_df)
        cut = max(0, int(n * cut_ratio)) if n > 5 else 0

        # 컷 적용 후 남길 인덱스 범위
        if cut > 0:
            keep_mask = np.zeros(n, dtype=bool)
            keep_mask[cut:n-cut] = True
        else:
            keep_mask = np.ones(n, dtype=bool)

        unit_df["Include"] = keep_mask  # ✅ 사용자가 log에서 수정할 컬럼
        unit_df["DefaultInclude"] = keep_mask  # 참고용(원래 기본값)

        # -------------------------
        # (B) 산출로그(후보행 단위) 누적
        # -------------------------
        boq_id = int(i)  # 1부터 증가 (loop의 i 사용)
        log_cols = [
            # BOQ 메타
            "BOQ_ID",
            "BOQ_내역",
            "BOQ_Unit",

            # 후보 핵심
            "Include",
            "DefaultInclude",
            "공종코드",
            "공종명",
            "내역",
            "Unit",
            "Unit Price",
            "통화",
            "계약년월",
            "현장코드",
            "현장명",
            "협력사코드",
            "협력사명",

            # 점수/보정
            "__hyb",
            "__adj_price",
            "산출통화",
            "__cpi_ratio",
            "__fx_ratio",
            "__fac_ratio",
            "__latest_ym",
        ]

        tmp = unit_df.copy()
        tmp["BOQ_ID"] = boq_id
        tmp["BOQ_내역"] = boq_item
        tmp["BOQ_Unit"] = boq_unit
        tmp["산출통화"] = target_currency

        # 없을 수 있는 컬럼 대비(안전)
        for c in log_cols:
            if c not in tmp.columns:
                tmp[c] = None

        logs.extend(tmp[log_cols].to_dict("records"))

        # -------------------------
        # (C) Include=True 기준으로 Final Price 계산 + 공종 분포(A안)
        # -------------------------
        inc = unit_df[unit_df["Include"] == True].copy()

        if inc.empty:
            final_price = None
            reason_text = "매칭 후보 없음(또는 전부 제외)"
            top_work = ""
        else:
            final_price = float(inc["__adj_price"].mean())

            currencies = sorted(inc["통화"].astype(str).str.upper().unique().tolist())
            reason_text = f"{len(currencies)}개국({', '.join(currencies)}) {len(inc)}개 내역 근거"

            # ✅ A안: 후보 공종코드 최빈값(Top1) 표시
            vc = inc["공종코드"].astype(str).value_counts()
            top_code = vc.index[0] if len(vc) else ""
            top_cnt = int(vc.iloc[0]) if len(vc) else 0
            top_work = f"{top_code} ({top_cnt}/{len(inc)})" if top_code else ""

        res_row = dict(boq_row)
        res_row["BOQ_ID"] = boq_id
        res_row["Final Price"] = f"{final_price:,.2f}" if final_price is not None else None
        res_row["산출근거"] = reason_text
        res_row["근거공종(최빈)"] = top_work
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
    boq_file = st.file_uploader("📤 BOQ 파일 업로드", type=["xlsx"])
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

        # auto_sites 계산
        if st.session_state["selected_feature_ids"]:
            auto_sites = (
                project_feature_long[
                    project_feature_long["특성ID"].astype(str).isin([str(x) for x in st.session_state["selected_feature_ids"]])
                ]["현장코드"].astype(str).unique().tolist()
            )
        else:
            auto_sites = []

        st.session_state["auto_sites"] = auto_sites

        # =========================
        # ✅ auto_sites 변경 시: 사이드바 선택 UI 강제 갱신 (안전 버전)
        # =========================
        # 1) 표준화 + 정렬(순서 안정화)
        new_auto_sites = sorted({
            norm_site_code(x)
            for x in (auto_sites or [])
            if norm_site_code(x)
        })

        # 2) 이전 값(표준화 + 정렬)
        old_auto_sites = sorted({
            norm_site_code(x)
            for x in (st.session_state.get("auto_sites", []) or [])
            if norm_site_code(x)
        })

        # 3) 변경된 경우에만 session 업데이트 + 사이드바 multiselect key 제거 + rerun 1회
        if new_auto_sites != old_auto_sites:
            st.session_state["auto_sites"] = new_auto_sites

            # ✅ 사이드바 multiselect key만 제거 (default 갱신 목적)
            for k in ["selected_auto_labels", "selected_extra_labels"]:
                if k in st.session_state:
                    del st.session_state[k]

            st.rerun()
        else:
            st.session_state["auto_sites"] = new_auto_sites

        st.success(f"자동 후보 현장: {len(new_auto_sites)}개")

        if len(new_auto_sites) <= 30:
            st.write(new_auto_sites)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("BOQ 업로드 후 프로젝트 특성을 선택할 수 있습니다.")


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

    # =========================
    # ✅ auto 후보가 바뀌면: 사이드바 자동 후보를 "즉시 전체 선택" 상태로 세팅
    #    (사용자는 체크 해제로 제외 가능)
    # =========================
    auto_sig = "|".join(auto_labels)  # auto 후보가 달라지면 시그니처도 달라짐

    # 1) auto 후보가 바뀐 최초 1회에만 '전체 선택'으로 초기화
    if st.session_state.get("_auto_sig") != auto_sig:
        st.session_state["_auto_sig"] = auto_sig
        st.session_state["selected_auto_labels"] = list(auto_labels)

    # 2) 키가 아예 없으면(최초 진입 등) 기본값 세팅
    if "selected_auto_labels" not in st.session_state:
        st.session_state["selected_auto_labels"] = list(auto_labels)
    if "selected_extra_labels" not in st.session_state:
        st.session_state["selected_extra_labels"] = []

    # 3) default를 쓰지 말고 session_state 값으로 렌더링
    selected_auto_labels = st.sidebar.multiselect(
        "자동 후보(제외 가능)",
        options=auto_labels,
        key="selected_auto_labels",
    )
    selected_auto_codes = [x.split(" | ")[0] for x in selected_auto_labels]

    selected_extra_labels = st.sidebar.multiselect(
        "기타 현장(추가 가능)",
        options=other_labels,
        key="selected_extra_labels",
    )
    selected_extra_codes = [x.split(" | ")[0] for x in selected_extra_labels]

    selected_site_codes = sorted(set(selected_auto_codes + selected_extra_codes))
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
# Run 버튼 (계산은 버튼에서만, 화면은 session_state 있으면 항상 표시)
# =========================
run_btn = st.sidebar.button("🚀 산출 실행")

# 1) 버튼을 눌렀을 때만 계산 수행 + session_state 저장
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

        # ✅ 계산 결과를 session_state에 저장 (rerun 되어도 유지)
        st.session_state["boq_df"] = boq
        st.session_state["result_df_base"] = result_df.copy()
        st.session_state["log_df_base"] = log_df.copy()
        st.session_state["has_results"] = True

        # 편집본이 있으면 최신 계산 기준으로 리셋(원하면 이 줄은 지워도 됨)
        st.session_state["log_df_edited"] = log_df.copy()
        st.session_state.pop("result_df_adjusted", None)

# 2) 버튼을 안 눌러도, 결과가 있으면 항상 결과/로그 UI를 보여줌
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

    tab1, tab2 = st.tabs(["📄 BOQ 결과", "🧾 산출 로그(편집 가능)"])

    with tab2:
        st.caption("✅ 체크 해제하면 평균단가 산출에서 제외됩니다. 체크하면 포함됩니다.")

        if "log_df_edited" not in st.session_state:
            st.session_state["log_df_edited"] = log_df.copy()

        log_all = st.session_state["log_df_edited"]

        # ✅ BOQ 선택 옵션을 "ID | 내역" 형태로 보기 좋게
        boq_ids = sorted(log_all["BOQ_ID"].dropna().astype(int).unique().tolist())

        # result_df_base에 BOQ_ID가 있고 BOQ 원문 내역 컬럼(예: '내역')이 있다고 가정
        base_for_label = st.session_state["result_df_base"].copy()
        # BOQ 원 내역 컬럼명이 다르면 여기만 바꿔주세요.
        boq_text_col = "내역" if "내역" in base_for_label.columns else "BOQ_내역"

        id_to_text = (
            base_for_label.dropna(subset=["BOQ_ID"])
            .assign(BOQ_ID=lambda d: d["BOQ_ID"].astype(int))
            .set_index("BOQ_ID")[boq_text_col]
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
            key="sel_boq_id",
        )

        # ✅ 선택된 BOQ 후보만
        log_view_full = log_all[log_all["BOQ_ID"].astype(int) == int(sel_id)].copy()

        # -------------------------
        # ✅ 화면 표시용 컬럼 구성/순서 (BOQ정보는 숨김)
        # -------------------------
        display_cols = [
            "Include", "DefaultInclude",
            "내역", "Unit",
            "Unit Price", "통화",
            "__adj_price", "산출통화",
            "__cpi_ratio", "__fx_ratio", "__fac_ratio", "__latest_ym",
            "__hyb",
            "공종코드", "공종명",
            "현장코드", "현장명",
            "협력사코드", "협력사명",
        ]

        # 없는 컬럼 대비(안전)
        for c in display_cols:
            if c not in log_view_full.columns:
                log_view_full[c] = None

        log_view = log_view_full[display_cols].copy()

        # ✅ 내역 폭 넓히기 + 라벨 바꾸기(가독성)
        edited_view = st.data_editor(
            log_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Include": st.column_config.CheckboxColumn("포함(평균 반영)", help="체크 해제하면 평균단가 산출에서 제외"),
                "DefaultInclude": st.column_config.CheckboxColumn("기본포함", help="초기 자동 포함 여부(컷 로직 결과)"),

                "내역": st.column_config.TextColumn("내역", width="large"),
                "Unit": st.column_config.TextColumn("Unit"),

                "Unit Price": st.column_config.NumberColumn("원단가(Unit Price)", format="%.4f"),
                "통화": st.column_config.TextColumn("원통화"),

                "__adj_price": st.column_config.NumberColumn("산출단가(보정후)", format="%.4f"),
                "산출통화": st.column_config.TextColumn("산출통화"),

                "__cpi_ratio": st.column_config.NumberColumn("CPI 보정", format="%.6f"),
                "__fx_ratio": st.column_config.NumberColumn("환율 보정", format="%.6f"),
                "__fac_ratio": st.column_config.NumberColumn("Factor 보정", format="%.6f"),
                "__latest_ym": st.column_config.TextColumn("CPI 최신월"),

                "__hyb": st.column_config.NumberColumn("유사도점수", format="%.2f"),

                "공종코드": st.column_config.TextColumn("공종코드"),
                "공종명": st.column_config.TextColumn("공종명"),

                "현장코드": st.column_config.TextColumn("현장코드"),
                "현장명": st.column_config.TextColumn("현장명"),

                "협력사코드": st.column_config.TextColumn("협력사코드"),
                "협력사명": st.column_config.TextColumn("협력사명"),
            },
            # ✅ Include만 편집 가능
            disabled=[c for c in log_view.columns if c not in ["Include"]],
            key="log_editor",
        )

        # -------------------------
        # ✅ 편집 반영: 원본(log_all)의 Include만 업데이트
        # -------------------------
        log_all_updated = log_all.copy()
        mask = log_all_updated["BOQ_ID"].astype(int) == int(sel_id)

        # 행수 불일치 방지(안전)
        if mask.sum() == len(edited_view):
            log_all_updated.loc[mask, "Include"] = edited_view["Include"].values
            st.session_state["log_df_edited"] = log_all_updated

            # 편집 즉시 결과 재계산
            st.session_state["result_df_adjusted"] = recompute_result_from_log(st.session_state["log_df_edited"])
        else:
            st.warning("로그 행수가 일치하지 않아 Include 반영을 건너뛰었습니다. 다시 선택해 주세요.")

        # BOQ_ID 단위로 Include만 반영
        log_all_updated = log_all.copy()
        mask = log_all_updated["BOQ_ID"].astype(int) == int(sel_id)

        # 행수 불일치 방지(안전)
        if mask.sum() == len(edited_view):
            log_all_updated.loc[mask, "Include"] = edited_view["Include"].values
            st.session_state["log_df_edited"] = log_all_updated

            # 편집 즉시 결과 재계산
            st.session_state["result_df_adjusted"] = recompute_result_from_log(st.session_state["log_df_edited"])
        else:
            st.warning("로그 행수가 일치하지 않아 Include 반영을 건너뛰었습니다. 다시 선택해 주세요.")

    with tab1:
        show_df = st.session_state.get("result_df_adjusted", result_df).copy()
        if "통화" in show_df.columns:
            show_df = show_df.drop(columns=["통화"])
        st.dataframe(show_df, use_container_width=True)

    # 다운로드도 조정값 기준
    out_result = st.session_state.get("result_df_adjusted", result_df).copy()
    out_log = st.session_state.get("log_df_edited", log_df).copy()

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        out_result.to_excel(writer, index=False, sheet_name="boq_with_price")
        out_log.to_excel(writer, index=False, sheet_name="calculation_log")
    bio.seek(0)
    st.download_button("⬇️ Excel 다운로드", data=bio.read(), file_name="result_unitrate.xlsx")












