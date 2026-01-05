# 📦 해외 실적단가 자동 산출 APP

해외 실적단가 DB를 활용하여  
BOQ(공사 내역서)를 업로드하면 단가를 자동 산출하는  
Streamlit 기반 웹 애플리케이션입니다.

---

## 🔧 주요 기능
- BOQ 내역과 해외 실적단가 **하이브리드 매칭**
  - 문자열 유사도 + 의미 유사도(임베딩)
- 계약시점 대비 **CPI 보정**
- **환율 변환(USD 기준)**
- **건설지수(Factor) 보정**
- 극단값 제거 후 평균 단가 산출
- 산출 결과 및 로그 **엑셀 다운로드**

---

## 📁 프로젝트 구조
```text
overseas-unitrate-app/
├─ app.py
├─ requirements.txt
├─ README.md
└─ data/
   ├─ cost_db.xlsx
   ├─ price_index.xlsx
   ├─ exchange.xlsx
   └─ Factor.xlsx
