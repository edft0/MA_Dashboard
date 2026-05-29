"""
애사비 신규 진입 시뮬레이션
합성 데이터 5,000명 | EDA 실측 + Laplace 보정 | 기저시장 + 트리거 + 리스크
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="애사비 시뮬레이션", layout="wide")

# ─────────────────────────────────────────
# 기저값 (신제품 → Active=0)
# ─────────────────────────────────────────
BASE = {
    '혈당관리 실천형': {'Unaware':368,'Aware':354,'Trial':247,'Active':58, 'Churn':317},
    '건강관심 잠재형': {'Unaware':363,'Aware':407,'Trial':231,'Active':84, 'Churn':364},
    '맛 편의 추구형':  {'Unaware':544,'Aware':594,'Trial':326,'Active':91, 'Churn':483},
    '가격 민감 실용성':{'Unaware': 41,'Aware': 53,'Trial': 20,'Active':16, 'Churn': 39},
}
SEGS = list(BASE.keys())
SEG_COLORS = {
    '혈당관리 실천형':'#2ECC71','건강관심 잠재형':'#4C9BE8',
    '맛 편의 추구형':'#9B59B6','가격 민감 실용성':'#E67E22',
}

# ─────────────────────────────────────────
# 기저 시장 — 제품 유형별 Aware→Trial 추가 전환율
# Triangular(mode×0.7, mode, mode×1.3)
# 근거: 비경험자 중 해당 형태 선호 비율
# ─────────────────────────────────────────
PRODUCT_WEIGHTS = {
    '액상 보틀 (기본)': {  # 기준선, 추가 전환 없음
        '혈당관리 실천형': (0.000, 0.000, 0.000),
        '건강관심 잠재형': (0.000, 0.000, 0.000),
        '맛 편의 추구형':  (0.000, 0.000, 0.000),
        '가격 민감 실용성':(0.000, 0.000, 0.000),
    },
    '액상 스틱': {  # 실측
        '혈당관리 실천형': (0.038, 0.054, 0.070),
        '건강관심 잠재형': (0.160, 0.229, 0.298),
        '맛 편의 추구형':  (0.106, 0.152, 0.198),
        '가격 민감 실용성':(0.070, 0.100, 0.130),  # Laplace
    },
    '분말 스틱': {  # 액상스틱 × 0.8
        '혈당관리 실천형': (0.030, 0.043, 0.056),
        '건강관심 잠재형': (0.128, 0.183, 0.238),
        '맛 편의 추구형':  (0.085, 0.122, 0.159),
        '가격 민감 실용성':(0.056, 0.080, 0.104),
    },
    '젤리': {  # 실측
        '혈당관리 실천형': (0.189, 0.270, 0.351),
        '건강관심 잠재형': (0.227, 0.325, 0.423),
        '맛 편의 추구형':  (0.441, 0.630, 0.819),
        '가격 민감 실용성':(0.070, 0.100, 0.130),  # Laplace
    },
    '알약': {  # 실측
        '혈당관리 실천형': (0.321, 0.459, 0.597),
        '건강관심 잠재형': (0.186, 0.265, 0.344),
        '맛 편의 추구형':  (0.061, 0.087, 0.113),
        '가격 민감 실용성':(0.420, 0.600, 0.780),  # Laplace
    },
    '음료': {  # 액상보틀 × 1.3
        '혈당관리 실천형': (0.197, 0.281, 0.365),
        '건강관심 잠재형': (0.164, 0.235, 0.306),
        '맛 편의 추구형':  (0.118, 0.169, 0.220),
        '가격 민감 실용성':(0.342, 0.488, 0.634),
    },
}
PRODUCT_DESC = {
    '액상 보틀 (기본)': '현재 시장 기준선. 추가 전환 없음.',
    '액상 스틱':  '스틱형 포장. 건강관심형·맛편의형 반응 높음.',
    '분말 스틱':  '스틱형 분말. 액상스틱 대비 보수적 적용(×0.8).',
    '젤리':       '맛편의형 63% 선호. 가장 큰 전환 효과.',
    '알약':       '혈당관리형 46%, 가격민감형 60% 선호.',
    '음료':       '접근성 최고. 액상보틀 대비 ×1.3 추정.',
}

# ─────────────────────────────────────────
# 트리거 가중치
# ─────────────────────────────────────────
TRIGGER_WEIGHTS = {
    'T1': {  # 효능·성분 콘텐츠 → Aware→Trial
        '혈당관리 실천형': (0.058, 0.083, 0.108),  # Laplace
        '건강관심 잠재형': (0.183, 0.262, 0.341),  # 실측
        '맛 편의 추구형':  (0.070, 0.100, 0.130),  # 실측
        '가격 민감 실용성':(0.350, 0.500, 0.650),  # Laplace
    },
    'T2': {  # 무료 샘플·맛 체험 → Aware→Trial
        '혈당관리 실천형': (0.058, 0.083, 0.108),  # Laplace
        '건강관심 잠재형': (0.050, 0.071, 0.092),  # 실측
        '맛 편의 추구형':  (0.245, 0.350, 0.455),  # 실측
        '가격 민감 실용성':(0.175, 0.250, 0.325),  # Laplace
    },
    'T3': {  # 가격 할인 쿠폰 → Aware→Trial
        '혈당관리 실천형': (0.292, 0.417, 0.542),  # Laplace
        '건강관심 잠재형': (0.083, 0.119, 0.155),  # 실측
        '맛 편의 추구형':  (0.035, 0.050, 0.065),  # 실측
        '가격 민감 실용성':(0.350, 0.500, 0.650),  # Laplace
    },
    'T4': {  # SNS 바이럴 → Unaware→Aware
        '혈당관리 실천형': (0.000, 0.000, 0.000),  # 실측 진짜 0
        '건강관심 잠재형': (0.077, 0.110, 0.143),  # 실측
        '맛 편의 추구형':  (0.108, 0.154, 0.200),  # 실측
        '가격 민감 실용성':(0.175, 0.250, 0.325),  # Laplace
    },
}

# ─────────────────────────────────────────
# 리스크 가중치
# ─────────────────────────────────────────
RISK_WEIGHTS = {
    'R1': {  # 효능 논란 → Trial↓ + Aware→Trial 억제
        '혈당관리 실천형': (0.069, 0.099, 0.129),
        '건강관심 잠재형': (0.025, 0.036, 0.047),
        '맛 편의 추구형':  (0.022, 0.032, 0.042),
        '가격 민감 실용성':(0.052, 0.074, 0.096),
    },
    'R2': {  # 경쟁사 저가 진입 → Aware→Trial↓
        '혈당관리 실천형': (0.082, 0.117, 0.152),
        '건강관심 잠재형': (0.025, 0.036, 0.047),
        '맛 편의 추구형':  (0.034, 0.048, 0.062),
        '가격 민감 실용성':(0.052, 0.074, 0.096),
    },
    'R3': {  # 대체재 출시 → Trial↓
        '혈당관리 실천형': (0.037, 0.053, 0.069),
        '건강관심 잠재형': (0.029, 0.042, 0.055),
        '맛 편의 추구형':  (0.118, 0.169, 0.220),
        '가격 민감 실용성':(0.032, 0.045, 0.059),  # Laplace
    },
}

# 히트맵용
VULN_DATA = {
    'R1 효능논란':   {'혈당관리 실천형':0.099,'건강관심 잠재형':0.036,'맛 편의 추구형':0.032,'가격 민감 실용성':0.074},
    'R2 경쟁사':     {'혈당관리 실천형':0.117,'건강관심 잠재형':0.036,'맛 편의 추구형':0.048,'가격 민감 실용성':0.074},
    'R3 대체재':     {'혈당관리 실천형':0.053,'건강관심 잠재형':0.042,'맛 편의 추구형':0.169,'가격 민감 실용성':0.045},
    'T1 효능콘텐츠': {'혈당관리 실천형':0.083,'건강관심 잠재형':0.262,'맛 편의 추구형':0.100,'가격 민감 실용성':0.500},
    'T2 샘플체험':   {'혈당관리 실천형':0.083,'건강관심 잠재형':0.071,'맛 편의 추구형':0.350,'가격 민감 실용성':0.250},
    'T3 가격할인':   {'혈당관리 실천형':0.417,'건강관심 잠재형':0.119,'맛 편의 추구형':0.050,'가격 민감 실용성':0.500},
    'T4 SNS바이럴':  {'혈당관리 실천형':0.000,'건강관심 잠재형':0.110,'맛 편의 추구형':0.154,'가격 민감 실용성':0.250},
}

def get_w(table, key, seg, use_mc=False):
    lo, mo, hi = table[key][seg]
    if mo == 0: return 0.0
    return float(np.random.triangular(lo, mo, hi)) if use_mc else mo

# ─────────────────────────────────────────
# 시뮬레이션
# ─────────────────────────────────────────
def run_sim(product, t1, t2, t3, t4, r1, r2, r3, use_mc=False):
    result = {}
    for seg in SEGS:
        uw = BASE[seg]['Unaware']
        aw = BASE[seg]['Aware']
        tr = BASE[seg]['Trial']
        ac = BASE[seg]['Active']
        ch = BASE[seg]['Churn']

        # ── 기저 시장: 제품 유형 → Aware→Trial 기저 전환
        dp = get_w(PRODUCT_WEIGHTS, product, seg, use_mc)
        if dp > 0:
            n_prod = min(aw, round(aw * dp))
            aw -= n_prod; tr += n_prod

        # ── 트리거
        # T4: Unaware→Aware (주) + Aware→Trial 부(×0.2)
        if t4 > 0:
            d4 = get_w(TRIGGER_WEIGHTS, 'T4', seg, use_mc) * (t4/5)
            n1 = min(uw, round(uw * d4))
            n2 = min(aw, round(aw * d4 * 0.2))
            uw -= n1; aw += n1
            aw -= n2; tr += n2

        # T1·T2·T3: Aware→Trial (주) + Unaware→Aware 부(×0.2) + Trial→Active 미세(×0.05)
        for tk, tv in [('T1',t1),('T2',t2),('T3',t3)]:
            if tv == 0: continue
            dm = get_w(TRIGGER_WEIGHTS, tk, seg, use_mc) * (tv/5)
            n1 = min(aw, round(aw * dm))
            n2 = min(uw, round(uw * dm * 0.2))
            n3 = min(tr, round(tr * dm * 0.05))
            aw -= n1; tr += n1
            uw -= n2; aw += n2
            tr -= n3; ac += n3

        # ── 리스크
        # R1: Trial↓ + Aware→Trial 억제 + Active→Churn
        if r1:
            dr1 = get_w(RISK_WEIGHTS, 'R1', seg, use_mc)
            n1 = min(tr, round(tr * dr1))
            n2 = min(tr, round(tr * dr1 * 0.3))
            n3 = min(ac, round(ac * dr1))  # Active도 일부 Churn으로
            tr -= n1; aw += n1
            tr -= n2
            ac -= n3; ch += n3

        # R2: Aware 이탈 + Active 일부 Churn
        if r2:
            dr2 = get_w(RISK_WEIGHTS, 'R2', seg, use_mc)
            n1 = min(aw, round(aw * dr2))
            n2 = min(ac, round(ac * dr2 * 0.5))  # Active도 일부 이탈
            aw -= n1
            ac -= n2; ch += n2

        # R3: Trial↓ + Active 일부 Churn (대체재로 전환)
        if r3:
            dr3 = get_w(RISK_WEIGHTS, 'R3', seg, use_mc)
            n1 = min(tr, round(tr * dr3))
            n2 = min(ac, round(ac * dr3 * 0.3))
            tr -= n1; ch += n1
            ac -= n2; ch += n2

        result[seg] = {
            'Unaware': max(0,uw), 'Aware': max(0,aw),
            'Trial': max(0,tr), 'Active': max(0,ac),
            'Churn': max(0,ch),
        }
    return result

def run_mc(product, t1,t2,t3,t4, r1,r2,r3, n=1000):
    return [run_sim(product,t1,t2,t3,t4,r1,r2,r3,use_mc=True) for _ in range(n)]

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.markdown("## 애사비 신규 진입 시뮬레이션")
st.caption("합성 데이터 5,000명 | 신제품 기준 (Active=0) | EDA 실측 + Laplace 보정")
st.divider()

with st.sidebar:

    # ── 기저 시장
    st.markdown("### 📦 기저 시장 설정")
    st.caption("어떤 제품이 있는 시장인가?")
    product = st.radio(
        "", list(PRODUCT_WEIGHTS.keys()),
        label_visibility="collapsed"
    )
    st.caption(PRODUCT_DESC[product])

    st.markdown("---")

    # ── 트리거
    st.markdown("### ✅ 트리거")
    st.caption("0=미실행 | 5=최대 강도")
    st.markdown("")

    st.markdown("**T4 · SNS 바이럴**")
    st.caption("Unaware→Aware | 부: Aware→Trial(×0.2)")
    t4 = st.slider("", 0, 5, 0, key="t4")

    st.markdown("**T1 · 효능·성분 콘텐츠**")
    st.caption("Aware→Trial | 부: Unaware→Aware(×0.2), Trial→Active(×0.05)")
    t1 = st.slider("", 0, 5, 0, key="t1")

    st.markdown("**T2 · 무료 샘플·맛 체험**")
    st.caption("Aware→Trial | 부: Unaware→Aware(×0.2), Trial→Active(×0.05)")
    t2 = st.slider("", 0, 5, 0, key="t2")

    st.markdown("**T3 · 가격 할인 쿠폰**")
    st.caption("Aware→Trial | 부: Unaware→Aware(×0.2)")
    t3 = st.slider("", 0, 5, 0, key="t3")

    st.markdown("---")

    # ── 리스크 (ON/OFF)
    st.markdown("### ⚠️ 리스크 (부록)")
    st.caption("발생 시 시뮬레이션 영향 확인")
    r1 = st.checkbox("R1 · 효능 논란", value=False)
    st.caption("Trial↓ + Aware→Trial 억제")
    r2 = st.checkbox("R2 · 경쟁사 저가 진입", value=False)
    st.caption("Aware 이탈 (경쟁사로)")
    r3 = st.checkbox("R3 · 맛·편의 대체재 출시", value=False)
    st.caption("Trial↓ (맛편의형 특히 취약)")

# ── 탭
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 시나리오 분석", "🎲 몬테카를로",
    "🗺️ 취약도 히트맵", "📋 가중치 테이블"
])

# ─────────────────────────────────────────
# TAB 1: 시나리오 분석
# ─────────────────────────────────────────
with tab1:
    res = run_sim(product, t1,t2,t3,t4, r1,r2,r3)

    base_trial  = sum(BASE[s]['Trial']  for s in SEGS)
    new_trial   = sum(res[s]['Trial']   for s in SEGS)
    base_active = sum(BASE[s]['Active'] for s in SEGS)
    new_active  = sum(res[s]['Active']  for s in SEGS)
    base_churn  = sum(BASE[s]['Churn']  for s in SEGS)
    new_churn   = sum(res[s]['Churn']   for s in SEGS)
    base_aware  = sum(BASE[s]['Aware']  for s in SEGS)
    new_aware   = sum(res[s]['Aware']   for s in SEGS)
    new_uw      = sum(res[s]['Unaware'] for s in SEGS)
    base_uw     = sum(BASE[s]['Unaware']for s in SEGS)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Unaware", f"{new_uw:,}명",    f"{new_uw-base_uw:+,}명")
    c2.metric("Aware",   f"{new_aware:,}명",  f"{new_aware-base_aware:+,}명")
    c3.metric("Trial",   f"{new_trial:,}명",  f"{new_trial-base_trial:+,}명",  delta_color="normal")
    c4.metric("Active",  f"{new_active:,}명", f"{new_active-base_active:+,}명",delta_color="normal")
    c5.metric("Churn",   f"{new_churn:,}명",  f"{new_churn-base_churn:+,}명",  delta_color="inverse")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**세그먼트별 Trial 변화**")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='기저', x=SEGS,
            y=[BASE[s]['Trial'] for s in SEGS],
            marker_color='#CBD5E0',
            text=[BASE[s]['Trial'] for s in SEGS],
            textposition='outside',
        ))
        fig.add_trace(go.Bar(
            name='시뮬레이션', x=SEGS,
            y=[res[s]['Trial'] for s in SEGS],
            marker_color=[SEG_COLORS[s] for s in SEGS],
            text=[res[s]['Trial'] for s in SEGS],
            textposition='outside',
        ))
        fig.update_layout(
            barmode='group', height=320,
            margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation='h',y=-0.2),
            plot_bgcolor='white', paper_bgcolor='white',
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**세그먼트 선택 — 전체 State 변화**")
        seg_sel = st.selectbox("", SEGS, label_visibility="collapsed")
        states = ['Unaware','Aware','Trial','Active','Churn']
        sc = ['#ADB5BD','#4C9BE8','#845EF7','#2ECC71','#E74C3C']
        bv = [BASE[seg_sel].get(s,0) for s in states]
        nv = [res[seg_sel].get(s,0) for s in states]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name='기저', x=states, y=bv,
            marker_color='#CBD5E0', text=bv, textposition='outside',
        ))
        fig2.add_trace(go.Bar(
            name='시뮬레이션', x=states, y=nv,
            marker_color=sc, text=nv, textposition='outside',
        ))
        fig2.update_layout(
            barmode='group', height=320,
            margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation='h',y=-0.2),
            plot_bgcolor='white', paper_bgcolor='white',
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**세그먼트 × State 상세**")
    rows = []
    for seg in SEGS:
        b, r = BASE[seg], res[seg]
        rows.append({
            '세그먼트': seg,
            'Unaware': f"{b['Unaware']}→{r['Unaware']} ({r['Unaware']-b['Unaware']:+d})",
            'Aware':   f"{b['Aware']}→{r['Aware']} ({r['Aware']-b['Aware']:+d})",
            'Trial':   f"{b['Trial']}→{r['Trial']} ({r['Trial']-b['Trial']:+d})",
            'Active':  f"{b['Active']}→{r['Active']} ({r['Active']-b['Active']:+d})",
            'Churn':   f"{b['Churn']}→{r['Churn']} ({r['Churn']-b['Churn']:+d})",
        })
    st.dataframe(pd.DataFrame(rows).set_index('세그먼트'), use_container_width=True)

# ─────────────────────────────────────────
# TAB 2: 몬테카를로
# ─────────────────────────────────────────
with tab2:
    st.markdown("**Triangular(±30%) 분포에서 1,000번 랜덤 추출 → 결과 범위**")

    col_mc1, _ = st.columns([1,3])
    with col_mc1:
        n_sim = st.selectbox("반복 횟수", [500,1000,5000], index=1)
        run_btn = st.button("▶ 실행", use_container_width=True)

    if run_btn:
        with st.spinner(f"{n_sim}회 실행 중..."):
            mc_res = run_mc(product,t1,t2,t3,t4,r1,r2,r3,n_sim)

        total_t = np.array([sum(r[s]['Trial'] for s in SEGS) for r in mc_res])

        c1,c2,c3 = st.columns(3)
        c1.metric("P5 (최악)", f"{int(np.percentile(total_t,5))}명")
        c2.metric("P50 (중간)", f"{int(np.percentile(total_t,50))}명")
        c3.metric("P95 (최선)", f"{int(np.percentile(total_t,95))}명")

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=total_t, nbinsx=40, marker_color='#4C9BE8', opacity=0.8,
        ))
        for pct,label,color in [(5,"P5","red"),(50,"P50","navy"),(95,"P95","green")]:
            val = np.percentile(total_t, pct)
            fig_mc.add_vline(x=val, line_dash="dash", line_color=color,
                            annotation_text=f"{label}: {int(val)}명")
        fig_mc.update_layout(
            title="전체 Trial 분포", height=280,
            margin=dict(l=0,r=0,t=40,b=0),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="Trial 인원(명)", yaxis_title="빈도",
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        st.markdown("**세그먼트별 Trial 분포 (P5/P50/P95)**")
        seg_rows = []
        for seg in SEGS:
            arr = np.array([r[seg]['Trial'] for r in mc_res])
            seg_rows.append({
                '세그먼트': seg,
                '기저': BASE[seg]['Trial'],
                'P5': int(np.percentile(arr,5)),
                'P50': int(np.percentile(arr,50)),
                'P95': int(np.percentile(arr,95)),
                'P50 증가': f"+{int(np.percentile(arr,50))-BASE[seg]['Trial']}명",
            })
        st.dataframe(pd.DataFrame(seg_rows).set_index('세그먼트'), use_container_width=True)

# ─────────────────────────────────────────
# TAB 3: 취약도 히트맵
# ─────────────────────────────────────────
with tab3:
    st.markdown("**세그먼트 × 리스크·트리거 취약도/반응도**")
    st.caption("상단 3행: 리스크(클수록 취약) | 하단 4행: 트리거(클수록 반응 좋음)")

    rows_label = list(VULN_DATA.keys())
    z = [[VULN_DATA[r][s] for s in SEGS] for r in rows_label]
    text = [[f"{VULN_DATA[r][s]:.3f}" for s in SEGS] for r in rows_label]

    fig_hm = go.Figure(go.Heatmap(
        z=z, x=SEGS, y=rows_label,
        text=text, texttemplate="%{text}",
        colorscale=[
            [0.0,'#EBF8EE'],[0.25,'#82D99B'],
            [0.55,'#F9B347'],[1.0,'#D9342B'],
        ],
        showscale=True,
        colorbar=dict(title="취약도/반응도"),
    ))
    fig_hm.add_shape(
        type="line",
        x0=-0.5, x1=len(SEGS)-0.5,
        y0=2.5, y1=2.5,
        line=dict(color="black", width=2, dash="dash"),
    )
    fig_hm.add_annotation(
        x=len(SEGS)-0.5, y=1.0,
        text="⚠️ 리스크 (위)",
        showarrow=False, xanchor="right",
        font=dict(size=11, color="#D9342B"),
    )
    fig_hm.add_annotation(
        x=len(SEGS)-0.5, y=4.5,
        text="✅ 트리거 (아래)",
        showarrow=False, xanchor="right",
        font=dict(size=11, color="#2E7D32"),
    )
    fig_hm.update_layout(
        height=420, margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(side='top'),
        yaxis=dict(autorange='reversed'),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(size=12),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.error("🩸 **혈당관리 실천형**\n\nR1·R2 모두 취약 1위\nT3 가격할인에만 강하게 반응\n→ 가격할인 = 유입 + 이탈 위험 동시")
    with col_i2:
        st.success("💪 **건강관심 잠재형**\n\n리스크 취약도 낮음\nT1 효능 콘텐츠 반응 최고\n→ 가장 안정적인 타겟")
    with col_i3:
        st.warning("🍭 **맛 편의 추구형**\n\nR3 대체재 취약도 압도적\nT2 샘플 체험 반응 최고\n→ 샘플 유입, 대체재에 즉각 이탈")

# ─────────────────────────────────────────
# TAB 4: 가중치 테이블
# ─────────────────────────────────────────
with tab4:
    st.markdown("**기저 시장 — 제품별 Aware→Trial 추가 전환율 (mode)**")
    prod_rows = []
    for prod in PRODUCT_WEIGHTS:
        row = {'제품': prod}
        for seg in SEGS:
            row[seg] = f"{PRODUCT_WEIGHTS[prod][seg][1]:.3f}"
        prod_rows.append(row)
    st.dataframe(pd.DataFrame(prod_rows).set_index('제품'), use_container_width=True)
    st.caption("근거: 비경험자 설문 선호 형태 비율 | 분말스틱=스틱×0.8 | 음료=액상보틀×1.3 | Laplace 적용 컬럼 있음")

    st.markdown("---")
    st.markdown("**트리거 가중치 (mode)**")
    trig_rows = []
    for seg in SEGS:
        trig_rows.append({
            '세그먼트': seg,
            'T1 효능': f"{TRIGGER_WEIGHTS['T1'][seg][1]:.3f}",
            'T2 샘플': f"{TRIGGER_WEIGHTS['T2'][seg][1]:.3f}",
            'T3 가격': f"{TRIGGER_WEIGHTS['T3'][seg][1]:.3f}",
            'T4 바이럴': f"{TRIGGER_WEIGHTS['T4'][seg][1]:.3f}",
        })
    st.dataframe(pd.DataFrame(trig_rows).set_index('세그먼트'), use_container_width=True)

    st.markdown("---")
    st.markdown("**리스크 가중치 (mode)**")
    risk_rows = []
    for seg in SEGS:
        risk_rows.append({
            '세그먼트': seg,
            'R1 효능논란': f"{RISK_WEIGHTS['R1'][seg][1]:.3f}",
            'R2 경쟁사':   f"{RISK_WEIGHTS['R2'][seg][1]:.3f}",
            'R3 대체재':   f"{RISK_WEIGHTS['R3'][seg][1]:.3f}",
        })
    st.dataframe(pd.DataFrame(risk_rows).set_index('세그먼트'), use_container_width=True)

st.divider()
st.caption("기저시장(제품유형) → 트리거(마케팅) → 리스크(외부충격) 순서로 적용 | 주효과·부효과 구조 | Triangular ±30%")
