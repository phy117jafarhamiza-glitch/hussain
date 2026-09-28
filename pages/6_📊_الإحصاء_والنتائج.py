# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from db import init_db, get_measurements
from utils.constants import TEST_NAMES, SESSION_STAGES
from utils.stats import build_comparison_table

st.set_page_config(page_title="الإحصاء والنتائج", page_icon="📊", layout="wide")
init_db()
st.markdown('<style>html,body,[class*="css"]{direction:rtl;text-align:right;}</style>', unsafe_allow_html=True)

st.title("📊 الإحصاء الجاهز للفصل الرابع (نتائج البحث)")
st.caption(
    "اختبار (t) للعينات المترابطة (قبلي/بعدي داخل كل مجموعة) واختبار (t) "
    "للعينات المستقلة (تجريبية مقابل ضابطة)، مع حجم الأثر (Cohen's d)."
)

col1, col2, col3 = st.columns(3)
test_key = col1.selectbox("المتغير المراد تحليله", list(TEST_NAMES.keys()), format_func=lambda k: TEST_NAMES[k])
pre_stage = col2.selectbox("مرحلة القياس القبلي", SESSION_STAGES, index=0)
post_stage = col3.selectbox("مرحلة القياس البعدي", SESSION_STAGES, index=len(SESSION_STAGES) - 1)

all_measurements = get_measurements()
if not all_measurements:
    st.info("لا توجد بيانات مسجلة بعد. أدخل نتائج الاختبارات من صفحة (الاختبارات القبلية والبعدية).")
    st.stop()

result = build_comparison_table(all_measurements, test_key, pre_stage, post_stage)


def show_ttest_block(title, res, group_names=None):
    st.markdown(f"#### {title}")
    if res.get("t") is None:
        st.caption("بيانات غير كافية لحساب الاختبار (يلزم فردان على الأقل في كل مجموعة).")
        return
    if "pre" in res:  # paired
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("المتوسط القبلي", f"{res['pre']['mean']:.2f}" if res['pre']['mean'] is not None else "-")
        c2.metric("المتوسط البعدي", f"{res['post']['mean']:.2f}" if res['post']['mean'] is not None else "-")
        c3.metric("قيمة t", f"{res['t']:.2f}")
        c4.metric("قيمة p", f"{res['p']:.4f}")
    else:  # independent
        c1, c2, c3, c4 = st.columns(4)
        a_name, b_name = group_names or ("المجموعة أ", "المجموعة ب")
        c1.metric(f"متوسط {a_name}", f"{res['a']['mean']:.2f}" if res['a']['mean'] is not None else "-")
        c2.metric(f"متوسط {b_name}", f"{res['b']['mean']:.2f}" if res['b']['mean'] is not None else "-")
        c3.metric("قيمة t", f"{res['t']:.2f}")
        c4.metric("قيمة p", f"{res['p']:.4f}")

    st.write(f"حجم الأثر (Cohen's d): **{res['d']:.2f}**")
    if res["significant"]:
        st.success("الفرق دالّ إحصائياً عند مستوى 0.05 ✅")
    else:
        st.warning("الفرق غير دالّ إحصائياً عند مستوى 0.05")


tab1, tab2 = st.tabs(["📈 داخل كل مجموعة (قبلي/بعدي)", "⚖️ بين المجموعتين"])
with tab1:
    show_ttest_block("المجموعة التجريبية (قبلي مقابل بعدي)", result["experimental_within"])
    st.markdown("---")
    show_ttest_block("المجموعة الضابطة (قبلي مقابل بعدي)", result["control_within"])

with tab2:
    show_ttest_block("الفروق البعدية بين المجموعتين", result["post_between_groups"], ("تجريبية", "ضابطة"))
    st.markdown("---")
    show_ttest_block("الفروق القبلية بين المجموعتين (تكافؤ العينة)", result["pre_between_groups"], ("تجريبية", "ضابطة"))

st.markdown("---")
st.subheader("رسم بياني مقارن")
raw = result["raw"]
chart_df = pd.DataFrame({
    "تجريبية - قبلي": pd.Series(raw["exp_pre"]),
    "تجريبية - بعدي": pd.Series(raw["exp_post"]),
    "ضابطة - قبلي": pd.Series(raw["ctrl_pre"]),
    "ضابطة - بعدي": pd.Series(raw["ctrl_post"]),
})
means = chart_df.mean().rename(TEST_NAMES[test_key])
st.bar_chart(means)

with st.expander("عرض البيانات الخام المستخدمة في التحليل"):
    st.dataframe(chart_df, use_container_width=True)
