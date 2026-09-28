# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from db import init_db, list_subjects, upsert_measurement, get_measurements
from utils.constants import SESSION_STAGES, TEST_NAMES

st.set_page_config(page_title="الاختبارات", page_icon="🧪", layout="wide")
init_db()
st.markdown('<style>html,body,[class*="css"]{direction:rtl;text-align:right;}</style>', unsafe_allow_html=True)

st.title("🧪 تسجيل نتائج الاختبارات")
st.caption("جميع الاختبارات المذكورة في الفصل الثالث: VAS، مدى الحركة، الداينوميتر، التحكم العصبي العضلي، EMG")

subjects = list_subjects()
if not subjects:
    st.warning("أضف لاعبين أولاً من صفحة (الحالات والمجموعات).")
    st.stop()

subj_map = {f'{s["name"]} ({s["group_name"]})': s["id"] for s in subjects}
c1, c2 = st.columns(2)
subject_label = c1.selectbox("اللاعب", list(subj_map.keys()))
subject_id = subj_map[subject_label]
stage = c2.selectbox("مرحلة القياس", SESSION_STAGES)

st.markdown("### إدخال القيم")
with st.form("tests_form"):
    values = {}
    cols = st.columns(3)
    for i, (key, label) in enumerate(TEST_NAMES.items()):
        with cols[i % 3]:
            values[key] = st.number_input(label, value=0.0, step=0.1, key=f"{key}_{subject_id}_{stage}")
    notes = st.text_area("ملاحظات عامة على الجلسة")
    submitted = st.form_submit_button("حفظ نتائج هذه المرحلة")
    if submitted:
        for key, val in values.items():
            upsert_measurement(subject_id, stage, key, val, notes if key == "vas" else None)
        st.success("تم حفظ النتائج.")

st.markdown("---")
st.subheader("سجل نتائج اللاعب المختار")
records = get_measurements(subject_id=subject_id)
if records:
    df = pd.DataFrame(records)
    df["الاختبار"] = df["test_key"].map(TEST_NAMES)
    pivot = df.pivot_table(index="الاختبار", columns="stage", values="value", aggfunc="first")
    # إعادة ترتيب الأعمدة حسب تسلسل المراحل الزمني
    ordered_cols = [s for s in SESSION_STAGES if s in pivot.columns]
    st.dataframe(pivot[ordered_cols], use_container_width=True)
else:
    st.info("لا توجد نتائج مسجلة لهذا اللاعب بعد.")
