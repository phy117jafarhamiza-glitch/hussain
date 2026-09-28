# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from db import init_db, add_subject, list_subjects, delete_subject
from utils.constants import GROUPS

st.set_page_config(page_title="الحالات والمجموعات", page_icon="📋", layout="wide")
init_db()
st.markdown('<style>html,body,[class*="css"]{direction:rtl;text-align:right;}</style>', unsafe_allow_html=True)

st.title("📋 إدارة اللاعبين والمجموعات")
st.caption("تسجيل عينة البحث: المجموعة التجريبية والمجموعة الضابطة (لاعبو الفنون القتالية)")

with st.form("add_subject_form", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns(4)
    name = c1.text_input("اسم اللاعب")
    group = c2.selectbox("المجموعة", GROUPS)
    age = c3.number_input("العمر", min_value=10, max_value=60, value=22)
    sport = c4.text_input("الرياضة القتالية", placeholder="مصارعة / جودو / تايكوندو / MMA")
    injury_note = st.text_area("ملاحظات الإصابة (نوع/شدة إصابة الرقبة)")
    submitted = st.form_submit_button("إضافة لاعب")
    if submitted:
        if not name.strip():
            st.error("يرجى إدخال اسم اللاعب.")
        else:
            add_subject(name.strip(), group, int(age), sport.strip(), injury_note.strip())
            st.success(f"تمت إضافة اللاعب: {name}")
            st.rerun()

st.markdown("---")
st.subheader("قائمة اللاعبين المسجلين")

subjects = list_subjects()
if not subjects:
    st.info("لا يوجد لاعبون مسجلون بعد. أضف أول لاعب من النموذج أعلاه.")
else:
    df = pd.DataFrame(subjects)
    tab1, tab2 = st.tabs(["عرض كجدول", "عرض حسب المجموعة"])
    with tab1:
        st.dataframe(df, use_container_width=True)
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### المجموعة التجريبية")
            exp_df = df[df["group_name"] == "تجريبية"]
            st.dataframe(exp_df, use_container_width=True)
            st.metric("عدد أفراد المجموعة التجريبية", len(exp_df))
        with col2:
            st.markdown("### المجموعة الضابطة")
            ctrl_df = df[df["group_name"] == "ضابطة"]
            st.dataframe(ctrl_df, use_container_width=True)
            st.metric("عدد أفراد المجموعة الضابطة", len(ctrl_df))

    st.markdown("---")
    with st.expander("حذف لاعب"):
        to_delete = st.selectbox(
            "اختر اللاعب المراد حذفه",
            options=[(s["id"], s["name"]) for s in subjects],
            format_func=lambda x: x[1],
        )
        if st.button("حذف نهائي", type="primary"):
            delete_subject(to_delete[0])
            st.success("تم الحذف.")
            st.rerun()
