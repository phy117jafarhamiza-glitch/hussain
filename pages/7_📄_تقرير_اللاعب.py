# -*- coding: utf-8 -*-
import streamlit as st

from db import init_db, list_subjects, get_measurements, list_emg_analyses, list_controller_logs
from utils.report import generate_subject_report

st.set_page_config(page_title="تقرير اللاعب", page_icon="📄", layout="wide")
init_db()
st.markdown('<style>html,body,[class*="css"]{direction:rtl;text-align:right;}</style>', unsafe_allow_html=True)

st.title("📄 تقرير تطور اللاعب (Word)")
st.caption("تقرير شامل يجمع نتائج الاختبارات، تحليل EMG، وسجل توصيات الجهاز الذكي.")

subjects = list_subjects()
if not subjects:
    st.warning("لا يوجد لاعبون مسجلون بعد.")
    st.stop()

subj_map = {f'{s["name"]} ({s["group_name"]})': s["id"] for s in subjects}
subject_label = st.selectbox("اختر اللاعب", list(subj_map.keys()))
subject_id = subj_map[subject_label]
subject = next(s for s in subjects if s["id"] == subject_id)

measurements = get_measurements(subject_id=subject_id)
emg_analyses = list_emg_analyses(subject_id=subject_id)
controller_logs = list_controller_logs(subject_id=subject_id)

st.write(f"عدد نتائج الاختبارات المسجلة: **{len(measurements)}**")
st.write(f"عدد تحليلات EMG المسجلة: **{len(emg_analyses)}**")
st.write(f"عدد سجلات توصيات الجهاز الذكي: **{len(controller_logs)}**")

if st.button("📝 توليد التقرير", type="primary"):
    path = generate_subject_report(subject, measurements, emg_analyses, controller_logs)
    with open(path, "rb") as f:
        st.download_button(
            "⬇️ تحميل التقرير (Word)",
            data=f.read(),
            file_name=path.split("/")[-1],
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    st.success("تم توليد التقرير بنجاح.")
