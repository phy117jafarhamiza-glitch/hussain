# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from db import init_db, list_subjects, add_controller_log, list_controller_logs
from ml.ai_controller import get_controller, RESISTANCE_LEVELS
from utils.constants import NECK_DIRECTIONS

st.set_page_config(page_title="محاكاة الجهاز الذكي", page_icon="🤖", layout="wide")
init_db()
st.markdown('<style>html,body,[class*="css"]{direction:rtl;text-align:right;}</style>', unsafe_allow_html=True)

st.title("🤖 محاكاة وحدة التحكم الذكية (AI Controller)")
st.caption(
    "يحاكي هذا القسم منطق \"الجهاز الذكي متعدد الاتجاهات لتقوية عضلات الرقبة\" "
    "الموصوف في الفصل الثالث: يلتقط بيانات EMG واتجاه/زاوية/سرعة حركة الرأس "
    "(IMU) ويحسب مستوى المقاومة التفاعلية الموصى بها عبر شبكة عصبية."
)

controller = get_controller()

subjects = list_subjects()
subj_map = {f'{s["name"]} ({s["group_name"]})': s["id"] for s in subjects} if subjects else {}

with st.form("controller_form"):
    c1, c2 = st.columns(2)
    subject_label = c1.selectbox("اللاعب (اختياري)", ["-- بدون ربط --"] + list(subj_map.keys()))
    direction = c2.selectbox("اتجاه الحركة", NECK_DIRECTIONS)

    c3, c4, c5 = st.columns(3)
    emg = c3.slider("شدة تنشيط العضلة EMG (% من أقصى انقباض إرادي)", 0, 100, 50)
    angle = c4.slider("زاوية انحراف الرأس عن المحايد (درجة)", 0, 70, 15)
    velocity = c5.slider("السرعة الزاوية للحركة (درجة/ثانية)", 0, 150, 40)

    submitted = st.form_submit_button("🔍 احصل على توصية المقاومة")

if submitted:
    rec = controller.recommend(direction, emg, angle, velocity)
    st.metric("مستوى المقاومة الموصى به", f"{rec['level']} ({rec['value']}%)")
    st.progress(rec["value"] / 100.0)
    if rec["warning"]:
        st.warning(rec["warning"])

    subject_id = subj_map.get(subject_label)
    add_controller_log(subject_id, direction, emg, angle, velocity, rec["level"], rec["value"])
    st.caption("تم تسجيل هذا الاستدعاء في سجل التوصيات أدناه.")

st.markdown("---")
st.subheader("🔁 إعادة تدريب النموذج على البيانات المسجلة فعلياً")
st.write(
    "بمجرد تجميع عدد كافٍ من سجلات الجلسات الحقيقية (من الجهاز الفعلي عند "
    "شرائه، أو من إدخالات هذا التطبيق)، يمكن إعادة تدريب الشبكة العصبية "
    "لتحسين دقة توصياتها."
)
if st.button("إعادة التدريب الآن"):
    logs = list_controller_logs(limit=5000)
    controller.refresh(logs)
    st.success(f"تمت إعادة تدريب النموذج باستخدام {len(logs)} سجلاً فعلياً (مدمجة مع بيانات تأسيسية).")

st.markdown("---")
st.subheader("سجل التوصيات السابقة")
logs = list_controller_logs(limit=100)
if logs:
    st.dataframe(pd.DataFrame(logs)[
        ["id", "subject_name", "direction", "emg_activation", "head_angle",
         "angular_velocity", "recommended_resistance", "recommended_resistance_value", "created_at"]
    ], use_container_width=True)
else:
    st.caption("لا توجد سجلات بعد.")
