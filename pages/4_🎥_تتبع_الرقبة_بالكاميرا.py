# -*- coding: utf-8 -*-
import numpy as np
import streamlit as st
from PIL import Image

from db import init_db, list_subjects, add_pose_rom_session
from ml.neck_pose import PoseEstimator, rom_from_neutral_and_peak
from utils.constants import SESSION_STAGES

st.set_page_config(page_title="تتبع الرقبة بالكاميرا", page_icon="🎥", layout="wide")
init_db()
st.markdown('<style>html,body,[class*="css"]{direction:rtl;text-align:right;}</style>', unsafe_allow_html=True)

st.title("🎥 تتبع مدى حركة الرقبة (ROM) بالكاميرا")
st.caption(
    "بديل عملي لمقياس الزوايا الرقمي (Digital Inclinometer) عند عدم توفر الجهاز: "
    "التقط صورة للوضعية المحايدة ثم صورة لذروة الحركة، وسيحسب النظام الفرق الزاوي "
    "بنفس منطق تصفير الجهاز المستخدم في الأطروحة."
)

st.warning(
    "📌 ملاحظة منهجية: الثني الأمامي/الانبساط الخلفي يحتاجان صورة **جانبية** "
    "(profile view)، بينما الميل الجانبي والدوران يمكن تقديرهما من صورة **أمامية**."
)

DIRECTION_OPTIONS = {
    "ثني أمامي (Flexion) - يتطلب صورة جانبية": "flexion",
    "انبساط خلفي (Extension) - يتطلب صورة جانبية": "extension",
    "ميل جانبي أيمن - صورة أمامية": "lateral_r",
    "ميل جانبي أيسر - صورة أمامية": "lateral_l",
    "دوران أيمن - صورة أمامية": "rotation_r",
    "دوران أيسر - صورة أمامية": "rotation_l",
}

subjects = list_subjects()
subj_map = {f'{s["name"]} ({s["group_name"]})': s["id"] for s in subjects} if subjects else {}

c1, c2, c3 = st.columns(3)
subject_label = c1.selectbox("اللاعب", list(subj_map.keys()) or ["-- لا يوجد لاعبون --"])
stage = c2.selectbox("مرحلة القياس", SESSION_STAGES)
direction_label = c3.selectbox("اتجاه الحركة المقاس", list(DIRECTION_OPTIONS.keys()))
direction = DIRECTION_OPTIONS[direction_label]


@st.cache_resource(show_spinner="تحميل نموذج تتبع الجسم (أول مرة فقط)...")
def load_estimator():
    return PoseEstimator()


def read_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)


col_a, col_b = st.columns(2)
with col_a:
    st.subheader("١) الوضعية المحايدة (Neutral)")
    neutral_img = st.camera_input("التقط صورة الوضعية المحايدة", key="neutral_cam")
    neutral_upload = st.file_uploader("أو ارفع صورة الوضعية المحايدة", type=["jpg", "jpeg", "png"], key="neutral_up")

with col_b:
    st.subheader("٢) وضعية ذروة الحركة (Peak)")
    peak_img = st.camera_input("التقط صورة ذروة الحركة", key="peak_cam")
    peak_upload = st.file_uploader("أو ارفع صورة ذروة الحركة", type=["jpg", "jpeg", "png"], key="peak_up")

neutral_file = neutral_img or neutral_upload
peak_file = peak_img or peak_upload

if st.button("🧮 احسب مدى الحركة (ROM)", type="primary"):
    if neutral_file is None or peak_file is None:
        st.error("يرجى توفير صورتي الوضعية المحايدة والذروة معاً.")
    else:
        try:
            estimator = load_estimator()
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        neutral_landmarks = estimator.estimate(read_image(neutral_file))
        peak_landmarks = estimator.estimate(read_image(peak_file))

        if neutral_landmarks is None or peak_landmarks is None:
            st.error("تعذّر اكتشاف الجسم في إحدى الصورتين. تأكد من وضوح الرأس والكتفين في الإطار.")
        else:
            rom_deg = rom_from_neutral_and_peak(neutral_landmarks, peak_landmarks, direction)
            st.session_state["last_rom_result"] = {"direction": direction, "value": rom_deg}
            st.success(f"مدى الحركة المقدّر ({direction_label.split(' - ')[0]}): **{rom_deg:.1f}°**")

if "last_rom_result" in st.session_state and subj_map:
    result = st.session_state["last_rom_result"]
    st.markdown("---")
    st.write(f"آخر نتيجة محسوبة: **{result['value']:.1f}°** ({result['direction']})")
    if st.button("💾 حفظ آخر قياس في سجل اللاعب"):
        add_pose_rom_session(subj_map[subject_label], stage, "camera", {result["direction"]: result["value"]})
        st.success("تم الحفظ في سجل اللاعب.")
        del st.session_state["last_rom_result"]
