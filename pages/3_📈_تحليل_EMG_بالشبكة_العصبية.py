# -*- coding: utf-8 -*-
import io
import numpy as np
import pandas as pd
import streamlit as st

from db import init_db, list_subjects, add_emg_analysis, list_emg_analyses
from ml.signal_processing import full_pipeline
from ml.emg_cnn import get_classifier
from utils.constants import SESSION_STAGES, EMG_MUSCLES

st.set_page_config(page_title="تحليل EMG بالشبكة العصبية", page_icon="📈", layout="wide")
init_db()
st.markdown('<style>html,body,[class*="css"]{direction:rtl;text-align:right;}</style>', unsafe_allow_html=True)

st.title("📈 تحليل إشارة EMG بالشبكة العصبية الالتفافية (1D-CNN)")
st.caption(
    "يرفع الباحث ملف إشارة EMG الخام (CSV بعمود واحد للقيم، مأخوذ من جهاز تخطيط "
    "كهربية العضلات De Luca 1997)، ويقوم النموذج بترشيحها واستخلاص النوافذ "
    "الزمنية وتصنيف نمط التنشيط العضلي."
)

classifier = get_classifier()
st.info(f"العتبة الحسابية الحالية للنموذج: **{classifier.backend}**")

subjects = list_subjects()
subj_map = {f'{s["name"]} ({s["group_name"]})': s["id"] for s in subjects} if subjects else {}

col1, col2, col3 = st.columns(3)
subject_label = col1.selectbox("اللاعب (اختياري)", ["-- بدون ربط --"] + list(subj_map.keys()))
stage = col2.selectbox("مرحلة القياس", SESSION_STAGES)
muscle = col3.selectbox("العضلة", EMG_MUSCLES)

fs = st.number_input("معدل أخذ العينات (Hz)", min_value=100, max_value=5000, value=1000, step=50)

uploaded = st.file_uploader("ملف إشارة EMG (CSV - عمود قيم واحد أو أكثر)", type=["csv", "txt"])

use_demo = st.checkbox("استخدام إشارة تجريبية (Demo) لاختبار النظام بدون رفع ملف")

signal = None
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded, header=None)
        signal = raw_df.iloc[:, 0].astype(float).to_numpy()
        st.success(f"تم تحميل الإشارة: {len(signal)} عينة")
    except Exception as e:
        st.error(f"تعذّرت قراءة الملف: {e}")
elif use_demo:
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4, int(fs * 4))
    signal = 0.05 * rng.standard_normal(len(t))
    for center in [0.6, 1.6, 2.6, 3.4]:
        burst = np.exp(-((t - center) ** 2) / (2 * 0.08 ** 2))
        signal += burst * (0.5 * np.sin(2 * np.pi * 90 * t))
    st.info("تم توليد إشارة EMG تجريبية (Demo) لأغراض العرض والاختبار.")

if signal is not None:
    with st.spinner("جارٍ معالجة الإشارة وتشغيل النموذج..."):
        out = full_pipeline(signal, fs)
        results, majority = classifier.predict_batch(out["segments"])

    tab1, tab2 = st.tabs(["📊 الرسوم البيانية", "🧠 نتيجة التصنيف"])
    with tab1:
        st.line_chart(pd.DataFrame({"الإشارة المرشّحة": out["filtered"]}))
        st.line_chart(pd.DataFrame({"غلاف RMS": out["envelope"]}))

    with tab2:
        st.metric("التصنيف الغالب لهذه الجلسة", majority)
        res_df = pd.DataFrame(results)
        st.dataframe(res_df[["label", "confidence"]], use_container_width=True)

        avg_conf = res_df["confidence"].mean()
        st.progress(min(1.0, float(avg_conf)), text=f"متوسط ثقة النموذج: {avg_conf:.2f}")

        if st.button("💾 حفظ نتيجة هذه الجلسة في سجل اللاعب"):
            subject_id = subj_map.get(subject_label)
            avg_features = {
                k: float(np.mean([r["features"][k] for r in results]))
                for k in ["rms", "mav", "zc", "wl"]
            }
            add_emg_analysis(
                subject_id=subject_id, stage=stage, muscle=muscle,
                source_file=getattr(uploaded, "name", "demo_signal"),
                predicted_label=majority, confidence=float(avg_conf),
                features=avg_features, model_backend=classifier.backend,
            )
            st.success("تم الحفظ في سجل تحليلات EMG.")
else:
    st.info("ارفع ملف إشارة EMG أو فعّل خيار الإشارة التجريبية لبدء التحليل.")

st.markdown("---")
st.subheader("سجل تحليلات EMG المحفوظة")
analyses = list_emg_analyses()
if analyses:
    st.dataframe(pd.DataFrame(analyses)[
        ["id", "subject_name", "stage", "muscle", "predicted_label", "confidence", "model_backend", "created_at"]
    ], use_container_width=True)
else:
    st.caption("لا توجد تحليلات محفوظة بعد.")
