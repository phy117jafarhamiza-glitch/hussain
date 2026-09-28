# -*- coding: utf-8 -*-
"""
محاكاة "وحدة التحكم الذكية" (AI Controller) للجهاز الذكي المتعدد الاتجاهات
الموصوف في الفصل الثالث من الأطروحة (Smart Multidirectional Neck Trainer).

حسب الوصف: الجهاز يلتقط بيانات IMU (اتجاه/زاوية حركة الرأس) و EMG (شدة
تنشيط العضلة) لحظياً، ويحللها بخوارزميات ذكاء اصطناعي ليولّد مقاومة تفاعلية
موجهة تتغير حسب قدرة العضلة ووضعية الرأس وسرعة الحركة.

هذا الملف يبني شبكة عصبية (Neural Network - MLPRegressor) تتعلم العلاقة:
    (اتجاه الحركة، شدة تنشيط EMG، زاوية الرأس، السرعة الزاوية)
        -> مستوى المقاومة الموصى به (قيمة مستمرة 0-100%)
ثم تُصنَّف القيمة الى (منخفضة/متوسطة/عالية) لعرضها على المدرب/الرياضي.

بما أن بيانات الجهاز الفعلي غير متوفرة بعد (سيُشترى لاحقاً وفق نص الأطروحة)،
يُهيَّأ النموذج ببيانات تركيبية تعكس منطق تأهيلي منطقي وآمن (مقاومة أعلى مع
نشاط عضلي جيد وزاوية آمنة، ومقاومة أقل عند ضعف النشاط العضلي أو اقتراب
الزاوية من الحدود القصوى لحماية الرياضي المصاب)، ويمكن العلاج لاحقاً بإعادة
تدريبه على بيانات حقيقية من الجهاز أو من سجلات الجلسات المدخلة يدوياً في
هذا التطبيق (زر "إعادة تدريب النموذج على بيانات اللاعبين المسجلة").
"""
import os
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from utils.constants import NECK_DIRECTIONS, RESISTANCE_LEVELS

MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "ai_controller_mlp.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

# الحد الآمن الأقصى لزاوية الرأس بالدرجة (توقف/تخفيف المقاومة قربه لحماية المصاب)
SAFE_MAX_ANGLE = 45.0


def _direction_index(direction_label: str) -> int:
    return NECK_DIRECTIONS.index(direction_label) if direction_label in NECK_DIRECTIONS else 0


def _synthetic_controller_data(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    n_dirs = len(NECK_DIRECTIONS)
    dir_idx = rng.integers(0, n_dirs, n)
    emg = rng.uniform(0, 100, n)          # % من أقصى تنشيط عضلي (MVC)
    angle = rng.uniform(0, 60, n)          # درجة انحراف الرأس عن المحايد
    velocity = rng.uniform(0, 120, n)      # سرعة زاوية الحركة (درجة/ثانية)

    # منطق تأهيلي مبسّط: المقاومة ترتفع مع قوة العضلة وسرعة تحكم معتدلة،
    # وتنخفض بشدة كلما اقترب الرأس من الحد الآمن الأقصى (حماية المصاب)،
    # وتنخفض أيضاً عند سرعة حركة عالية جداً (خطر عدم التحكم).
    safety_factor = np.clip(1 - (angle / SAFE_MAX_ANGLE) ** 2, 0.05, 1.0)
    velocity_factor = np.clip(1 - np.abs(velocity - 40) / 120, 0.2, 1.0)
    resistance = (emg * 0.7) * safety_factor * velocity_factor
    resistance += rng.normal(0, 3, n)
    resistance = np.clip(resistance, 0, 100)

    X = np.column_stack([dir_idx, emg, angle, velocity])
    y = resistance
    return X, y


def _train():
    X, y = _synthetic_controller_data()
    scaler = StandardScaler().fit(X)
    model = MLPRegressor(hidden_layer_sizes=(24, 12), max_iter=3000, random_state=7)
    model.fit(scaler.transform(X), y)
    joblib.dump({"scaler": scaler, "model": model}, MODEL_PATH)
    return scaler, model


def _load_or_train():
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        return bundle["scaler"], bundle["model"]
    return _train()


def retrain_from_logs(logs: list):
    """يعيد تدريب النموذج باستخدام سجلات فعلية من قاعدة البيانات
    (controller_logs) بمجرد توفر عدد كافٍ منها، مع دمجها مع البيانات
    التركيبية للحفاظ على الاستقرار عند قلة العينات."""
    X_synth, y_synth = _synthetic_controller_data(n=500)
    if logs:
        X_real = np.array([[
            _direction_index(l["direction"]), l["emg_activation"],
            l["head_angle"], l["angular_velocity"],
        ] for l in logs])
        y_real = np.array([l["recommended_resistance_value"] for l in logs])
        X = np.vstack([X_synth, X_real])
        y = np.concatenate([y_synth, y_real])
    else:
        X, y = X_synth, y_synth
    scaler = StandardScaler().fit(X)
    model = MLPRegressor(hidden_layer_sizes=(24, 12), max_iter=3000, random_state=7)
    model.fit(scaler.transform(X), y)
    joblib.dump({"scaler": scaler, "model": model}, MODEL_PATH)
    return scaler, model


def _level_from_value(value: float) -> str:
    if value < 33:
        return RESISTANCE_LEVELS[0]
    if value < 66:
        return RESISTANCE_LEVELS[1]
    return RESISTANCE_LEVELS[2]


class AIController:
    def __init__(self):
        self.scaler, self.model = _load_or_train()

    def recommend(self, direction: str, emg_activation: float, head_angle: float, angular_velocity: float):
        x = np.array([[_direction_index(direction), emg_activation, head_angle, angular_velocity]])
        value = float(np.clip(self.model.predict(self.scaler.transform(x))[0], 0, 100))
        level = _level_from_value(value)

        warning = None
        if head_angle >= SAFE_MAX_ANGLE * 0.9:
            warning = "تحذير: زاوية الرأس قريبة من الحد الآمن الأقصى — يوصى بتخفيف المقاومة ومراقبة الرياضي."
        return {
            "direction": direction,
            "value": round(value, 1),
            "level": level,
            "warning": warning,
        }

    def refresh(self, logs=None):
        self.scaler, self.model = retrain_from_logs(logs or [])


_controller_singleton = None


def get_controller():
    global _controller_singleton
    if _controller_singleton is None:
        _controller_singleton = AIController()
    return _controller_singleton
