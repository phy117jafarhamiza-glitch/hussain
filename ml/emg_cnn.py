# -*- coding: utf-8 -*-
"""
نموذج الشبكة العصبية الالتفافية (1D-CNN) لتصنيف نمط تنشيط عضلات الرقبة من
إشارة EMG، تطبيقاً لما ورد في الفصل الثاني من الأطروحة:

  "تستخدم نماذج CNN (Convolutional Neural Networks) في الكشف عن التمزقات
   العضلية... والتشخيص التفريقي ... باستخدام بيانات تخطيط العضلات (EMG)
   والتحليل الذكي للأنماط" (Litjens et al., 2017؛ Foley et al., 2021)

هذا الملف يوفر مسارين للتنفيذ:
  1) TensorFlow/Keras Conv1D  -> يُستخدم تلقائياً اذا كانت المكتبة مثبتة
     (وهو الخيار المستخدم عند نشر التطبيق فعلياً / إلحاقه بالأطروحة).
  2) شبكة عصبية احتياطية (MLP من scikit-learn) على ميزات EMG القياسية،
     تعمل فوراً بدون تثبيت مكتبات ثقيلة، لضمان عمل التطبيق في أي بيئة.

المخرجات في الحالتين: تصنيف نافذة الإشارة الى واحدة من ثلاث فئات:
  "ضعيف/غير متوازن" | "طبيعي" | "مرتفع/إجهاد محتمل"
بالإضافة الى درجة الثقة (confidence) لكل تنبؤ.
"""
import os
import numpy as np

from utils.constants import EMG_QUALITY_LABELS
from ml.signal_processing import features_matrix, raw_windows_matrix

MODEL_DIR = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    import tensorflow as tf  # noqa: F401
    TF_AVAILABLE = True
except Exception:  # pragma: no cover
    TF_AVAILABLE = False

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib


# --------------------------------------------------------------------------
# توليد بيانات تدريب تركيبية (Synthetic) عند عدم توفر بيانات EMG حقيقية موسومة
# بما يوافق الأنماط الفيزيولوجية الموصوفة في الأطروحة: نشاط ضعيف (إصابة/ضعف
# العضلات العميقة)، نشاط طبيعي متوازن، نشاط مرتفع (إجهاد/تعويض بالعضلات
# السطحية). تُستخدم فقط لتهيئة نموذج أولي قابل لإعادة التدريب على بيانات
# حقيقية بمجرد توفرها من الجهاز الذكي أو جهاز EMG المختبري.
# --------------------------------------------------------------------------
def _synthetic_training_set(n_per_class=120, fs=1000, window_s=0.5, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, window_s, int(fs * window_s), endpoint=False)
    X_raw, X_feat, y = [], [], []
    profiles = {
        0: dict(amp=0.15, noise=0.08, freq=60),   # ضعيف / غير متوازن
        1: dict(amp=0.45, noise=0.05, freq=90),   # طبيعي
        2: dict(amp=0.85, noise=0.12, freq=140),  # مرتفع / إجهاد
    }
    for label, p in profiles.items():
        for _ in range(n_per_class):
            sig = p["amp"] * np.sin(2 * np.pi * p["freq"] * t + rng.uniform(0, np.pi))
            sig += p["noise"] * rng.standard_normal(len(t))
            sig *= (1 + 0.3 * rng.standard_normal())  # تباين فردي بين الرياضيين
            X_raw.append(sig)
            y.append(label)
    X_raw = np.array(X_raw)
    feats, _ = features_matrix(list(X_raw))
    return X_raw, feats, np.array(y)


# --------------------------------------------------------------------------
# المسار الأول: Keras Conv1D (يُستخدم عند توفر tensorflow)
# --------------------------------------------------------------------------
def _build_keras_model(input_len, n_classes=3):
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Conv1D(16, kernel_size=7, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, kernel_size=5, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _train_keras(epochs=15):
    X_raw, _, y = _synthetic_training_set()
    X = X_raw[..., np.newaxis]
    model = _build_keras_model(X.shape[1])
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0, validation_split=0.2)
    model.save(os.path.join(MODEL_DIR, "emg_cnn.keras"))
    return model


def _load_or_train_keras():
    path = os.path.join(MODEL_DIR, "emg_cnn.keras")
    from tensorflow.keras.models import load_model
    if os.path.exists(path):
        return load_model(path)
    return _train_keras()


# --------------------------------------------------------------------------
# المسار الاحتياطي: MLP (scikit-learn) على ميزات EMG
# --------------------------------------------------------------------------
def _train_mlp():
    _, X_feat, y = _synthetic_training_set()
    scaler = StandardScaler().fit(X_feat)
    Xs = scaler.transform(X_feat)
    clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    clf.fit(Xs, y)
    joblib.dump({"scaler": scaler, "clf": clf}, os.path.join(MODEL_DIR, "emg_mlp.joblib"))
    return scaler, clf


def _load_or_train_mlp():
    path = os.path.join(MODEL_DIR, "emg_mlp.joblib")
    if os.path.exists(path):
        bundle = joblib.load(path)
        return bundle["scaler"], bundle["clf"]
    return _train_mlp()


# --------------------------------------------------------------------------
# واجهة موحّدة
# --------------------------------------------------------------------------
class EMGClassifier:
    """واجهة موحدة تختار تلقائياً بين Conv1D (Keras) و MLP الاحتياطي."""

    def __init__(self):
        self.backend = "CNN (TensorFlow/Keras)" if TF_AVAILABLE else "MLP احتياطي (scikit-learn)"
        if TF_AVAILABLE:
            self.model = _load_or_train_keras()
        else:
            self.scaler, self.clf = _load_or_train_mlp()

    def predict_window(self, raw_window: np.ndarray, feature_dict: dict = None):
        """يتنبأ بفئة نافذة واحدة من الإشارة (خام أو ميزات) ويعيد
        (label, confidence)."""
        if TF_AVAILABLE:
            x = np.asarray(raw_window, dtype=float)
            target_len = self.model.input_shape[1]
            if len(x) != target_len:
                x = raw_windows_matrix([x], target_len=target_len)[0]
            x = x.reshape(1, -1, 1)
            probs = self.model.predict(x, verbose=0)[0]
        else:
            if feature_dict is None:
                feats, _ = features_matrix([raw_window])
            else:
                feats = np.array([[feature_dict[k] for k in ["rms", "mav", "zc", "wl", "var", "ssc"]]])
            xs = self.scaler.transform(feats)
            probs = self.clf.predict_proba(xs)[0]
        idx = int(np.argmax(probs))
        return EMG_QUALITY_LABELS[idx], float(probs[idx]), probs

    def predict_batch(self, segments):
        """يصنّف مجموعة من نوافذ الإشارة (segments) ويعيد قائمة نتائج،
        بالإضافة إلى الفئة الغالبة (majority vote) لجلسة القياس كاملة."""
        results = []
        feats, _ = features_matrix(segments)
        for i, seg in enumerate(segments):
            fdict = dict(zip(["rms", "mav", "zc", "wl", "var", "ssc"], feats[i]))
            label, conf, _ = self.predict_window(seg, fdict)
            results.append({"label": label, "confidence": conf, "features": fdict})
        labels = [r["label"] for r in results]
        majority = max(set(labels), key=labels.count) if labels else None
        return results, majority


_classifier_singleton = None


def get_classifier():
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = EMGClassifier()
    return _classifier_singleton
