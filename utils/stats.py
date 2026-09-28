# -*- coding: utf-8 -*-
"""
الوسائل الإحصائية المطلوبة للفصل الرابع من الأطروحة (نتائج البحث):
- اختبار (t) للعينات المترابطة (Paired t-test): الفروق القبلية/البعدية
  داخل كل مجموعة (تجريبية أو ضابطة).
- اختبار (t) للعينات المستقلة (Independent t-test): المقارنة بين
  المجموعتين التجريبية والضابطة (بعدياً أو في مقدار التحسن).
- الإحصاء الوصفي (الوسط الحسابي، الانحراف المعياري).
- حجم الأثر (Cohen's d).
"""
import numpy as np
import pandas as pd
from scipy import stats


def descriptive(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def cohens_d_paired(pre, post):
    pre, post = np.asarray(pre, dtype=float), np.asarray(post, dtype=float)
    diff = post - pre
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def cohens_d_independent(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / pooled_sd) if pooled_sd > 0 else 0.0


def paired_ttest(pre, post):
    """اختبار t للعينات المترابطة (قبلي/بعدي لنفس المجموعة)."""
    pre = np.asarray([v for v in pre if v is not None], dtype=float)
    post = np.asarray([v for v in post if v is not None], dtype=float)
    n = min(len(pre), len(post))
    pre, post = pre[:n], post[:n]
    if n < 2:
        return {"n": n, "t": None, "p": None, "d": None, "significant": None}
    t, p = stats.ttest_rel(pre, post)
    return {
        "n": n,
        "pre": descriptive(pre),
        "post": descriptive(post),
        "t": float(t),
        "p": float(p),
        "d": cohens_d_paired(pre, post),
        "significant": bool(p < 0.05),
    }


def independent_ttest(group_a, group_b):
    """اختبار t للعينات المستقلة (تجريبية مقابل ضابطة)."""
    group_a = np.asarray([v for v in group_a if v is not None], dtype=float)
    group_b = np.asarray([v for v in group_b if v is not None], dtype=float)
    if len(group_a) < 2 or len(group_b) < 2:
        return {"t": None, "p": None, "d": None, "significant": None}
    t, p = stats.ttest_ind(group_a, group_b, equal_var=False)
    return {
        "a": descriptive(group_a),
        "b": descriptive(group_b),
        "t": float(t),
        "p": float(p),
        "d": cohens_d_independent(group_a, group_b),
        "significant": bool(p < 0.05),
    }


def build_comparison_table(measurements: list, test_key: str, pre_stage: str, post_stage: str):
    """يبني جدول مقارنة كامل (تجريبية/ضابطة × قبلي/بعدي) لمتغير واحد،
    بصيغة مناسبة لإدراجه مباشرة في جداول نتائج الأطروحة (الفصل الرابع)."""
    df = pd.DataFrame(measurements)
    df = df[df["test_key"] == test_key]

    def _vals(group, stage):
        sub = df[(df["group_name"] == group) & (df["stage"] == stage)]
        return sub.sort_values("subject_id")["value"].tolist()

    exp_pre = _vals("تجريبية", pre_stage)
    exp_post = _vals("تجريبية", post_stage)
    ctrl_pre = _vals("ضابطة", pre_stage)
    ctrl_post = _vals("ضابطة", post_stage)

    return {
        "test_key": test_key,
        "experimental_within": paired_ttest(exp_pre, exp_post),
        "control_within": paired_ttest(ctrl_pre, ctrl_post),
        "post_between_groups": independent_ttest(exp_post, ctrl_post),
        "pre_between_groups": independent_ttest(exp_pre, ctrl_pre),
        "raw": {
            "exp_pre": exp_pre, "exp_post": exp_post,
            "ctrl_pre": ctrl_pre, "ctrl_post": ctrl_post,
        },
    }
