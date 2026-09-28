# -*- coding: utf-8 -*-
"""
طبقة قاعدة البيانات (SQLite) لتطبيق التأهيل الذكي لعضلات الرقبة.

يخزّن التطبيق:
- بيانات اللاعبين (المجموعة التجريبية / الضابطة)
- نتائج الاختبارات القبلية والبعدية وعبر مراحل البرنامج (8 أسابيع)
- سجلات تحليل EMG بالشبكة العصبية
- سجلات توصيات "الجهاز الذكي" المحاكاة بالذكاء الاصطناعي
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from utils.constants import DB_PATH


def _ensure_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@contextmanager
def get_conn():
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                group_name TEXT NOT NULL CHECK(group_name IN ('تجريبية', 'ضابطة')),
                age INTEGER,
                sport TEXT,
                injury_note TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL,
                stage TEXT NOT NULL,
                test_key TEXT NOT NULL,
                value REAL,
                notes TEXT,
                recorded_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS emg_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER,
                stage TEXT,
                muscle TEXT,
                source_file TEXT,
                predicted_label TEXT,
                confidence REAL,
                rms REAL,
                mav REAL,
                zero_crossings REAL,
                waveform_length REAL,
                model_backend TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS controller_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER,
                direction TEXT,
                emg_activation REAL,
                head_angle REAL,
                angular_velocity REAL,
                recommended_resistance TEXT,
                recommended_resistance_value REAL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS pose_rom_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER,
                stage TEXT,
                source TEXT,
                flexion REAL,
                extension REAL,
                lateral_r REAL,
                lateral_l REAL,
                rotation_r REAL,
                rotation_l REAL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
            );
            """
        )


# ---------------------------------------------------------------- subjects
def add_subject(name, group_name, age=None, sport=None, injury_note=None):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO subjects (name, group_name, age, sport, injury_note) VALUES (?,?,?,?,?)",
            (name, group_name, age, sport, injury_note),
        )
        return cur.lastrowid


def list_subjects(group_name=None):
    with get_conn() as conn:
        if group_name:
            rows = conn.execute(
                "SELECT * FROM subjects WHERE group_name=? ORDER BY id", (group_name,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM subjects ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def delete_subject(subject_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM subjects WHERE id=?", (subject_id,))


# ------------------------------------------------------------ measurements
def upsert_measurement(subject_id, stage, test_key, value, notes=None):
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM measurements WHERE subject_id=? AND stage=? AND test_key=?",
            (subject_id, stage, test_key),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE measurements SET value=?, notes=?, recorded_at=? WHERE id=?",
                (value, notes, datetime.utcnow().isoformat(), existing["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO measurements (subject_id, stage, test_key, value, notes) VALUES (?,?,?,?,?)",
                (subject_id, stage, test_key, value, notes),
            )


def get_measurements(subject_id=None, stage=None, test_key=None):
    query = "SELECT m.*, s.name as subject_name, s.group_name FROM measurements m JOIN subjects s ON s.id = m.subject_id WHERE 1=1"
    params = []
    if subject_id:
        query += " AND m.subject_id=?"
        params.append(subject_id)
    if stage:
        query += " AND m.stage=?"
        params.append(stage)
    if test_key:
        query += " AND m.test_key=?"
        params.append(test_key)
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# ------------------------------------------------------------- emg analyses
def add_emg_analysis(subject_id, stage, muscle, source_file, predicted_label,
                      confidence, features, model_backend):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO emg_analyses
               (subject_id, stage, muscle, source_file, predicted_label, confidence,
                rms, mav, zero_crossings, waveform_length, model_backend)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                subject_id, stage, muscle, source_file, predicted_label, confidence,
                features.get("rms"), features.get("mav"), features.get("zc"),
                features.get("wl"), model_backend,
            ),
        )


def list_emg_analyses(subject_id=None):
    query = "SELECT e.*, s.name as subject_name FROM emg_analyses e LEFT JOIN subjects s ON s.id = e.subject_id"
    params = []
    if subject_id:
        query += " WHERE e.subject_id=?"
        params.append(subject_id)
    query += " ORDER BY e.id DESC"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------- controller logs
def add_controller_log(subject_id, direction, emg_activation, head_angle,
                        angular_velocity, recommended_resistance, recommended_resistance_value):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO controller_logs
               (subject_id, direction, emg_activation, head_angle, angular_velocity,
                recommended_resistance, recommended_resistance_value)
               VALUES (?,?,?,?,?,?,?)""",
            (subject_id, direction, emg_activation, head_angle, angular_velocity,
             recommended_resistance, recommended_resistance_value),
        )


def list_controller_logs(subject_id=None, limit=200):
    query = "SELECT c.*, s.name as subject_name FROM controller_logs c LEFT JOIN subjects s ON s.id=c.subject_id"
    params = []
    if subject_id:
        query += " WHERE c.subject_id=?"
        params.append(subject_id)
    query += " ORDER BY c.id DESC LIMIT ?"
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# ------------------------------------------------------------- pose sessions
def add_pose_rom_session(subject_id, stage, source, angles: dict):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO pose_rom_sessions
               (subject_id, stage, source, flexion, extension, lateral_r, lateral_l, rotation_r, rotation_l)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                subject_id, stage, source,
                angles.get("flexion"), angles.get("extension"),
                angles.get("lateral_r"), angles.get("lateral_l"),
                angles.get("rotation_r"), angles.get("rotation_l"),
            ),
        )


def list_pose_rom_sessions(subject_id=None):
    query = "SELECT p.*, s.name as subject_name FROM pose_rom_sessions p LEFT JOIN subjects s ON s.id=p.subject_id"
    params = []
    if subject_id:
        query += " WHERE p.subject_id=?"
        params.append(subject_id)
    query += " ORDER BY p.id DESC"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
