"""
analytics.py — аналитические функции: шум, темп речи, паузы, баланс,
               слова-паразиты, вопросы, вовлечённость.
"""

import re
import numpy as np

from .utils import fmt_time_short


# ─────────────────────────────────────────────
# Шумность аудио
# ─────────────────────────────────────────────

def analyze_noise(audio, segments, n_raw=0, n_removed=0, sr=16000):
    """Оценка шумности аудио по SNR и другим метрикам."""
    audio_f = audio.astype(np.float64)
    speech_mask = np.zeros(len(audio_f), dtype=bool)
    for seg in segments:
        s = max(0, min(int(seg["start"] * sr), len(audio_f)))
        e = max(0, min(int(seg["end"] * sr), len(audio_f)))
        speech_mask[s:e] = True
    speech_audio  = audio_f[speech_mask]
    silence_audio = audio_f[~speech_mask]

    if len(speech_audio) > sr and len(silence_audio) > sr:
        speech_rms  = np.sqrt(np.mean(speech_audio ** 2))
        silence_rms = np.sqrt(np.mean(silence_audio ** 2))
        snr_db = 20 * np.log10(speech_rms / silence_rms) if silence_rms > 0 else 40.0
    else:
        snr_db = 20.0

    snr_score  = float(np.clip((snr_db - 5) / 25 * 10, 0, 10))
    noise_floor = float(np.sqrt(np.mean(silence_audio ** 2))) if len(silence_audio) > sr else 0.001
    nf_score    = float(np.clip((0.025 - noise_floor) / 0.022 * 10, 0, 10))

    noise_cv = 0.0
    if len(silence_audio) > sr * 2:
        win = sr // 2
        n_win = len(silence_audio) // win
        if n_win > 3:
            energies = np.array([np.sqrt(np.mean(silence_audio[i*win:(i+1)*win]**2)) for i in range(n_win)])
            mu = np.mean(energies)
            noise_cv = float(np.std(energies) / mu) if mu > 0 else 0
    var_score  = float(np.clip((1.5 - noise_cv) / 1.2 * 10, 0, 10))
    hall_rate  = n_removed / n_raw * 100 if n_raw > 0 else 0
    hall_score = float(np.clip((15 - hall_rate) / 13 * 10, 0, 10))
    overall    = round(snr_score * 0.35 + nf_score * 0.25 + var_score * 0.25 + hall_score * 0.15, 1)

    if overall >= 8:   verdict = "отличная"
    elif overall >= 6: verdict = "нормальная"
    elif overall >= 4: verdict = "заметный шум"
    else:              verdict = "шумная"

    return {"score": overall, "verdict": verdict,
            "snr_db": round(snr_db, 1), "noise_floor_rms": round(noise_floor, 5),
            "speech_sec": round(len(speech_audio)/sr, 1),
            "silence_sec": round(len(silence_audio)/sr, 1)}


# ─────────────────────────────────────────────
# Темп речи
# ─────────────────────────────────────────────

def analyze_speech_tempo(segments):
    """Темп речи (слов/мин) по каждому спикеру."""
    by_speaker = {}
    for seg in segments:
        if seg.get("_is_placeholder"):
            continue
        sp = seg.get("speaker", "UNKNOWN")
        words = len(seg["text"].split())
        dur = seg["end"] - seg["start"]
        if dur < 0.1:
            continue
        if sp not in by_speaker:
            by_speaker[sp] = {"words": 0, "duration": 0}
        by_speaker[sp]["words"] += words
        by_speaker[sp]["duration"] += dur
    result = {}
    for sp, d in by_speaker.items():
        wpm = d["words"] / (d["duration"] / 60) if d["duration"] > 1 else 0
        result[sp] = {
            "wpm": round(wpm, 1), "total_words": d["words"],
            "total_sec": round(d["duration"], 1),
            "assessment": ("медленный" if wpm < 80 else "нормальный" if wpm <= 180 else "быстрый")
        }
    return result


# ─────────────────────────────────────────────
# Паузы
# ─────────────────────────────────────────────

def analyze_pauses(segments, teacher_speaker):
    """Анализ пауз между репликами."""
    pauses = []
    for i in range(len(segments) - 1):
        gap = segments[i+1]["start"] - segments[i]["end"]
        if gap > 0.5:
            pauses.append({
                "start": round(segments[i]["end"], 1),
                "end": round(segments[i+1]["start"], 1),
                "duration": round(gap, 1),
                "after_speaker": segments[i].get("speaker", "?"),
                "before_speaker": segments[i+1].get("speaker", "?")
            })
    total_pause = sum(p["duration"] for p in pauses)
    longest = max(pauses, key=lambda p: p["duration"]) if pauses else None
    return {
        "total_pause_sec": round(total_pause, 1), "count": len(pauses),
        "longest": {"time": fmt_time_short(longest["start"]),
                    "duration": longest["duration"]} if longest else None,
        "pauses_over_10s": [{"time": fmt_time_short(p["start"]),
                              "duration": p["duration"]} for p in pauses if p["duration"] > 10]
    }


# ─────────────────────────────────────────────
# Формат урока
# ─────────────────────────────────────────────

def detect_lesson_format(segments):
    """Определяет формат урока: индивидуальный или групповой."""
    from .utils import compute_stats
    speakers = set(seg.get("speaker", "?") for seg in segments if not seg.get("_is_placeholder"))
    speakers.discard("UNKNOWN")
    speakers.discard("?")
    stats = compute_stats([s for s in segments if not s.get("_is_placeholder")])
    teacher = max(stats, key=lambda s: stats[s]["duration"]) if stats else None
    fmt = "individual" if len(speakers) <= 2 else "group"
    return fmt, teacher, len(speakers)


# ─────────────────────────────────────────────
# Баланс учитель/ученики
# ─────────────────────────────────────────────

def analyze_teacher_student_balance(segments, teacher_speaker, lesson_format):
    """Баланс времени речи: учитель vs ученики."""
    by_speaker = {}
    total = 0
    for seg in segments:
        if seg.get("_is_placeholder"):
            continue
        sp = seg.get("speaker", "?")
        dur = seg["end"] - seg["start"]
        by_speaker[sp] = by_speaker.get(sp, 0) + dur
        total += dur
    teacher_pct = by_speaker.get(teacher_speaker, 0) / total * 100 if total > 0 else 0
    if lesson_format == "individual":
        score = 10 - abs(teacher_pct - 60) / 15 * 3 if 45 <= teacher_pct <= 75 else (2 if teacher_pct > 85 else 5)
    else:
        score = 10 - abs(teacher_pct - 50) / 15 * 3 if 35 <= teacher_pct <= 65 else 4
    return {
        "teacher_pct": round(teacher_pct, 1),
        "students_pct": round(100 - teacher_pct, 1),
        "balance_score": round(max(0, min(10, score)), 1),
        "by_speaker": {sp: round(dur/total*100, 1) for sp, dur in by_speaker.items()} if total > 0 else {}
    }


# ─────────────────────────────────────────────
# Слова-паразиты
# ─────────────────────────────────────────────

FILLER_WORDS = {
    "ну": 1, "вот": 1, "как бы": 2, "типа": 2, "типо": 2,
    "это самое": 2, "в общем": 1, "короче": 2, "значит": 1,
    "так сказать": 2, "ну вот": 2, "ну такое": 2, "блин": 1
}
FILLER_SOUNDS = re.compile(r"\b(э{2,}|м{2,}|ээ|эм|ммм|ааа)\b", re.IGNORECASE)


def analyze_filler_words(segments):
    """Анализ слов-паразитов по каждому спикеру."""
    by_speaker = {}
    for seg in segments:
        if seg.get("_is_placeholder"):
            continue
        sp = seg.get("speaker", "?")
        text = seg["text"].lower()
        words = text.split()
        if sp not in by_speaker:
            by_speaker[sp] = {"total_words": 0, "fillers": {}, "sounds": 0}
        by_speaker[sp]["total_words"] += len(words)
        for filler in FILLER_WORDS:
            cnt = text.count(filler)
            if cnt > 0:
                by_speaker[sp]["fillers"][filler] = by_speaker[sp]["fillers"].get(filler, 0) + cnt
        by_speaker[sp]["sounds"] += len(FILLER_SOUNDS.findall(text))
    result = {}
    for sp, d in by_speaker.items():
        total_fillers = sum(d["fillers"].values()) + d["sounds"]
        per100 = total_fillers / d["total_words"] * 100 if d["total_words"] > 0 else 0
        top3 = sorted(d["fillers"].items(), key=lambda x: -x[1])[:3]
        result[sp] = {
            "per_100_words": round(per100, 1), "total_fillers": total_fillers,
            "top": [f[0] for f in top3],
            "assessment": ("чистая речь" if per100 < 3 else "нормально" if per100 < 8 else "много паразитов")
        }
    return result


# ─────────────────────────────────────────────
# Вопросы
# ─────────────────────────────────────────────

QUESTION_STARTERS = re.compile(
    r"^(как|что|где|когда|почему|зачем|сколько|какой|какая|какое|какие|кто|куда|откуда|чем)\b",
    re.IGNORECASE
)


def analyze_questions(segments, teacher_speaker):
    """Определяет вопросы в транскрипции и их тип."""
    questions = []
    for i, seg in enumerate(segments):
        if seg.get("_is_placeholder"):
            continue
        text = seg["text"].strip()
        sp = seg.get("speaker", "?")
        is_q = "?" in text or QUESTION_STARTERS.search(text)
        if not is_q:
            continue
        next_sp = segments[i+1].get("speaker", "?") if i+1 < len(segments) else None
        if sp == teacher_speaker:
            q_type = "rhetorical" if next_sp == teacher_speaker else "pedagogical"
        else:
            q_type = "student_question"
        questions.append({"time": fmt_time_short(seg["start"]), "speaker": sp, "type": q_type})
    return {
        "total": len(questions),
        "pedagogical": sum(1 for q in questions if q["type"] == "pedagogical"),
        "student": sum(1 for q in questions if q["type"] == "student_question"),
        "rhetorical": sum(1 for q in questions if q["type"] == "rhetorical")
    }


# ─────────────────────────────────────────────
# Вовлечённость учеников
# ─────────────────────────────────────────────

def analyze_student_engagement(segments, teacher_speaker, pauses_data, questions_data):
    """Оценка вовлечённости учеников."""
    student_segs = [s for s in segments
                    if s.get("speaker") != teacher_speaker and not s.get("_is_placeholder")]
    if not student_segs or not segments:
        return {"score": 0, "verdict": "нет данных"}
    total_min = (segments[-1]["end"] - segments[0]["start"]) / 60
    reply_rate = len(student_segs) / total_min if total_min > 0 else 0
    avg_words = float(np.mean([len(s["text"].split()) for s in student_segs]))
    student_q = questions_data.get("student", 0)
    freq_score   = min(reply_rate / 5 * 10, 10)
    length_score = min(avg_words / 15 * 10, 10)
    q_score      = min(student_q / 5 * 10, 10)
    overall = round(freq_score*0.4 + length_score*0.35 + q_score*0.25, 1)
    return {
        "score": overall, "reply_rate_per_min": round(reply_rate, 1),
        "avg_reply_words": round(avg_words, 1), "student_questions": student_q,
        "verdict": ("активный" if overall >= 7 else "средний" if overall >= 4 else "пассивный")
    }


# ─────────────────────────────────────────────
# Общий запуск аналитики
# ─────────────────────────────────────────────

def run_all_analytics(audio, segments, n_raw, n_removed, pass2_log, sr=16000):
    """Запускает все аналитические модули и возвращает единый словарь метрик."""
    print("\nАналитика...")
    lesson_fmt, teacher, n_speakers = detect_lesson_format(segments)
    print(f"   Формат: {lesson_fmt} | Учитель: {teacher} | Спикеров: {n_speakers}")

    noise      = analyze_noise(audio, segments, n_raw, n_removed, sr)
    tempo      = analyze_speech_tempo(segments)
    pauses     = analyze_pauses(segments, teacher)
    balance    = analyze_teacher_student_balance(segments, teacher, lesson_fmt)
    fillers    = analyze_filler_words(segments)
    questions  = analyze_questions(segments, teacher)
    engagement = analyze_student_engagement(segments, teacher, pauses, questions)

    print(f"   Шумность: {noise['score']}/10 ({noise['verdict']})")
    print(f"   Баланс: учитель {balance['teacher_pct']:.0f}% | ученики {balance['students_pct']:.0f}%")
    print(f"   Вовлечённость: {engagement['score']}/10 ({engagement['verdict']})")

    n_recovered = sum(1 for l in pass2_log if l.get("status") == "recovered") if pass2_log else 0
    n_placeholders = sum(1 for s in segments if s.get("_is_placeholder"))

    return {
        "lesson_info": {"format": lesson_fmt, "teacher_speaker": teacher, "n_speakers": n_speakers},
        "transcription_quality": {
            "total_raw": n_raw, "hallucinations_removed": n_removed,
            "pass2_recovered": n_recovered, "placeholders": n_placeholders,
            "final_segments": len(segments),
        },
        "noise": noise,
        "speech_tempo": tempo,
        "pauses": pauses,
        "balance": balance,
        "fillers": fillers,
        "questions": questions,
        "engagement": engagement,
    }
