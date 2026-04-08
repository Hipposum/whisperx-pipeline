"""
utils.py — вспомогательные функции: GPU, форматирование, галлюцинации, слияние сегментов.
"""

import gc
import re
import numpy as np
import torch


# ─────────────────────────────────────────────
# GPU-утилиты
# ─────────────────────────────────────────────

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_mem():
    if torch.cuda.is_available():
        u = torch.cuda.memory_allocated() / 1e9
        t = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{u:.1f}/{t:.1f} GB"
    return "CPU"


# ─────────────────────────────────────────────
# Форматирование времени
# ─────────────────────────────────────────────

def fmt_time(s):
    h, m = int(s // 3600), int(s % 3600 // 60)
    return f"{h:02d}:{m:02d}:{s % 60:06.3f}"


def fmt_time_short(s):
    h, m, sec = int(s // 3600), int(s % 3600 // 60), int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ─────────────────────────────────────────────
# Форматирование транскрипции в читаемый текст
# ─────────────────────────────────────────────

def format_output_txt(segments, file_name="", analytics=None):
    """
    Форматирует транскрипцию в читаемый текст для асессора.
    Формат:
      ── 00:01:13  Спикер 1 ────────────────────────────
      Текст реплики...

      ── 00:01:48  Спикер 2 ────────────────────────────
      Другой спикер...
    """
    if not segments:
        return ""

    MERGE_GAP_SEC = 1.5
    LINE_WIDTH    = 60

    speaker_map   = {}
    speaker_count = [0]

    speaker_freq = {}
    for seg in segments:
        raw = seg.get("speaker", "")
        if raw and raw not in ("SPEAKER_??", "UNKNOWN", ""):
            speaker_freq[raw] = speaker_freq.get(raw, 0) + 1

    MIN_REPLICS = 3

    def get_label(raw):
        if not raw or raw in ("SPEAKER_??", "UNKNOWN", ""):
            return "Спикер ?"
        if speaker_freq.get(raw, 0) < MIN_REPLICS:
            return "Спикер ?"
        if raw not in speaker_map:
            speaker_count[0] += 1
            speaker_map[raw] = f"Спикер {speaker_count[0]}"
        return speaker_map[raw]

    # Объединить близкие сегменты одного спикера
    merged = []
    for seg in segments:
        if not merged:
            merged.append(dict(seg))
            continue
        prev = merged[-1]
        same_sp = seg.get("speaker", "") == prev.get("speaker", "")
        gap     = seg["start"] - prev["end"]
        if same_sp and gap <= MERGE_GAP_SEC and not seg.get("_is_placeholder"):
            prev["end"]  = seg["end"]
            t1 = prev["text"].strip()
            t2 = seg["text"].strip()
            prev["text"] = (t1 + " " + t2).strip() if t1 else t2
        else:
            merged.append(dict(seg))

    # ── Шапка ──
    lines = []
    sep = "═" * LINE_WIDTH
    lines.append(sep)
    if file_name:
        lines.append(f"  ТРАНСКРИПЦИЯ: {file_name}")
    if segments:
        duration = segments[-1]["end"] - segments[0]["start"]
        h, m = int(duration // 3600), int(duration % 3600 // 60)
        n_speakers = len({v for v in speaker_map.values()} | {"Спикер ?"}) - 1
        lines.append(f"  Длительность: {h}ч {m:02d}мин  |  Сегментов: {len(merged)}")
    if analytics:
        bal  = analytics.get("balance", {})
        eng  = analytics.get("engagement", {})
        nois = analytics.get("noise", {})
        teacher = analytics.get("lesson_info", {}).get("teacher_speaker", "")
        teacher_label = get_label(teacher) if teacher else "?"
        lines.append(f"  Учитель: {teacher_label}  |  Баланс: {bal.get('teacher_pct','?'):.0f}% / {bal.get('students_pct','?'):.0f}%  |  Шум: {nois.get('score','?')}/10")
        lines.append(f"  Вовлечённость: {eng.get('score','?')}/10 ({eng.get('verdict','')})")
    lines.append(sep)
    lines.append("")

    # ── Реплики ──
    last_label = None
    for seg in merged:
        text = seg["text"].strip()
        if not text:
            continue

        label = get_label(seg.get("speaker", ""))
        t = seg["start"]
        h, m, s = int(t // 3600), int(t % 3600 // 60), int(t % 60)
        ts = f"{h:02d}:{m:02d}:{s:02d}"

        if label != last_label:
            if last_label is not None:
                lines.append("")
            header = f"── {ts}  {label} "
            lines.append(header + "─" * max(4, LINE_WIDTH - len(header)))
            last_label = label

        if seg.get("_is_placeholder"):
            lines.append(f"  [{text}]")
        else:
            lines.append(f"  {text}")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def format_metrics_txt(analytics, llm_result, audio_duration, file_name=""):
    """Читаемый отчёт с метриками для асессора."""
    W = 60
    sep  = "═" * W
    sep2 = "─" * W
    lines = []

    def row(label, value, width=W):
        dots = "." * (width - len(label) - len(str(value)) - 2)
        return f"  {label} {dots} {value}"

    def rating_bar(score, max_score=10):
        filled = int(round(score / max_score * 10))
        return "█" * filled + "░" * (10 - filled) + f"  {score}/{max_score}"

    lines += [sep, f"  ОТЧЁТ ПО УРОКУ: {file_name}", sep, ""]

    # ── Общая информация ──
    h = int(audio_duration // 3600)
    m = int(audio_duration % 3600 // 60)
    lines.append("  ОБЩАЯ ИНФОРМАЦИЯ")
    lines.append(sep2)
    lines.append(row("Длительность", f"{h}ч {m:02d}мин"))

    info = analytics.get("lesson_info", {})
    fmt_map = {"group": "Групповой", "individual": "Индивидуальный"}
    lines.append(row("Формат урока",  fmt_map.get(info.get("format", ""), "—")))
    lines.append(row("Спикеров",      info.get("n_speakers", "—")))
    lines.append(row("Учитель",       info.get("teacher_speaker", "—")))

    tq = analytics.get("transcription_quality", {})
    lines.append(row("Сегментов всего",    tq.get("total_raw", "—")))
    lines.append(row("Восстановлено (pass2)", tq.get("pass2_recovered", 0)))
    lines.append(row("Неразборчивых",     tq.get("placeholders", 0)))
    lines.append("")

    # ── Метрики ──
    lines.append("  АВТОМАТИЧЕСКИЕ МЕТРИКИ")
    lines.append(sep2)

    noise = analytics.get("noise", {})
    lines.append(row("Качество звука",  f"{noise.get('score',0):.1f}/10  {noise.get('verdict','')}"))
    lines.append(row("SNR",             f"{noise.get('snr_db','?')} дБ"))
    lines.append("")

    bal = analytics.get("balance", {})
    lines.append(row("Учитель говорит",  f"{bal.get('teacher_pct',0):.0f}%"))
    lines.append(row("Ученики говорят",  f"{bal.get('students_pct',0):.0f}%"))
    lines.append(row("Баланс (оценка)",  f"{bal.get('balance_score',0):.1f}/10"))
    lines.append("")

    eng = analytics.get("engagement", {})
    lines.append(row("Вовлечённость",        f"{eng.get('score',0):.1f}/10  {eng.get('verdict','')}"))
    lines.append(row("Реплик учеников/мин",  f"{eng.get('reply_rate_per_min','?')}"))
    lines.append(row("Средняя длина реплики", f"{eng.get('avg_reply_words','?')} слов"))
    lines.append("")

    pauses = analytics.get("pauses", {})
    lines.append(row("Пауз всего",     f"{pauses.get('count',0)} шт"))
    lines.append(row("Суммарно пауз",  f"{pauses.get('total_pause_sec',0):.0f} сек"))
    if pauses.get("longest"):
        lines.append(row("Самая длинная пауза",
            f"{pauses['longest']['duration']:.1f}с в {pauses['longest']['time']}"))
    if pauses.get("pauses_over_10s"):
        lines.append(row("Пауз > 10 сек", str(len(pauses["pauses_over_10s"]))))
    lines.append("")

    q = analytics.get("questions", {})
    lines.append(row("Вопросов учителя",  q.get("pedagogical", 0)))
    lines.append(row("Вопросов учеников", q.get("student", 0)))
    lines.append(row("Риторических",      q.get("rhetorical", 0)))
    lines.append("")

    tempo = analytics.get("speech_tempo", {})
    if tempo:
        lines.append("  ТЕМП РЕЧИ ПО СПИКЕРАМ")
        lines.append(sep2)
        for sp, t in sorted(tempo.items()):
            wpm = t.get("wpm", 0)
            lines.append(row(sp, f"{wpm:.0f} сл/мин  {t.get('assessment','')}"))
        lines.append("")

    fillers = analytics.get("fillers", {})
    if fillers:
        lines.append("  СЛОВА-ПАРАЗИТЫ")
        lines.append(sep2)
        for sp, f in sorted(fillers.items()):
            top = ", ".join(f.get("top", [])) or "—"
            lines.append(row(sp, f"{f.get('per_100_words',0):.1f}/100 слов  ({top})"))
        lines.append("")

    # ── GigaChat оценка ──
    if llm_result:
        lines.append("  ОЦЕНКА ГИГАЧАТ (АСЕССОРИНГ)")
        lines.append(sep2)
        score = llm_result.get("total_score")
        topic = llm_result.get("lesson_topic", "")
        if topic:
            lines.append(row("Тема урока", topic))
        if score is not None:
            lines.append(row("ИТОГОВЫЙ БАЛЛ", f"{score} / 14"))
        lines.append("")

        CRITERION_NAMES = {
            "quality_sound":       "Качество звука",
            "quality_recording":   "Качество записи",
            "quality_demo":        "Качество демонстрации",
            "background":          "Задний фон",
            "homework_discussion": "Обсуждение ДЗ",
            "communication":       "Коммуникация с учениками",
            "answering_questions": "Ответы на вопросы",
            "practice":            "Практика на уроке",
            "subject_knowledge":   "Знания темы",
            "new_homework":        "Новое ДЗ",
            "group_dynamics":      "Групповая динамика",
            "structure_clarity":   "Структура и понятность",
        }
        assessment = llm_result.get("assessment", {}) or {}
        for key, name in CRITERION_NAMES.items():
            c = assessment.get(key, {})
            if not c:
                continue
            rating  = c.get("rating", "—")
            comment = c.get("comment", "")
            icon = {"В норме": "✓", "Спорно": "!", "Особое внимание": "✗", "Без оценки": "·"}.get(rating, " ")
            lines.append(f"  {icon} {name:<32} {rating}")
            if comment and comment not in ("нужен просмотр видео", ""):
                lines.append(f"      → {comment}")
        lines.append("")

        comment = llm_result.get("comment", "")
        if comment:
            lines.append("  КОММЕНТАРИЙ")
            lines.append(sep2)
            # Перенос длинных строк
            for word in comment.split():
                if not lines[-1].startswith("  ") or len(lines[-1]) + len(word) + 1 > W - 2:
                    lines.append("  " + word)
                else:
                    lines[-1] += " " + word
            lines.append("")

        if llm_result.get("timeline"):
            lines.append("  ЭТАПЫ УРОКА")
            lines.append(sep2)
            for stage in llm_result["timeline"]:
                dur = stage.get("duration_min", "?")
                lines.append(f"  {stage.get('start','?')}–{stage.get('end','?')}  "
                             f"({dur} мин)  {stage.get('stage','')}")
            lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Паттерны галлюцинаций Whisper
# ─────────────────────────────────────────────

HALLUCINATION_PATTERNS = [
    r"[Рр]едактор субтитров",
    r"[Кк]орректор\s+[А-Я]\.",
    r"[Пп]родолжение следует",
    r"[Сс]убтитры\s+(сделан|создан|выполнен)",
    r"[Пп]одписывайтесь на канал",
    r"[Сс]пасибо за (просмотр|подписку|внимание)",
    r"[Дд]о новых встреч",
    r"[Дд]о следующего (выпуска|видео|эфира)",
    r"[Нн]е забудьте подписаться",
    r"[Сс]тавьте лайк",
    r"[Рр]ечь академическая",
    r"[Вв]озможны термины из",
    r"[Оо]нлайн урок с репетитором",
    r"[Уу]читель объясняет материал",
    r"[Уу]ченик задаёт вопросы",
    r"^\.+$",
    r"^\s*\.{2,}\s*$",
    r"^[\s\.\,\!\?]+$",
    r"^\s*Музыка\s*$",
    r"^\s*♪",
]
_HALL_RE = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]


def is_hallucination(text):
    t = text.strip()
    if not t:
        return True
    return any(p.search(t) for p in _HALL_RE)


# ─────────────────────────────────────────────
# Обнаружение речи в промежутке
# ─────────────────────────────────────────────

def check_speech_in_gap(audio, gap_start, gap_end, sr=16000):
    """Проверяет наличие речи в промежутке между сегментами."""
    s = max(0, int(gap_start * sr))
    e = min(len(audio), int(gap_end * sr))
    if e <= s or e - s < sr // 4:
        return False
    chunk = audio[s:e].astype(np.float64)
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    if rms < 0.003:
        return False
    active_ratio = float(np.mean(np.abs(chunk) > 0.01))
    return active_ratio > 0.05


# ─────────────────────────────────────────────
# Объединение зон
# ─────────────────────────────────────────────

def merge_zones(zones):
    """Объединяет перекрывающиеся проблемные зоны."""
    if not zones:
        return zones
    zones = sorted(zones, key=lambda z: z["start"])
    merged = [zones[0]]
    for z in zones[1:]:
        if z["start"] <= merged[-1]["end"] + 1.0:
            merged[-1]["end"] = max(merged[-1]["end"], z["end"])
            merged[-1]["reason"] += "+" + z["reason"]
        else:
            merged.append(z)
    return merged


# ─────────────────────────────────────────────
# Обнаружение проблемных зон
# ─────────────────────────────────────────────

def detect_problem_zones(segments, n_raw, audio=None, sr=16000,
                         pass2_low_confidence=0.45):
    """Фильтрует галлюцинации и находит зоны для прохода 2."""
    clean = []
    problems = []
    removed = []
    prev_text = ""

    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        dur = seg.get("end", 0) - seg.get("start", 0)

        if is_hallucination(text):
            problems.append({"start": seg.get("start", 0), "end": seg.get("end", 0),
                             "reason": "hallucination", "original_text": text})
            removed.append({"reason": "hallucination", "text": text,
                            "time": f"{seg.get('start', 0):.1f}s"})
            continue

        if dur < 0.3 and len(text) <= 2:
            removed.append({"reason": "too_short", "text": text,
                            "time": f"{seg.get('start', 0):.1f}s"})
            continue

        if text == prev_text and len(text) > 10:
            removed.append({"reason": "duplicate", "text": text,
                            "time": f"{seg.get('start', 0):.1f}s"})
            continue

        words = seg.get("words", [])
        if words and len(words) > 2:
            confidences = [w.get("score", 1.0) for w in words if "score" in w]
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                if avg_conf < pass2_low_confidence:
                    problems.append({"start": seg.get("start", 0), "end": seg.get("end", 0),
                                     "reason": "low_confidence", "original_text": text,
                                     "avg_confidence": round(avg_conf, 3)})

        clean.append(seg)
        prev_text = text

    # Проверка промежутков между сегментами
    for i in range(len(clean) - 1):
        gap = clean[i+1]["start"] - clean[i]["end"]
        gap_start = clean[i]["end"]
        gap_end = clean[i+1]["start"]
        if 0.5 < gap < 3.0:
            words_end = clean[i].get("words", [])
            words_start = clean[i+1].get("words", [])
            low_end = words_end and words_end[-1].get("score", 1.0) < 0.5
            low_start = words_start and words_start[0].get("score", 1.0) < 0.5
            if low_end or low_start:
                problems.append({"start": gap_start - 1.0, "end": gap_end + 1.0,
                                 "reason": "chunk_boundary", "original_text": ""})
        elif 3.0 < gap < 60.0:
            has_speech = True
            if audio is not None and len(audio) > 0:
                has_speech = check_speech_in_gap(audio, gap_start, gap_end, sr)
            if has_speech:
                problems.append({"start": gap_start, "end": gap_end,
                                 "reason": "large_gap_with_speech",
                                 "original_text": f"gap {gap:.1f}s"})

    problems = merge_zones(problems)

    if removed:
        reasons = {}
        for r in removed:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        print(f"\n   Удалено {len(removed)} из {n_raw} сегментов:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lbl = {"hallucination": "галлюцинации", "too_short": "короткие",
                   "duplicate": "дубликаты"}.get(reason, reason)
            print(f"      {lbl}: {count}")
    if problems:
        print(f"   Проблемных зон для прохода 2: {len(problems)}")

    return clean, problems, removed


# ─────────────────────────────────────────────
# Объединение сегментов Pass1 + Pass2
# ─────────────────────────────────────────────

def merge_pass2_segments(clean_segments, recovered_segments):
    """
    Объединяет сегменты прохода 1 и прохода 2.
    Удаляет дубли: если Pass2-сегмент перекрывается с Pass1 более чем на 50% — выбрасываем Pass2.
    """
    all_segs = clean_segments + recovered_segments
    all_segs.sort(key=lambda s: s["start"])

    result = []
    for seg in all_segs:
        if not result:
            result.append(seg)
            continue
        prev = result[-1]
        overlap_start = max(seg["start"], prev["start"])
        overlap_end   = min(seg["end"],   prev["end"])
        overlap = max(0.0, overlap_end - overlap_start)
        seg_dur = max(0.01, seg["end"] - seg["start"])
        # Если текущий сегмент перекрывается с предыдущим более чем на 40% — пропускаем
        if overlap / seg_dur > 0.4:
            # Оставляем тот у которого выше уверенность (или Pass1 по умолчанию)
            seg_conf  = seg.get("avg_logprob", -1.0)
            prev_conf = prev.get("avg_logprob", -1.0)
            if seg_conf > prev_conf:
                result[-1] = seg
            # иначе оставляем prev
            continue
        result.append(seg)
    return result


# ─────────────────────────────────────────────
# Статистика по спикерам
# ─────────────────────────────────────────────

def compute_stats(segments):
    stats = {}
    for seg in segments:
        sp = seg.get("speaker", "UNKNOWN")
        dur = seg["end"] - seg["start"]
        if sp not in stats:
            stats[sp] = {"count": 0, "duration": 0.0, "words": 0}
        stats[sp]["count"] += 1
        stats[sp]["duration"] += dur
        stats[sp]["words"] += len(seg["text"].split())
    return stats
