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

def format_output_txt(segments, file_name=""):
    """
    Форматирует транскрипцию в читаемый текст.
    Формат (как у speech2text.ru):
      Спикер 1: 00:01:13 — Текст реплики.
      00:01:45 — Продолжение того же спикера.
      Спикер 2: 00:01:48 — Другой спикер.
    Правила:
      - Спикеры нумеруются по порядку появления (Спикер 1, 2, ...)
      - Имя спикера показывается только при смене
      - Соседние сегменты одного спикера (gap < MERGE_GAP_SEC) объединяются
    """
    if not segments:
        return ""

    MERGE_GAP_SEC = 1.5

    speaker_map   = {}
    speaker_count = [0]

    # Считаем сколько раз встречается каждый сырой спикер
    speaker_freq = {}
    for seg in segments:
        raw = seg.get("speaker", "")
        if raw and raw not in ("SPEAKER_??", "UNKNOWN", ""):
            speaker_freq[raw] = speaker_freq.get(raw, 0) + 1

    # Редкие спикеры (< 3 реплик) → "Спикер ?"
    MIN_REPLICS = 3

    def get_label(raw):
        if not raw or raw in ("SPEAKER_??", "UNKNOWN", ""):
            return "Спикер ?"
        if speaker_freq.get(raw, 0) < MIN_REPLICS:
            return "Спикер ?"          # слишком мало реплик — скорее всего ошибка диаризации
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
        if same_sp and gap <= MERGE_GAP_SEC:
            prev["end"]  = seg["end"]
            t1 = prev["text"].strip()
            t2 = seg["text"].strip()
            prev["text"] = (t1 + " " + t2).strip() if t1 else t2
        else:
            merged.append(dict(seg))

    lines = []
    if file_name:
        lines.append(file_name)
        lines.append("")

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
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"{label}: {ts} — {text}")
            last_label = label
        else:
            lines.append(f"{ts} — {text}")

    lines.append("")
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
