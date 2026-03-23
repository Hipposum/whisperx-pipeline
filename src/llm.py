"""
llm.py — инициализация GigaChat, сжатие транскрипта, LLM-анализ (таймлайн + отчёт).
         Промпты читаются из файлов в директории prompts/ (относительно корня проекта).
"""

import json
import os
from pathlib import Path

from .utils import fmt_time_short


# ─────────────────────────────────────────────
# Загрузка промптов из файлов
# ─────────────────────────────────────────────

def load_prompts_from_files(prompts_dir=None):
    """
    Читает промпты из текстовых файлов в директории prompts/.
    prompts_dir — путь к папке (по умолчанию: <project_root>/prompts/).
    Возвращает словарь {имя_файла: текст}.
    """
    if prompts_dir is None:
        # Корень проекта — на два уровня выше src/
        project_root = Path(__file__).resolve().parent.parent
        prompts_dir = project_root / "prompts"

    prompts_dir = Path(prompts_dir)
    names = [
        "timeline_system.txt",
        "timeline_user.txt",
        "report_system.txt",
        "report_user.txt",
    ]
    result = {}
    for name in names:
        path = prompts_dir / name
        if path.exists():
            result[name] = path.read_text(encoding="utf-8")
        else:
            print(f"   Промпт не найден: {path}")
    return result


# ─────────────────────────────────────────────
# Инициализация GigaChat
# ─────────────────────────────────────────────

def init_gigachat(credentials, scope, model):
    """Инициализирует клиент GigaChat и проверяет соединение."""
    if not credentials:
        return None
    try:
        from gigachat import GigaChat
        from gigachat.models import Chat, Messages, MessagesRole
        client = GigaChat(
            credentials=credentials, verify_ssl_certs=False,
            scope=scope, model=model
        )
        test = client.chat(Chat(
            messages=[Messages(role=MessagesRole.USER, content="Ответь одним словом: работает?")],
            max_tokens=10, temperature=0.1
        ))
        print(f"   GigaChat ({model}): {test.choices[0].message.content.strip()}")
        return client
    except Exception as e:
        print(f"   GigaChat недоступен: {e}")
        return None


# ─────────────────────────────────────────────
# Сжатие транскрипта
# ─────────────────────────────────────────────

def compress_transcript(segments, max_chars):
    """Сжимает транскрипт для передачи в LLM (объединяет реплики одного спикера)."""
    merged = []
    for seg in segments:
        if seg.get("_is_placeholder"):
            merged.append(seg)
            continue
        if (merged and merged[-1].get("speaker") == seg.get("speaker")
                and not merged[-1].get("_is_placeholder")
                and seg["start"] - merged[-1]["end"] < 2.0):
            merged[-1]["text"] += " " + seg["text"].strip()
            merged[-1]["end"] = seg["end"]
        else:
            merged.append({"start": seg["start"], "end": seg["end"],
                           "speaker": seg.get("speaker", "?"),
                           "text": seg["text"].strip()})
    lines = [f"[{fmt_time_short(s['start'])}] {s.get('speaker','?')}: {s['text']}" for s in merged]
    full = "\n".join(lines)
    if len(full) <= max_chars:
        return full
    step = max(1, int(len(lines) / (max_chars / 80)))
    sampled = (lines[:5] + ["\n[... сжато ...]"] +
               [lines[i] for i in range(5, len(lines)-5, step)] +
               ["\n[... конец ...]"] + lines[-5:])
    return "\n".join(sampled)[:max_chars]


# ─────────────────────────────────────────────
# Запрос к GigaChat
# ─────────────────────────────────────────────

def gc_req(client, sys_p, user_p, temperature, max_tokens=4096):
    """Выполняет один запрос к GigaChat."""
    from gigachat.models import Chat, Messages, MessagesRole
    try:
        resp = client.chat(Chat(
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=sys_p),
                Messages(role=MessagesRole.USER, content=user_p)
            ],
            max_tokens=max_tokens, temperature=temperature
        ))
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"   GigaChat ошибка: {e}")
        return None


def parse_json(raw):
    """Парсит JSON из ответа GigaChat (убирает markdown-обёртку)."""
    if not raw:
        return None
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1]
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    try:
        return json.loads(clean.strip())
    except Exception:
        return {"raw": raw[:2000]}


# ─────────────────────────────────────────────
# Полный LLM-анализ
# ─────────────────────────────────────────────

def run_gigachat_analysis(client, transcript_txt, analytics, config, segments=None, prompts=None):
    """
    Запускает двухэтапный LLM-анализ: таймлайн урока + оценка качества.
    prompts — словарь с ключами: timeline_system.txt, timeline_user.txt,
              report_system.txt, report_user.txt.
    """
    if not client:
        return None

    GIGACHAT_MAX_TRANSCRIPT_CHARS = config.get("gigachat_max_transcript_chars", 25000)
    GIGACHAT_TEMPERATURE = config.get("gigachat_temperature", 0.3)

    if not prompts:
        prompts = load_prompts_from_files()

    TIMELINE_SYSTEM = prompts.get("timeline_system.txt", "")
    TIMELINE_PROMPT = prompts.get("timeline_user.txt", "")
    REPORT_SYSTEM   = prompts.get("report_system.txt", "")
    REPORT_PROMPT   = prompts.get("report_user.txt", "")

    print(f"\nGigaChat анализ...")
    compressed = (compress_transcript(segments, GIGACHAT_MAX_TRANSCRIPT_CHARS)
                  if segments else transcript_txt[:GIGACHAT_MAX_TRANSCRIPT_CHARS])

    print("   Таймлайн урока...")
    timeline = parse_json(gc_req(
        client, TIMELINE_SYSTEM,
        TIMELINE_PROMPT.format(transcript=compressed),
        temperature=GIGACHAT_TEMPERATURE
    ))

    print("   Оценка качества...")
    metrics_short = {k: analytics.get(k)
                     for k in ["noise", "balance", "speech_tempo", "engagement", "questions"]}
    report = parse_json(gc_req(
        client, REPORT_SYSTEM,
        REPORT_PROMPT.format(
            transcript=compressed[:15000],
            metrics=json.dumps(metrics_short, ensure_ascii=False, indent=2)
        ),
        temperature=GIGACHAT_TEMPERATURE
    ))

    if report and "overall_score" in report:
        print(f"   Оценка: {report['overall_score']}/10")

    return {"timeline": timeline, "report": report}
