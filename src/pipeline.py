"""
pipeline.py — основная точка входа. Оркестрирует все 7 этапов обработки видео:
  1. Загрузка аудио
  1б. Шумоподавление
  2. Транскрибация (Pass 1 + Silero VAD)
  3. Пост-обработка + Pass 2
  4. Alignment (wav2vec2)
  5. Диаризация (pyannote)
  6. Аналитика
  7. GigaChat LLM-анализ

Конфигурация — config/settings.yaml.
Секреты — переменные среды: HF_TOKEN, GIGACHAT_CREDENTIALS, YANDEX_TOKEN.
"""

import gc
import json
import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Чекпоинты (per-stage resume)
# ─────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    """JSON-encoder с поддержкой numpy scalar/array."""
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def _ckpt_dir(work_dir, base_name):
    d = os.path.join(work_dir, f"_{base_name}_ckpt")
    os.makedirs(d, exist_ok=True)
    return d


def _save_ckpt(work_dir, base_name, stage, data):
    path = os.path.join(_ckpt_dir(work_dir, base_name), f"s{stage}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, cls=_NpEncoder)


def _load_ckpt(work_dir, base_name, stage):
    path = os.path.join(os.path.join(work_dir, f"_{base_name}_ckpt"), f"s{stage}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _clear_ckpt(work_dir, base_name):
    shutil.rmtree(os.path.join(work_dir, f"_{base_name}_ckpt"), ignore_errors=True)


if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]


# ─────────────────────────────────────────────
# Загрузка конфигурации
# ─────────────────────────────────────────────

def load_config(config_path=None):
    """Читает config/settings.yaml. Возвращает словарь настроек."""
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "config" / "settings.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────
# Главная функция
# ─────────────────────────────────────────────

def main(config_path=None):
    from .transcription import (
        load_silero_vad, load_audio, apply_noise_reduction,
        transcribe_pass1, retranscribe_zones, run_alignment,
    )
    from .diarization import run_diarization
    from .analytics import run_all_analytics
    from .llm import init_gigachat, run_gigachat_analysis, load_prompts_from_files
    from .storage import (
        download_videos, upload_results, load_progress, save_progress,
    )
    from .utils import (
        free_gpu, format_output_txt, detect_problem_zones,
        merge_pass2_segments, compute_stats,
    )

    # ── Конфиг ──
    cfg = load_config(config_path)

    # ── Секреты из переменных среды ──
    HF_TOKEN              = os.environ.get("HF_TOKEN", "")
    GIGACHAT_CREDENTIALS  = os.environ.get("GIGACHAT_CREDENTIALS", "")
    YANDEX_TOKEN          = os.environ.get("YANDEX_TOKEN", "")

    # ── Параметры из конфига ──
    MODE                    = cfg.get("mode", "yadisk")
    WHISPER_MODEL           = cfg["whisper_model"]
    LANGUAGE                = cfg["language"]
    WORK_DIR                = cfg.get("work_dir", "/kaggle/working/results")
    DOWNLOAD_DIR            = cfg.get("download_dir", "/kaggle/working/videos")
    VIDEOS_FOLDER           = cfg.get("videos_folder", "/PachcaVideos")
    OUTPUT_FOLDER           = cfg.get("output_folder", "/settings/output")
    MAX_VIDEOS              = cfg.get("max_videos", 5)
    SKIP_ALREADY_TRANSCRIBED = cfg.get("skip_already_transcribed", True)
    LOCAL_FILE              = cfg.get("local_file", "")
    USE_SILERO_VAD          = cfg.get("use_silero_vad", True)
    USE_NOISE_REDUCE        = cfg.get("use_noise_reduce", True)
    ENABLE_PASS2            = cfg.get("enable_pass2", True)
    DIARIZE                 = cfg.get("diarize", True)
    RUN_LLM_ANALYSIS        = cfg.get("run_llm_analysis", True)
    UPLOAD_RESULTS          = cfg.get("upload_results_to_yadisk", True)
    GIGACHAT_MODEL          = cfg.get("gigachat_model", "GigaChat")
    GIGACHAT_SCOPE          = cfg.get("gigachat_scope", "GIGACHAT_API_PERS")

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    ZIP_EXTENSIONS   = {".zip"}

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Авто-отключение при отсутствии токенов
    if DIARIZE and not HF_TOKEN:
        print("HF_TOKEN отсутствует — диаризация отключена")
        DIARIZE = False
    if RUN_LLM_ANALYSIS and not GIGACHAT_CREDENTIALS:
        print("GIGACHAT_CREDENTIALS отсутствует — LLM-анализ отключён")
        RUN_LLM_ANALYSIS = False

    # ── Получение списка видео ──
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if MODE == "yadisk":
        video_files = download_videos(
            YANDEX_TOKEN, VIDEOS_FOLDER, DOWNLOAD_DIR, WORK_DIR,
            MAX_VIDEOS, SKIP_ALREADY_TRANSCRIBED,
            VIDEO_EXTENSIONS, ZIP_EXTENSIONS
        )
    elif MODE == "local":
        if LOCAL_FILE and os.path.exists(LOCAL_FILE):
            ext = os.path.splitext(LOCAL_FILE)[-1].lower()
            from .storage import resolve_to_video_files
            raw = [{"local_path": LOCAL_FILE, "remote_path": None,
                    "name": os.path.basename(LOCAL_FILE), "is_zip": ext in ZIP_EXTENSIONS}]
            video_files = resolve_to_video_files(raw, DOWNLOAD_DIR, VIDEO_EXTENSIONS, ZIP_EXTENSIONS)
        else:
            print("LOCAL_FILE не задан или файл не найден!")
            video_files = []
    else:
        video_files = []

    print(f"\nИтого видео к обработке: {len(video_files)}")
    for vf in video_files:
        tag = f" (из {vf['from_zip']})" if vf.get('from_zip') else ""
        print(f"   {vf['name']}{tag} ({os.path.getsize(vf['local_path'])/1e6:.0f} MB)")

    # ── Инициализация GigaChat ──
    gigachat_client = None
    if RUN_LLM_ANALYSIS:
        print("\nИнициализация GigaChat...")
        gigachat_client = init_gigachat(GIGACHAT_CREDENTIALS, GIGACHAT_SCOPE, GIGACHAT_MODEL)
        if not gigachat_client:
            print("   GigaChat недоступен — LLM-анализ будет пропущен")

    # ── Загрузка промптов из репозитория (prompts/) ──
    print("\nЗагрузка промптов из GitHub репозитория...")
    prompts = load_prompts_from_files()
    print(f"   Загружено промптов: {len(prompts)}")

    # ── Загрузка Silero VAD ──
    silero_model, silero_utils = None, None
    if USE_SILERO_VAD:
        print("\nЗагрузка Silero VAD...")
        silero_model, silero_utils = load_silero_vad(
            DEVICE,
            cfg.get("silero_threshold", 0.15),
            cfg.get("silero_min_speech_ms", 100),
            cfg.get("silero_min_silence_ms", 600),
        )

    progress_file = os.path.join(WORK_DIR, "_progress.json")
    progress = load_progress(progress_file)

    # ════════════════════════════════════════════════════════════════════
    # ГЛАВНЫЙ ЦИКЛ ОБРАБОТКИ
    # ════════════════════════════════════════════════════════════════════

    MIN_DURATION_SEC = cfg.get("min_duration_sec", 120)  # пропускать видео < 2 мин

    for file_idx, vf in enumerate(video_files, 1):
        file_path = vf['local_path']
        file_name = vf['name']
        base_name = Path(file_name).stem
        total_start = time.time()

        print(f"\n{'='*65}")
        print(f"[{file_idx}/{len(video_files)}] {file_name}")
        if vf.get('from_zip'):
            print(f"   (извлечено из {vf['from_zip']})")
        print(f"{'='*65}")

        # ── Загрузка чекпоинтов текущего видео ────────────────────────
        ckpt2 = _load_ckpt(WORK_DIR, base_name, 2)
        ckpt3 = _load_ckpt(WORK_DIR, base_name, 3)
        ckpt4 = _load_ckpt(WORK_DIR, base_name, 4)
        ckpt5 = _load_ckpt(WORK_DIR, base_name, 5)
        ckpt6 = _load_ckpt(WORK_DIR, base_name, 6)
        ckpt7 = _load_ckpt(WORK_DIR, base_name, 7)
        if any([ckpt2, ckpt3, ckpt4, ckpt5, ckpt6, ckpt7]):
            found = [s for s, c in [(2,ckpt2),(3,ckpt3),(4,ckpt4),(5,ckpt5),(6,ckpt6),(7,ckpt7)] if c]
            print(f"   Найден чекпоинт: этапы {found} — продолжаем с места остановки")

        # Нужно ли загружать аудио (не нужно если этапы 2-5 уже в кэше)
        need_audio = not (ckpt2 and ckpt3 and ckpt4 and ckpt5)

        # ── Этап 1/7: Загрузка аудио ──────────────────────────────────
        if need_audio:
            print(f"\nЭтап 1/7: Загрузка аудио")
            t0 = time.time()
            try:
                audio, audio_duration = load_audio(file_path)
            except Exception as e:
                print(f"   ОШИБКА загрузки аудио: {e} — пропускаем")
                continue
            print(f"   {audio_duration:.0f}с ({audio_duration/60:.1f} мин) | {time.time()-t0:.1f}с")
            if audio_duration < MIN_DURATION_SEC:
                print(f"   Видео слишком короткое ({audio_duration:.0f}с < {MIN_DURATION_SEC}с) — пропускаем")
                continue
        else:
            audio = None
            audio_duration = ckpt2.get("audio_duration", 0)
            print(f"\nЭтап 1/7: Загрузка аудио — пропущено (из кэша, {audio_duration/60:.1f} мин)")

        # ── Этап 1б/7: Шумоподавление ─────────────────────────────────
        if need_audio and not ckpt2:
            print(f"\nЭтап 1б/7: Шумоподавление + нормализация")
            if USE_NOISE_REDUCE:
                audio = apply_noise_reduction(
                    audio,
                    cfg.get("nr_prop_decrease", 0.4),
                    cfg.get("nr_stationary", False)
                )
            else:
                print("   Пропущено (use_noise_reduce=false)")

        # ── Этап 2/7: Транскрибация — проход 1 ────────────────────────
        if ckpt2:
            print(f"\nЭтап 2/7: Транскрибация — [из кэша]")
            result = {"segments": ckpt2["segments"]}
            n_raw = ckpt2["n_raw"]
        else:
            print(f"\nЭтап 2/7: Транскрибация — проход 1 ({WHISPER_MODEL})")
            try:
                result, n_raw, _ = transcribe_pass1(
                    audio, cfg, DEVICE, silero_model, silero_utils
                )
            except Exception as e:
                print(f"   ОШИБКА транскрибации: {e} — пропускаем видео")
                free_gpu()
                continue
            _save_ckpt(WORK_DIR, base_name, 2, {
                "segments": result["segments"], "n_raw": n_raw,
                "audio_duration": round(audio_duration, 1)
            })

        # ── Этап 3/7: Пост-обработка + Проход 2 ───────────────────────
        if ckpt3:
            print(f"\nЭтап 3/7: Пост-обработка — [из кэша]")
            segments_merged = ckpt3["segments_merged"]
            pass2_log       = ckpt3["pass2_log"]
            n_removed       = ckpt3["n_removed"]
            n_recovered     = ckpt3["n_recovered"]
            n_placeholders  = sum(1 for s in segments_merged if s.get("_is_placeholder"))
            print(f"   {len(segments_merged)} сегм. | восстановлено: {n_recovered} | неразборчиво: {n_placeholders}")
        else:
            print(f"\nЭтап 3/7: Пост-обработка + {'проход 2' if ENABLE_PASS2 else 'без прохода 2'}")
            clean_segs, problem_zones, removal_log = detect_problem_zones(
                result["segments"], n_raw, audio=audio, sr=16000,
                pass2_low_confidence=cfg.get("pass2_low_confidence", 0.45)
            )
            n_removed = len(removal_log)
            pass2_log = []
            if ENABLE_PASS2 and problem_zones:
                recovered_segs, pass2_log = retranscribe_zones(audio, problem_zones, cfg, DEVICE)
                segments_merged = merge_pass2_segments(clean_segs, recovered_segs)
            else:
                placeholders = [{
                    "start": round(z["start"], 3), "end": round(z["end"], 3),
                    "text": f"[неразборчиво — {z['end'] - z['start']:.1f}с]",
                    "speaker": "UNKNOWN", "_is_placeholder": True,
                } for z in problem_zones]
                segments_merged = merge_pass2_segments(clean_segs, placeholders)
            n_placeholders = sum(1 for s in segments_merged if s.get("_is_placeholder"))
            n_recovered    = sum(1 for l in pass2_log if l.get("status") == "recovered")
            print(f"   Итого: {len(segments_merged)} сегм. | восстановлено: {n_recovered} | неразборчиво: {n_placeholders}")
            _save_ckpt(WORK_DIR, base_name, 3, {
                "segments_merged": segments_merged,
                "pass2_log": pass2_log,
                "n_removed": n_removed,
                "n_recovered": n_recovered,
            })

        # ── Этап 4/7: Alignment ────────────────────────────────────────
        if ckpt4:
            print(f"\nЭтап 4/7: Выравнивание — [из кэша]")
            all_segments     = ckpt4["all_segments"]
            placeholders_list = ckpt4["placeholders"]
            aligned_result   = {"word_segments": ckpt4["word_segments"]}
        else:
            print(f"\nЭтап 4/7: Выравнивание (wav2vec2)")
            t0 = time.time()
            aligned_result, all_segments, placeholders_list = run_alignment(
                audio, segments_merged, cfg, DEVICE
            )
            print(f"   {time.time()-t0:.0f}с")
            _save_ckpt(WORK_DIR, base_name, 4, {
                "all_segments": all_segments,
                "placeholders": placeholders_list,
                "word_segments": aligned_result.get("word_segments", []),
            })

        # ── Этап 5/7: Диаризация ──────────────────────────────────────
        if ckpt5:
            print(f"\nЭтап 5/7: Диаризация — [из кэша]")
            segments = ckpt5["segments"]
        elif DIARIZE and HF_TOKEN:
            try:
                segments = run_diarization(
                    audio, aligned_result, all_segments, placeholders_list,
                    cfg, DEVICE, HF_TOKEN
                )
                _save_ckpt(WORK_DIR, base_name, 5, {"segments": segments})
            except Exception as e:
                print(f"   ОШИБКА диаризации: {e} — продолжаем без спикеров")
                segments = all_segments
                free_gpu()
        else:
            print(f"\nЭтап 5: Диаризация пропущена")
            segments = all_segments

        # Аудио больше не нужно — освобождаем память
        if audio is not None:
            del audio
            free_gpu()
            audio = None

        # ── Этап 6/7: Аналитика ───────────────────────────────────────
        if ckpt6:
            print(f"\nЭтап 6/7: Аналитика — [из кэша]")
            analytics = ckpt6["analytics"]
        else:
            print(f"\nЭтап 6/7: Аналитика")
            # analytics нужен audio только для SNR — загружаем снова если нужно
            _audio_for_analytics, _ = load_audio(file_path)
            analytics = run_all_analytics(_audio_for_analytics, segments, n_raw, n_removed, pass2_log, sr=16000)
            del _audio_for_analytics
            _save_ckpt(WORK_DIR, base_name, 6, {"analytics": analytics})

        # ── Этап 7/7: GigaChat LLM-анализ ────────────────────────────
        if ckpt7:
            print(f"\nЭтап 7/7: GigaChat LLM-анализ — [из кэша]")
            llm_result = ckpt7["llm_result"]
        elif RUN_LLM_ANALYSIS and gigachat_client:
            print(f"\nЭтап 7/7: GigaChat LLM-анализ")
            transcript_txt = format_output_txt(segments, file_name=file_name)
            llm_result = run_gigachat_analysis(
                gigachat_client, transcript_txt, analytics, cfg,
                segments=segments, prompts=prompts
            )
            _save_ckpt(WORK_DIR, base_name, 7, {"llm_result": llm_result})
        else:
            llm_result = None
            print(f"\nЭтап 7: LLM-анализ пропущен")

        # ── Сохранение: 2 файла ──────────────────────────────────────
        print(f"\nСохранение...")
        transcript_path = os.path.join(WORK_DIR, f"{base_name}_transcript.txt")
        transcript_content = format_output_txt(segments, file_name=file_name)
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_content)
        print(f"   {os.path.basename(transcript_path)} ({os.path.getsize(transcript_path)/1024:.0f} KB)")

        metrics_path = os.path.join(WORK_DIR, f"{base_name}_metrics.json")
        metrics_data = {
            "file": file_name,
            "pipeline_version": "v6",
            "audio_duration_sec": round(audio_duration, 1),
            "processing_sec": round(time.time() - total_start, 1),
            "settings": {
                "model": WHISPER_MODEL,
                "no_speech_threshold": cfg.get("no_speech_threshold"),
                "silero_vad": USE_SILERO_VAD,
                "noise_reduce": USE_NOISE_REDUCE,
                "diarize_model": cfg.get("diarize_model"),
                "min_speakers": cfg.get("min_speakers"),
                "max_speakers": cfg.get("max_speakers"),
            },
            "analytics": analytics,
            "llm_analysis": llm_result,
            "human_review": {
                "reviewed_by": None,
                "reviewed_at": None,
                "corrections": None,
                "notes": None,
            },
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2, default=str)
        print(f"   {os.path.basename(metrics_path)} ({os.path.getsize(metrics_path)/1024:.0f} KB)")

        saved_files = [transcript_path, metrics_path]

        # ── Загрузка на Яндекс.Диск ──
        if UPLOAD_RESULTS and YANDEX_TOKEN:
            try:
                upload_results(
                    YANDEX_TOKEN, OUTPUT_FOLDER, base_name, saved_files,
                    analytics, llm_result, audio_duration,
                    time.time() - total_start, file_name
                )
            except Exception as e:
                print(f"   Яндекс.Диск: {str(e)[:100]}")

        # ── Прогресс + очистка чекпоинтов ──
        remote_key = vf.get('remote_path') or file_name
        progress["transcribed"].append(remote_key)
        save_progress(progress, progress_file)
        _clear_ckpt(WORK_DIR, base_name)

        # ── Итоговый отчёт ──
        total_elapsed = time.time() - total_start
        stats = compute_stats([s for s in segments if not s.get("_is_placeholder")])
        total_speech = sum(s["duration"] for s in stats.values()) if stats else 0

        print(f"\n{'─'*55}")
        print(f"{file_name}: {total_elapsed:.0f}с ({total_elapsed/60:.1f} мин)")
        print(f"   Аудио: {audio_duration/60:.1f} мин | RTF: {total_elapsed/audio_duration:.2f}x")
        print(f"   Сегментов: {n_raw} raw -> {len(segments)} final")
        print(f"   Шумность: {analytics['noise']['score']}/10 ({analytics['noise']['verdict']})")
        print(f"   Баланс: учитель {analytics['balance']['teacher_pct']:.0f}% | ученики {analytics['balance']['students_pct']:.0f}%")
        print(f"   Вовлечённость: {analytics['engagement']['score']}/10 ({analytics['engagement']['verdict']})")
        if llm_result:
            assessment = llm_result.get("assessment") or llm_result.get("report")
            if isinstance(assessment, dict):
                score = llm_result.get("total_score", "?")
                topic = llm_result.get("lesson_topic", "?")
                print(f"   GigaChat: {score}/14 | Тема: {topic}")
        if stats:
            print(f"\nСпикеры:")
            for sp, st in sorted(stats.items(), key=lambda x: -x[1]["duration"]):
                pct = st["duration"] / total_speech * 100 if total_speech > 0 else 0
                wpm = analytics.get("speech_tempo", {}).get(sp, {}).get("wpm", "?")
                print(f"   {sp}: {st['count']} реплик, {st['duration']:.0f}с ({pct:.0f}%), {wpm} сл/мин")

        # Освобождаем память
        if MODE == "yadisk":
            try:
                os.remove(file_path)
            except Exception:
                pass

        try:
            del result, segments, aligned_result
        except NameError:
            pass
        free_gpu()

    print(f"\n{'='*65}")
    print(f"ГОТОВО: {len(video_files)} видео обработано")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
