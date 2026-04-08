"""
transcription.py — загрузка аудио, шумоподавление, транскрибация (Whisper Pass 1 + Silero VAD),
                   пост-обработка и повторная транскрибация проблемных зон (Pass 2).
"""

import time
import torch
import whisperx
import noisereduce as nr

from .utils import free_gpu, gpu_mem, is_hallucination


def load_silero_vad(device, threshold, min_speech_ms, min_silence_ms):
    """Загружает модель Silero VAD. Возвращает (model, utils) или (None, None) при ошибке."""
    print("Загрузка Silero VAD...")
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )
        model = model.to(device)
        print("   Silero VAD загружен")
        return model, utils
    except Exception as e:
        print(f"   Silero VAD не загружен: {e} — используется встроенный VAD")
        return None, None


def load_audio(file_path):
    """Загружает аудио через whisperx. Возвращает (audio_array, duration_sec)."""
    audio = whisperx.load_audio(file_path)
    audio_duration = len(audio) / 16000
    return audio, audio_duration


def apply_noise_reduction(audio, nr_prop_decrease, nr_stationary):
    """Применяет noisereduce и нормализует амплитуду."""
    import numpy as np
    rms_before = float(np.sqrt(np.mean(audio ** 2)))
    print(f"   Аудио до шумоподавления: RMS={rms_before:.5f}, max={float(np.max(np.abs(audio))):.5f}")
    try:
        audio_reduced = nr.reduce_noise(
            y=audio, sr=16000,
            stationary=nr_stationary,
            prop_decrease=nr_prop_decrease
        )
        rms_after = float(np.sqrt(np.mean(audio_reduced ** 2)))
        peak = float(np.max(np.abs(audio_reduced)))
        print(f"   После шумоподавления: RMS={rms_after:.5f}, max={peak:.5f}")
        if peak < 1e-6:
            print("   ПРЕДУПРЕЖДЕНИЕ: сигнал после шумоподавления почти нулевой — отключаем шумоподавление!")
            return audio  # возвращаем оригинал
        audio = audio_reduced / (peak + 1e-9)
        print(f"   Шумоподавление применено (stationary={nr_stationary}, prop_decrease={nr_prop_decrease})")
    except Exception as e:
        print(f"   Ошибка noisereduce: {e} — продолжаем без шумоподавления")
    return audio


def transcribe_pass1(audio, config, device, silero_model=None, silero_utils=None):
    """
    Транскрибация — Проход 1.
    Использует Silero VAD (если передан) или встроенный pyannote VAD.
    Возвращает (result, n_raw, elapsed).
    """
    WHISPER_MODEL = config["whisper_model"]
    COMPUTE_TYPE  = config["compute_type"]
    LANGUAGE      = config["language"]
    BATCH_SIZE    = config["batch_size"]
    CHUNK_LENGTH  = config["chunk_length"]
    BEAM_SIZE     = config["beam_size"]
    WORD_TIMESTAMPS         = config["word_timestamps"]
    CONDITION_ON_PREVIOUS   = config["condition_on_previous"]
    NO_SPEECH_THRESHOLD     = config["no_speech_threshold"]
    COMPRESSION_RATIO_THRESHOLD = config["compression_ratio_threshold"]
    TEMPERATURE_FALLBACK    = config.get("temperature_fallback", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    INITIAL_PROMPT          = config.get("initial_prompt", False)
    SILERO_THRESHOLD        = config["silero_threshold"]
    SILERO_MIN_SPEECH_MS    = config["silero_min_speech_ms"]
    SILERO_MIN_SILENCE_MS   = config["silero_min_silence_ms"]

    t0 = time.time()
    audio_duration = len(audio) / 16000

    asr_options = {
        "beam_size": BEAM_SIZE,
        "word_timestamps": WORD_TIMESTAMPS,
        "condition_on_previous_text": CONDITION_ON_PREVIOUS,
        "no_speech_threshold": NO_SPEECH_THRESHOLD,
        "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
    }
    if INITIAL_PROMPT:
        asr_options["initial_prompt"] = INITIAL_PROMPT
    if isinstance(TEMPERATURE_FALLBACK, (list, tuple)):
        asr_options["temperatures"] = list(TEMPERATURE_FALLBACK)

    vad_options = None
    if silero_model is not None and silero_utils is not None:
        print("   Silero VAD: детектирование речи...")
        try:
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = silero_utils
            audio_tensor = torch.FloatTensor(audio).to(device)
            with torch.no_grad():
                speech_timestamps = get_speech_timestamps(
                    audio_tensor, silero_model,
                    threshold=SILERO_THRESHOLD,
                    min_speech_duration_ms=SILERO_MIN_SPEECH_MS,
                    min_silence_duration_ms=SILERO_MIN_SILENCE_MS,
                    return_seconds=True
                )
            speech_ratio = sum(t['end'] - t['start'] for t in speech_timestamps) / audio_duration * 100
            print(f"   Silero VAD: {len(speech_timestamps)} речевых фрагментов | речь: {speech_ratio:.0f}%")
            if len(speech_timestamps) == 0:
                print("   Silero VAD не нашёл речь — переключаемся на pyannote VAD")
                vad_options = {"vad_onset": 0.01, "vad_offset": 0.02}
            else:
                vad_options = None  # Silero уже нашёл речевые зоны
        except Exception as e:
            print(f"   Silero VAD ошибка: {e} — переключаемся на pyannote VAD")
            vad_options = {"vad_onset": 0.05, "vad_offset": 0.1}
    else:
        vad_options = {"vad_onset": 0.05, "vad_offset": 0.1}
        print("   Используется встроенный pyannote VAD")

    model = whisperx.load_model(
        WHISPER_MODEL, device, compute_type=COMPUTE_TYPE,
        language=LANGUAGE, asr_options=asr_options,
        vad_options=vad_options
    )
    print(f"   Модель загружена | VRAM: {gpu_mem()}")

    result = model.transcribe(
        audio, batch_size=BATCH_SIZE, language=LANGUAGE,
        chunk_size=CHUNK_LENGTH, print_progress=True
    )
    n_raw = len(result["segments"])
    elapsed = time.time() - t0
    rtf = elapsed / audio_duration if audio_duration > 0 else 0
    print(f"\n   {n_raw} сегментов | {elapsed:.0f}с (RTF={rtf:.2f}x)")
    del model
    free_gpu()

    return result, n_raw, elapsed


def retranscribe_zones(audio, problem_zones, config, device, sr=16000):
    """Повторная транскрибация проблемных зон с более мягкими настройками (Pass 2)."""
    if not problem_zones:
        return [], []

    WHISPER_MODEL   = config["whisper_model"]
    COMPUTE_TYPE    = config["compute_type"]
    LANGUAGE        = config["language"]
    BATCH_SIZE      = config["batch_size"]
    PASS2_BEAM_SIZE      = config["pass2_beam_size"]
    PASS2_TEMPERATURE    = config["pass2_temperature"]
    PASS2_NO_SPEECH      = config["pass2_no_speech"]
    PASS2_COMPRESSION    = config["pass2_compression"]
    PASS2_INITIAL_PROMPT = config.get("pass2_initial_prompt", "")
    PASS2_PADDING_SEC    = config["pass2_padding_sec"]
    PASS2_MIN_ZONE_SEC   = config["pass2_min_zone_sec"]

    print(f"\n   Проход 2: перетранскрибация {len(problem_zones)} зон...")
    t0 = time.time()

    asr_options_p2 = {
        "beam_size": PASS2_BEAM_SIZE,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "no_speech_threshold": PASS2_NO_SPEECH,
        "compression_ratio_threshold": PASS2_COMPRESSION,
        "temperatures": [PASS2_TEMPERATURE],
    }
    if PASS2_INITIAL_PROMPT:
        asr_options_p2["initial_prompt"] = PASS2_INITIAL_PROMPT

    vad_options_p2 = {"vad_onset": 0.2, "vad_offset": 0.3}
    model_p2 = whisperx.load_model(
        WHISPER_MODEL, device, compute_type=COMPUTE_TYPE,
        language=LANGUAGE, asr_options=asr_options_p2,
        vad_options=vad_options_p2
    )

    recovered = []
    pass2_log = []
    audio_duration = len(audio) / sr

    for zone in problem_zones:
        z_start = max(0, zone["start"] - PASS2_PADDING_SEC)
        z_end = min(audio_duration, zone["end"] + PASS2_PADDING_SEC)
        z_dur = z_end - z_start
        if z_dur < PASS2_MIN_ZONE_SEC:
            continue

        chunk = audio[int(z_start * sr):int(z_end * sr)]
        try:
            result = model_p2.transcribe(
                chunk, batch_size=BATCH_SIZE, language=LANGUAGE,
                chunk_size=int(z_dur) + 1, print_progress=False
            )
            good_segs = []
            for seg in result.get("segments", []):
                text = seg.get("text", "").strip()
                if text and not is_hallucination(text):
                    seg["start"] = round(seg["start"] + z_start, 3)
                    seg["end"]   = round(seg["end"]   + z_start, 3)
                    good_segs.append(seg)
            if good_segs:
                recovered.extend(good_segs)
                pass2_log.append({"zone": f"{zone['start']:.1f}-{zone['end']:.1f}",
                                   "status": "recovered", "n": len(good_segs)})
            else:
                recovered.append({
                    "start": round(zone["start"], 3),
                    "end": round(zone["end"], 3),
                    "text": f"[неразборчиво — {zone['end'] - zone['start']:.1f}с]",
                    "speaker": "UNKNOWN", "_is_placeholder": True
                })
                pass2_log.append({"zone": f"{zone['start']:.1f}-{zone['end']:.1f}",
                                   "status": "unrecognized"})
        except Exception as e:
            recovered.append({
                "start": round(zone["start"], 3),
                "end": round(zone["end"], 3),
                "text": f"[неразборчиво — {zone['end'] - zone['start']:.1f}с]",
                "speaker": "UNKNOWN", "_is_placeholder": True
            })
            pass2_log.append({"zone": f"{zone['start']:.1f}-{zone['end']:.1f}",
                               "status": f"error: {str(e)[:50]}"})

    del model_p2
    free_gpu()
    n_ok = sum(1 for l in pass2_log if l["status"] == "recovered")
    n_fail = len(pass2_log) - n_ok
    print(f"   Проход 2: восстановлено {n_ok}, [неразборчиво] {n_fail} | {time.time()-t0:.0f}с")
    return recovered, pass2_log


def run_alignment(audio, segments_merged, config, device):
    """Выравнивание (wav2vec2). Возвращает (aligned_result, all_segments, placeholders_list)."""
    LANGUAGE                = config["language"]
    ALIGN_MODEL             = config.get("align_model")
    RETURN_CHAR_ALIGNMENTS  = config.get("return_char_alignments", False)
    INTERPOLATE_METHOD      = config.get("interpolate_method", "nearest")

    segs_for_align    = [s for s in segments_merged if not s.get("_is_placeholder")]
    placeholders_list = [s for s in segments_merged if s.get("_is_placeholder")]

    align_kwargs = {"language_code": LANGUAGE, "device": device}
    if ALIGN_MODEL:
        align_kwargs["model_name"] = ALIGN_MODEL

    model_a, metadata = whisperx.load_align_model(**align_kwargs)
    aligned_result = whisperx.align(
        segs_for_align, model_a, metadata, audio, device,
        return_char_alignments=RETURN_CHAR_ALIGNMENTS,
        interpolate_method=INTERPOLATE_METHOD
    )
    all_segments = sorted(
        aligned_result["segments"] + placeholders_list,
        key=lambda s: s["start"]
    )
    del model_a, metadata
    free_gpu()
    return aligned_result, all_segments, placeholders_list
