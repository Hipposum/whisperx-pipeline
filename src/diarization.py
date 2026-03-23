"""
diarization.py — диаризация спикеров через pyannote (whisperx.DiarizationPipeline).
"""

import time
from .utils import free_gpu


def run_diarization(audio, aligned_result, all_segments, placeholders_list, config, device, hf_token):
    """
    Запускает диаризацию pyannote и присваивает спикеров сегментам.
    Возвращает итоговый список сегментов с полем 'speaker'.
    """
    DIARIZE_MODEL = config["diarize_model"]
    MIN_SPEAKERS  = config.get("min_speakers")
    MAX_SPEAKERS  = config.get("max_speakers")

    print(f"\nЭтап 5/7: Диаризация ({DIARIZE_MODEL})")
    print(f"   Спикеров: min={MIN_SPEAKERS}, max={MAX_SPEAKERS}")
    t0 = time.time()

    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    diarize_model = DiarizationPipeline(
        model_name=DIARIZE_MODEL, token=hf_token, device=device
    )
    diarize_kwargs = {}
    if MIN_SPEAKERS is not None:
        diarize_kwargs["min_speakers"] = MIN_SPEAKERS
    if MAX_SPEAKERS is not None:
        diarize_kwargs["max_speakers"] = MAX_SPEAKERS

    diarize_raw = diarize_model(audio, **diarize_kwargs)

    real_segs = [s for s in all_segments if not s.get("_is_placeholder")]
    assign_result = {
        "segments": real_segs,
        "word_segments": aligned_result.get("word_segments", [])
    }
    assign_result = assign_word_speakers(diarize_raw, assign_result)
    segments = sorted(
        assign_result["segments"] + placeholders_list,
        key=lambda s: s["start"]
    )
    n_speakers = len(set(s.get("speaker", "?") for s in assign_result["segments"]))
    print(f"   {n_speakers} спикеров | {time.time()-t0:.0f}с")

    del diarize_model
    free_gpu()

    return segments
