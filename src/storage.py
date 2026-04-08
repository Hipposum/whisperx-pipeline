"""
storage.py — работа с Яндекс Диском: сканирование, скачивание видео,
             загрузка результатов, прогресс, промпты.
"""

import os
import gc
import json
import shutil
import time
import zipfile
from pathlib import Path


# ─────────────────────────────────────────────
# Вспомогательные функции ZIP
# ─────────────────────────────────────────────

def extract_videos_from_zip(zip_path, dest_dir, video_extensions, zip_extensions):
    """Распаковывает ZIP и возвращает список видеофайлов (рекурсивно)."""
    extracted = []
    print(f"   Распаковка {os.path.basename(zip_path)}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                name = os.path.basename(member)
                if not name:
                    continue
                ext = os.path.splitext(name)[-1].lower()
                if ext in video_extensions:
                    out_path = os.path.join(dest_dir, name)
                    counter = 1
                    base, e = os.path.splitext(out_path)
                    while os.path.exists(out_path):
                        out_path = f"{base}_{counter}{e}"
                        counter += 1
                    with zf.open(member) as src, open(out_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(out_path)
                    print(f"      {name}")
                elif ext in zip_extensions:
                    tmp_zip = os.path.join(dest_dir, name)
                    with zf.open(member) as src, open(tmp_zip, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.extend(extract_videos_from_zip(tmp_zip, dest_dir, video_extensions, zip_extensions))
                    os.remove(tmp_zip)
    except zipfile.BadZipFile as e:
        print(f"   Повреждённый ZIP: {e}")
    return extracted


# ─────────────────────────────────────────────
# Прогресс
# ─────────────────────────────────────────────

def load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"transcribed": []}


def save_progress(progress, progress_file):
    tmp = progress_file + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    os.replace(tmp, progress_file)


# ─────────────────────────────────────────────
# Сканирование Яндекс Диска
# ─────────────────────────────────────────────

SCAN_DEPTH = 5


def scan_yadisk_folder(yd, folder_path, extensions, max_count, skip_paths=None, depth=0):
    """
    Рекурсивно сканирует папку Яндекс Диска и собирает файлы с нужными расширениями.
    Возвращает список dicts: {path, name, size, is_zip}.
    """
    skip_paths = set(skip_paths or [])
    videos = []

    def _scan(folder, d):
        if d > SCAN_DEPTH or len(videos) >= max_count:
            return
        try:
            items = list(yd.listdir(folder, fields=["name", "path", "type", "size"], limit=500))
        except Exception as e:
            print(f"   {folder}: {e}")
            return
        for item in items:
            if len(videos) >= max_count:
                return
            if item.type == "file":
                ext = os.path.splitext(item.name)[-1].lower()
                if ext in extensions and item.path not in skip_paths:
                    videos.append({
                        "path": item.path, "name": item.name,
                        "size": item.size or 0, "is_zip": ext == ".zip"
                    })
        for d_item in [i for i in items if i.type == "dir"]:
            if len(videos) >= max_count:
                return
            _scan(d_item.path, d + 1)

    _scan(folder_path, depth)
    return videos[:max_count]


# ─────────────────────────────────────────────
# Скачивание видео
# ─────────────────────────────────────────────

def resolve_to_video_files(raw_files, download_dir, video_extensions, zip_extensions):
    """Распаковывает ZIP-файлы и возвращает плоский список видео."""
    result = []
    for vf in raw_files:
        local_path = vf["local_path"]
        ext = os.path.splitext(local_path)[-1].lower()
        if ext in zip_extensions:
            print(f"\nZIP: {vf['name']}")
            extracted = extract_videos_from_zip(local_path, download_dir, video_extensions, zip_extensions)
            os.remove(local_path)
            for ep in extracted:
                result.append({"local_path": ep, "remote_path": vf.get("remote_path"),
                                "name": os.path.basename(ep), "from_zip": vf["name"]})
            print(f"   Извлечено: {len(extracted)} видео")
        else:
            result.append(vf)
    return result


def download_videos(yandex_token, videos_folder, download_dir, work_dir,
                    max_videos, skip_already_transcribed,
                    video_extensions, zip_extensions):
    """
    Скачивает видео с Яндекс.Диска с учётом прогресса.
    Возвращает список dicts: {local_path, remote_path, name, from_zip?}.
    """
    import yadisk as yadisk_lib

    if not yandex_token:
        print("YANDEX_TOKEN не задан!")
        return []

    yd = yadisk_lib.YaDisk(token=yandex_token)
    print("Проверка токена...", end=" ")
    if not yd.check_token():
        print("Токен недействителен!")
        return []
    print("OK")

    all_accepted = video_extensions | zip_extensions
    progress_file = os.path.join(work_dir, "_progress.json")

    # Загружаем прогресс с Яндекс.Диска
    try:
        progress_remote = f"{videos_folder}/transcription_progress.json"
        if yd.exists(progress_remote):
            yd.download(progress_remote, progress_file)
    except Exception:
        pass

    progress = load_progress(progress_file)
    skip = set(progress["transcribed"]) if skip_already_transcribed else set()

    print(f"Сканирование {videos_folder}...")
    found = scan_yadisk_folder(yd, videos_folder, all_accepted, max_videos, skip)
    batch = found[:max_videos]
    print(f"   Найдено: {len(found)} | В запуске: {len(batch)} | Пропущено: {len(skip)}")
    if not batch:
        return []

    local_files = []
    for i, video in enumerate(batch, 1):
        size_mb = video['size'] / (1024 * 1024)
        tag = "ZIP" if video.get("is_zip") else "video"
        print(f"\n[{i}/{len(batch)}] {tag} {video['name']} ({size_mb:.0f} MB)")
        local_path = os.path.join(download_dir, video['name'])
        counter = 1
        base, ext = os.path.splitext(local_path)
        while os.path.exists(local_path):
            local_path = f"{base}_{counter}{ext}"
            counter += 1
        try:
            t0 = time.time()
            yd.download(video['path'], local_path)
            actual_size = os.path.getsize(local_path) / (1024 * 1024)
            print(f"   {actual_size:.0f} MB за {time.time()-t0:.0f}с")
            local_files.append({"local_path": local_path, "remote_path": video['path'],
                                 "name": video['name'], "is_zip": video.get("is_zip", False)})
        except Exception as e:
            print(f"   Ошибка: {str(e)[:100]}")

    return resolve_to_video_files(local_files, download_dir, video_extensions, zip_extensions)


# ─────────────────────────────────────────────
# Загрузка результатов на Яндекс Диск
# ─────────────────────────────────────────────

def upload_results(yandex_token, output_folder, base_name, saved_files,
                   analytics, llm_result, audio_duration, proc_time, file_name):
    """
    Загружает результаты транскрибации и метрики на Яндекс Диск.
    """
    import yadisk as yadisk_lib

    yd = yadisk_lib.YaDisk(token=yandex_token)
    video_folder = f"{output_folder}/{base_name}"

    # Основные файлы
    for sf in saved_files:
        remote = f"{video_folder}/{os.path.basename(sf)}"
        try:
            if yd.exists(remote):
                yd.remove(remote)
            yd.upload(sf, remote)
            print(f"   -> {remote}")
        except Exception as e:
            print(f"   Ошибка загрузки {os.path.basename(sf)}: {e}")

    # Раздельные файлы метрик
    _upload_metrics(yd, output_folder, base_name, analytics, llm_result,
                    audio_duration, proc_time, file_name)


def _upload_metrics(yd, output_folder, base_name, analytics, llm_result,
                    audio_duration, proc_time, file_name):
    """Сохраняет метрики в JSON-файлы и загружает на Яндекс Диск."""
    metrics_folder = f"{output_folder}/{base_name}/metrics"
    local_metrics_dir = f"/kaggle/working/results/{base_name}/metrics"
    os.makedirs(local_metrics_dir, exist_ok=True)

    try:
        if not yd.exists(metrics_folder):
            yd.mkdir(metrics_folder)
    except Exception:
        pass

    sections = {
        "meta":                  {"file": file_name, "audio_duration_sec": round(audio_duration, 1),
                                  "processing_sec": round(proc_time, 1)},
        "lesson_info":           analytics.get("lesson_info", {}),
        "transcription_quality": analytics.get("transcription_quality", {}),
        "noise":                 analytics.get("noise", {}),
        "speech_tempo":          analytics.get("speech_tempo", {}),
        "pauses":                analytics.get("pauses", {}),
        "balance":               analytics.get("balance", {}),
        "fillers":               analytics.get("fillers", {}),
        "questions":             analytics.get("questions", {}),
        "engagement":            analytics.get("engagement", {}),
        "llm_timeline":          (llm_result or {}).get("timeline"),
        "llm_assessment":        (llm_result or {}).get("assessment"),
        "llm_comment":           (llm_result or {}).get("comment"),
        "llm_total_score":       (llm_result or {}).get("total_score"),
        "llm_lesson_topic":      (llm_result or {}).get("lesson_topic"),
    }

    saved = []
    for section_name, data in sections.items():
        if data is None:
            continue
        local_path  = f"{local_metrics_dir}/{base_name}_{section_name}.json"
        remote_path = f"{metrics_folder}/{base_name}_{section_name}.json"
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        try:
            if yd.exists(remote_path):
                yd.remove(remote_path)
            yd.upload(local_path, remote_path)
            saved.append(section_name)
        except Exception as e:
            print(f"   Не удалось загрузить {section_name}: {e}")

    print(f"   Метрики ({len(saved)} файлов) -> {metrics_folder}/")
    return local_metrics_dir


# Промпты хранятся в репозитории GitHub (prompts/)
# Используй src/llm.py → load_prompts_from_files()
