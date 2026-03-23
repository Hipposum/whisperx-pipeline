# WhisperX Pipeline v6

Автоматическая транскрибация видео уроков с диаризацией спикеров, аналитикой качества
и LLM-анализом через GigaChat. Оптимизирован для запуска на Kaggle (GPU T4/P100).

---

## Архитектура

```
whisperx-pipeline/
├── config/
│   └── settings.yaml          # все настройки (кроме секретов)
├── prompts/
│   ├── timeline_system.txt    # системный промпт для таймлайна урока
│   ├── timeline_user.txt      # пользовательский промпт (с {transcript})
│   ├── report_system.txt      # системный промпт для оценки качества
│   └── report_user.txt        # пользовательский промпт (с {transcript} и {metrics})
├── src/
│   ├── pipeline.py            # точка входа, оркестратор 7 этапов
│   ├── transcription.py       # Whisper Pass1/Pass2 + Silero VAD + noisereduce
│   ├── diarization.py         # pyannote диаризация + присвоение спикеров
│   ├── analytics.py           # метрики: шум, темп, паузы, баланс, вовлечённость
│   ├── llm.py                 # GigaChat: инициализация, compress, запросы
│   ├── storage.py             # Яндекс Диск: скачивание, загрузка, прогресс
│   └── utils.py               # GPU, форматирование, галлюцинации, merge
├── notebooks/
│   └── kaggle_runner.ipynb    # минимальный ноутбук для запуска на Kaggle
└── requirements.txt
```

### Поток данных

```
Видео (Яндекс Диск / локально)
    |
    v
[Этап 1]  Загрузка аудио (whisperx.load_audio, 16kHz mono)
    |
    v
[Этап 1б] Шумоподавление (noisereduce) + нормализация амплитуды
    |
    v
[Этап 2]  Транскрибация Pass 1 (WhisperX large-v3, Silero VAD)
    |
    v
[Этап 3]  Пост-обработка: удаление галлюцинаций, поиск проблемных зон,
          Pass 2 — повторная транскрибация с мягкими параметрами
    |
    v
[Этап 4]  Alignment — выравнивание слов по времени (wav2vec2)
    |
    v
[Этап 5]  Диаризация спикеров (pyannote/speaker-diarization-3.1)
    |
    v
[Этап 6]  Аналитика (шум, темп, паузы, баланс, слова-паразиты, вопросы)
    |
    v
[Этап 7]  GigaChat LLM: таймлайн урока + оценка качества (JSON)
    |
    v
Результаты: {base}_transcript.txt + {base}_metrics.json → Яндекс Диск
```

---

## Быстрый старт (Kaggle)

### 1. Подготовка репозитория (GitHub)

```bash
git clone https://github.com/Hipposum/whisperx-pipeline.git
cd whisperx-pipeline
# Отредактируйте config/settings.yaml под свои нужды
# Отредактируйте prompts/*.txt под свои промпты
git add -A && git commit -m "config" && git push
```

### 2. Секреты в Kaggle

Перейдите: **Add-ons → Secrets** и добавьте:

| Ключ | Описание |
|------|----------|
| `HF_TOKEN` | Hugging Face токен (для pyannote диаризации) |
| `GIGACHAT_CREDENTIALS` | Base64 ключ GigaChat API |
| `YANDEX_TOKEN` | OAuth токен Яндекс Диска |

### 3. Запуск ноутбука

Откройте `notebooks/kaggle_runner.ipynb` на Kaggle и выполните ячейки по порядку:

- **Ячейка 1** — установка зависимостей + клонирование репозитория
- **Ячейка 2** — загрузка секретов в переменные среды
- **Ячейка 3** — запуск пайплайна

---

## Конфигурация

### config/settings.yaml

Основной файл настроек. Редактируйте перед пушем на GitHub:

```yaml
# Откуда берём видео
mode: "yadisk"                  # "yadisk" или "local"
videos_folder: "/PachcaVideos"  # папка на Яндекс Диске
max_videos: 5                   # сколько видео обработать за один запуск

# Модель Whisper
whisper_model: "large-v3"
language: "ru"
batch_size: 4                   # T4: 4, P100: 8

# Шумоподавление
use_noise_reduce: true
nr_prop_decrease: 0.4           # 0.0 = нет, 1.0 = максимум

# Диаризация
diarize: true
diarize_model: "pyannote/speaker-diarization-3.1"
min_speakers: null              # null = авто
max_speakers: null

# LLM-анализ
run_llm_analysis: true
gigachat_model: "GigaChat"
```

Полный список параметров с описаниями — в самом файле `config/settings.yaml`.

### Структура папок на Яндекс Диске

Яндекс Диск используется **только для видео и результатов**.
Промпты и настройки хранятся в репозитории GitHub.

```
/PachcaVideos/          <- входные видео (videos_folder)
/settings/
    output/             <- результаты (output_folder)
        {base_name}/
            {base_name}_transcript.txt
            {base_name}_metrics.json
            metrics/
                {base_name}_noise.json
                {base_name}_balance.json
                ...
```

---

## Редактирование промптов

Промпты хранятся в репозитории в папке `prompts/`. При каждом запуске Kaggle клонирует
репо и получает актуальные версии.

```bash
# Отредактировал промпт → запушил → на следующем запуске Kaggle уже видит изменения
vim prompts/report_user.txt
git add prompts/ && git commit -m "update prompt" && git push
```

### Шаблоны переменных в промптах

- `timeline_user.txt` — доступна переменная `{transcript}` (сжатый транскрипт)
- `report_user.txt` — доступны переменные `{transcript}` и `{metrics}` (JSON с метриками)

---

## Описание этапов пайплайна

| Этап | Модуль | Описание |
|------|--------|----------|
| 1    | `transcription.py` | Загрузка аудио через whisperx (16kHz mono) |
| 1б   | `transcription.py` | Шумоподавление noisereduce + нормализация |
| 2    | `transcription.py` | Транскрибация WhisperX + Silero VAD |
| 3    | `utils.py` + `transcription.py` | Фильтрация галлюцинаций, Pass 2 |
| 4    | `transcription.py` | Выравнивание слов wav2vec2 |
| 5    | `diarization.py` | Диаризация pyannote 3.1 |
| 6    | `analytics.py` | Метрики качества урока |
| 7    | `llm.py` | GigaChat: таймлайн + оценка |

### Метрики аналитики (Этап 6)

- **noise** — оценка шумности (SNR, noise floor), вердикт 0–10
- **speech_tempo** — темп речи по спикерам (слов/мин)
- **pauses** — количество и длительность пауз, самая длинная пауза
- **balance** — процент времени учителя vs учеников, оценка баланса
- **fillers** — слова-паразиты ("ну", "вот", "как бы" и др.), топ-3 по спикеру
- **questions** — количество вопросов (педагогические / студенческие / риторические)
- **engagement** — оценка вовлечённости учеников 0–10

---

## Лицензия

MIT
