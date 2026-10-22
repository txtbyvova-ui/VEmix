# VEmix 3.0 — аудит архитектуры и план реализации (Этап A)

ТЗ: [`ТЗ/task_02.md`](../ТЗ/task_02.md). Документ — результат этапа A: что есть в репозитории,
куда и как встраивается 3.0, какие файлы меняются, какие решения по зависимостям приняты и почему.
Аудит выполнен параллельно (8 читателей подсистем + 3 исследования внешних источников + критик
полноты, 32 утверждения проверены по коду), план показан до кода этапов B–F.

## 1. Итоги аудита

| Подсистема | Что есть (2.0) | Что важно для 3.0 |
|---|---|---|
| `analysis.py` + `cache.py` | BPM/key (алгоритмы v1, окна закреплены тестами бит-в-бит), эвристика энергии (окно 60 с, уверенность 0.4), probe, pass-1 громкость; кэш с независимыми **секциями** (`version` + хэш `params`), отпечаток файла, read-only в dry-run | Сегментный анализ — **новая секция** `segments` и **отдельный** декод (общий декод сдвинул бы BPM/ключи); `ANALYSIS_CACHE_SCHEMA_VERSION` не трогать (иначе кэш уйдёт в `.corrupt-N`, ключи переопределятся — правило памяти проекта); новые предупреждения анализа не писать в `review_reasons` (иначе WARNING в QC) |
| `models.py` | `TrackFeatures.feature_sources` + `confidence()`; **любой незнакомый источник получает 0.8** — больше, чем `id3` (0.6) и `auto:` (0.5) | Внешние факты и AI — **отдельные** структуры (`TrackContext`), не поля `TrackFeatures`; новые поля датаклассов — только в конец со значением по умолчанию (`LibraryAnalysis`, `TrackQC` создаются позиционно) |
| `sequencing.py` | Beam + локальный ремонт + точная проверка выполнимости, `transition_score` (5 компонентов, уверенности, стягивание к нейтрали) и векторизованный `_FeatureMatrix` (равенство закреплено тестом); кривая — 2 точки по **индексу**; якорей нет (только префикс «openers») | Планировщик — **новый модуль** поверх `transition_score`/`_FeatureMatrix`; `build_sequence`, веса и формулировки флагов 2.0 не трогать (их парсят QC и тесты) |
| `curation.py` | Строгий загрузчик профиля (опечатка → ошибка с подсказкой, дубли ключей), `curate()`, правило списков по частям артиста | Общий ридер YAML/JSON для профиля и сценария; якорь сравнивается по **всему** исполнителю (`Fire` ≠ `Earth, Wind & Fire`); якорь, исключённый профилем, — ошибка с причиной (QC перепроверяет curate) |
| `cli.py` + `pipeline.py` | `main()` ищет FFmpeg до любой команды; argparse допускает сокращения опций; `ValueError`→2, `OSError`→5 | Новые подкоманды — `allow_abbrev=False`, своя диспетчеризация; `compare` без FFmpeg; сетевые ошибки ловятся внутри модулей |
| `playlist.py` / рендер микса / `quality_control.py` | Экспорт принимает готовый `SequenceResult`; `render_m3u8` отвергает абсолютные пути и `..`; неизвестный `kind` флага и пропущенная неизвестная проверка → WARNING; подстрока «energy jump» считается подозрительным переходом | Рендер плана — те же функции **без пересортировки**; раздел `manifest.plan` пишется только `render` (у 2.0 его нет вовсе); запланированный контраст — не флаг; экспорт плана — свой писатель M3U |
| Тесты | 629 тестов (98 с реальным FFmpeg), ≈140 с | Всё новое тестируется без сети: подменяемый HTTP-транспорт, фейковые AI-клиенты |

Найденные расхождения документации с кодом (`docs/ARCHITECTURE.md`: `put()` возвращает `bool`,
`detect_key` → `float | None`, `plan_transition(fx_history=)`, `write_mix_tracklist(mix_name=)`,
модель памяти рендера, `scan_silence`) исправлены.

## 2. Главный принцип: категории сведений

| Категория | Источник (provenance) | Пример |
|---|---|---|
| measured — измерено из файла | `auto:<algo-version>` | BPM, тональность, энергия, кривая интенсивности |
| file tags — теги файла / имя | `id3`, `filename` | жанр из тега |
| curator — куратор | `manual` | настроение, энергия, роль (`role:<роль>` в `vibe_tags`) |
| external — внешний источник | `research:<source>` + URL + время + поля + лицензия | год, релиз, ISRC, MBID |
| derived — правило над измерениями | `rules:interpret-v2` | роль «breather» по энергии и траектории |
| ai — интерпретация модели | `ai:<provider>/<model>` + уверенность (≤ 0.8) | настроение «ominous» 0.6 |

Неоднозначное внешнее совпадение не применяется. Теги/жанры MusicBrainz — пользовательские
данные под **CC BY-NC-SA 3.0** (некоммерческие): хранятся отдельно и влияют на план только по
явному `--use-community-tags`. AI получает только измеренное, проверенные описательные факты и
данные куратора; теги сообщества (теги/жанры MusicBrainz и правимые сообществом жанры Wikidata)
уходят модели лишь по явному `interpret --use-community-tags` — отдельным блоком
`community_tags` с пометкой `verified: false` и примечанием о лицензиях (у `plan` тот же флаг
выбирает закэшированные ответы, полученные с тегами). Утверждение без ссылки на реально
переданное поле отбрасывается валидатором.

## 3. Архитектура 3.0

```
library ─► analysis (2.0, без изменений) ─► TrackFeatures
              └─► audio_intel.analyze_segments  (секция кэша "segments", по запросу)
research/ (enrich)  ─► veg_research_cache.json  (online | cache | off)
ai/ (interpret)     ─► veg_ai_cache.json        (anthropic SDK | openai-compatible | fake)
scenario.load_scenario ─► resolve ─► dramaturgy.plan (цель, жёсткое > мягкое, K вариантов)
   └─► plan_report (dry-run, compare, plan.json, M3U8, CSV) ─► render (2.0 экспорт) ─► qc
```

### Новые модули

| Файл | Этап | Назначение |
|---|---|---|
| `veg_mixer/scenario.py` | B | Модель и строгий загрузчик сценария; ссылки на треки; `song_key` |
| `veg_mixer/dramaturgy.py` | B/E | Контекст трека с provenance, целевая функция, beam + локальный поиск, нарушения, альтернативы, объяснения и контрфакты |
| `veg_mixer/plan_report.py` | B | Dry-run отчёт, сравнение, план-файл `vegroove.plan`, M3U8/CSV |
| `veg_mixer/plan_pipeline.py` | B–E | Оркестрация `enrich/interpret/suggest/plan/compare/render` |
| `veg_mixer/audio_intel.py` | B | Сегментный анализ: интенсивность во времени, уровень, ударные/гармония (HPSS), бас, спектр, плотность, темп + half/double-time, структура (новизна Фута), прокси вокала, тембр |
| `veg_mixer/moods.py` | B/D | Словарь настроений (валентность/возбуждение) для сценариев, планировщика и AI |
| `veg_mixer/interpret_rules.py` | B | Роли из измерений (derived), без AI и сети |
| `veg_mixer/net.py` | C | HTTP на stdlib: таймауты, повторы с `Retry-After`, лимит частоты по хосту, ограничение размера, ключи не логируются |
| `veg_mixer/research/` | C | `models`, `cache`, `matching`, `musicbrainz`, `wikidata`, `discogs`, `enrich` |
| `veg_mixer/ai/` | D | `schema` (строгая схема + валидатор доказательств), `providers` (Anthropic SDK, OpenAI-совместимый, fake), `cache`, `interpret` |
| `veg_mixer/scenario_suggest.py` | D | AI-черновик сценария (`suggest`): обезличенная сводка библиотеки (без путей, имён файлов, треков и артистов), проверка ответа по `SUGGESTION_SCHEMA`, починка типов с журналом исправлений, валидация штатным `scenario.parse_scenario`; не больше одного повторного запроса, затем `ScenarioError` (провайдер так и не ответил — `SuggestProviderError`); план по-прежнему строит детерминированный планировщик |
| `scenarios/halloween_special.yaml` | F | Пример сценария |
| `examples/make_halloween_demo.py` | F | Синтетическая демо-библиотека под сценарий |

### Изменяемые файлы (точечно, обратно совместимо)

| Файл | Изменение |
|---|---|
| `veg_mixer/cli.py` | Подкоманды `enrich`, `interpret`, `plan`, `suggest`, `compare`, `render`; флаг `analyze --segments`; опции 2.0 не тронуты |
| `veg_mixer/pipeline.py` | `analyze_library(segments="off")`, `run_analyze(segments=False)`, `_render_one_mix(plan=None)`; `LibraryAnalysis.segments` в конце |
| `veg_mixer/curation.py` | Общий ридер `load_config_document`; `_Parser(error=)`; публичные `track_match_keys`/`reference_keys` |
| `veg_mixer/models.py` | `PlaylistManifest.plan` (сериализуется только если задан) |
| `veg_mixer/sequencing.py` | `ROLE_TAG_PREFIX = "role:"`: кураторские теги ролей 3.0 (`role:*`) исключены из Jaccard-сходства тегов (`_tags_of`) и из меток соответствия дейпарту (`_labels`), чтобы CSV куратора, подготовленный для 3.0, не менял порядок плейлистов и миксов 2.0; `build_sequence`, веса и формулировки флагов не тронуты |
| `veg_mixer/playlist.py`, `quality_control.py` | `export_playlist(plan=None)`; проверки `plan_order`/`plan_anchors` только при `manifest.plan` |
| `veg_mixer/__init__.py`, `pyproject.toml` | Версия 3.0.0 |
| `README.md`, `docs/ARCHITECTURE.md`, `CHANGES.md`, `.gitignore` | Документация, новые кэши |

## 4. Решения по зависимостям

* **Новых обязательных зависимостей нет.** Аудио — librosa/numpy/scipy; HTTP — `urllib`
  (`requests` есть только транзитивно; `musicbrainzngs` не поддерживается с 2020–2022 и прячет
  исходный JSON, нужный для provenance).
* **Claude** — официальный SDK `anthropic` (опционально, `requirements-ai.txt`, точный пин);
  модель по умолчанию `claude-opus-5-5`, structured outputs (`output_config.format`), серверный
  fallback при отказе. Без пакета/ключа — понятная ошибка, остальная система работает.
* **Локальные модели** (Ollama, LM Studio, llama.cpp) — OpenAI-совместимый провайдер на stdlib.
* **Тяжёлые ML-модели не добавлены**: Demucs (torch, GPU для приемлемой скорости),
  Essentia-TF (AGPL + модели CC BY-NC-ND), CLAP/PANNs/openl3 (сотни МБ весов) — несоразмерно
  выигрышу для фонового плейлиста; интерфейс ролей/настроений позволяет подключить их позже.
* **Не реализованы** (обоснование в README): Bandcamp (нет API, скрейпинг запрещён), Last.fm
  (некоммерческие условия), AcoustID (отправка отпечатка аудио требует согласия; платный план).

## 5. Этапы и критерии

| Этап | Содержание | Проверка |
|---|---|---|
| B | Сценарии, якоря, кривая, dry-run без AI и интернета; сегментный анализ; роли по правилам | Тесты сценария, планировщика (якоря, запреты, длительность, кривая, детерминизм, проверка плана), отчёта; весь набор 2.0 зелёный |
| C | `net.py`, `research/*`, `enrich` (online / cache / off), кэш с TTL, лимит частоты | Подменённый транспорт: совпадение, неоднозначность (не применяется), противоречия, 429/503/таймаут/битый JSON, cache-only без сети |
| D | `ai/*`, `interpret`, кэш ответов, фолбэк на правила | Валидатор схемы и доказательств, re-ask, отрицательный кэш, circuit breaker, фейковые SDK-клиенты |
| E | Факты/интерпретации в цели, альтернативы, `compare`, `render`, QC плана | Сквозной CLI-тест с FFmpeg: analyze → plan → compare → render (playlist/mix) → qc; устаревший план |
| F | Документация, Halloween Special, PowerShell-команды, ревью со скептической проверкой | Полный прогон тестов |
