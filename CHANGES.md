# CHANGES

## VEmix 3.0 (2026-10-22)

Драматургическое планирование плейлиста по ТЗ `ТЗ/task_02.md` поверх Automix 2.0 (команды 2.0 без
новых аргументов работают как раньше; форматы кэша анализа, манифестов и отчётов 2.0 не менялись).

* **Сценарии** (`scenarios/*.yaml`, `veg_mixer/scenario.py`): якоря на позициях и в фазах,
  обязательные/исключённые треки, запрещённые и желательные соседства, фазы с кривой энергии по
  относительному времени, роли и настроения фаз, допуски темпа/тональности/скачка энергии,
  контролируемый контраст, длительность, разнообразие (артисты, одна версия песни), веса цели.
  Пример — Halloween Special.
* **Планировщик** (`veg_mixer/dramaturgy.py`): явная целевая функция из 15 нормированных слагаемых,
  жёсткие ограничения важнее мягких (лексикографически), beam-построение + детерминированный
  локальный поиск, альтернативы по пресетам весов со штрафом за похожесть, объяснения и
  контрфакты из реально вычисленных слагаемых, проверка сохранённого плана.
* **Отчёты и экспорт** (`veg_mixer/plan_report.py`): dry-run таблица (номер, фаза, трек, энергия,
  цель и отклонение, BPM, тональность, роль, оценка перехода, предупреждения), причины по
  позициям, сравнение вариантов, план-файл `vegroove.plan`, M3U8 и CSV.
* **Сегментный аудиоанализ** (`veg_mixer/audio_intel.py`, `analyze --segments`): интенсивность и
  уровень во времени, ударные/бас/спектр, плотность, траектория, темп и half/double-time,
  структура (интро/билд/пик/брейк/аутро — кандидаты), прокси вокала с низкой уверенностью.
  Новая секция кэша `segments`; BPM/тональность/энергия 2.0 не изменились.
* **Music Research Agent** (`veg_mixer/research/`, `enrich`): MusicBrainz (+ Wikidata для
  перекрёстной проверки, Discogs по токену), собственная оценка совпадения, неоднозначное не
  применяется, противоречия фиксируются, источник/время/поля/лицензия у каждого факта, кэш с TTL,
  режимы online / cache / off, лимит частоты. Теги MusicBrainz (CC BY-NC-SA) — только по флагу.
* **AI Track Understanding** (`veg_mixer/ai/`, `interpret`): подключаемые провайдеры (Claude через
  официальный SDK — опционально, OpenAI-совместимые локальные серверы, фейковый для тестов),
  строгая JSON-схема и валидатор доказательств, кэш ответов, повтор при невалидном ответе,
  предохранитель. Роли по правилам из измерений (`interpret_rules.py`) работают без AI.
* **Рендер плана** (`render`) — экспорт выбранного варианта существующими playlist/mix без
  пересортировки, раздел `manifest.plan`, QC `plan_order`/`plan_anchors`, отказ на устаревшем плане.
* `compare` работает без FFmpeg; новые подкоманды не допускают сокращений опций.
* Документация: `docs/VEMIX3_PLAN.md` (аудит и план), `docs/ARCHITECTURE.md` (контракты 3.0,
  исправлены расхождения 2.0 с кодом), README (команды, лицензии, PowerShell).

### Исправления по финальному ревью (50 подтверждённых находок)

* Планировщик: идентичность артиста/песни — объединение имени и MBID (частичный research больше
  не отключает правила разнообразия); метки куратора заменяют AI-метки (настроения, роли, подсказки
  позиций); авто-тональность никогда не становится жёстким лимитом (порог 0.6); авто-BPM, которому
  противоречит сегментный анализ, не даёт жёсткого лимита темпа; роль, которую не мог оценить ни один
  источник, — «неизвестна», а не «не подходит»; `contrast_bpm_jump` работает и без `max_bpm_jump`;
  пул только из якорей больше не роняет планировщик; пропущенный якорь — одно нарушение;
  `use_all_tracks` сверх 600 кандидатов — ошибка сценария; «полировка» варианта одиночными ходами;
  контрфакты разделяют цель и нарушения и не называют компромисс пресета «пропуском поиска».
* Сценарии: `excluded` — только точное совпадение; `ai_positions` для фаз с любыми именами;
  `description` фазы принимается; `--max-tracks`/`--duration-min` проверяются против сценария.
* Правила ролей: доказательство `curator.energy` для кураторской энергии, энергии начала/конца на
  шкале трека, `judged_roles` (`rules:interpret-v2`).
* CLI/выходы: выходы не перезаписывают входы и кэши команды; кэши research/AI — рядом с кэшем
  анализа; `render` по умолчанию берёт вариант, показанный `plan`, и его кэш анализа, пишет
  исключённые/невыбранные треки в манифест; полный сбой `enrich`/`interpret` — код 5; код 3 — только
  FFmpeg или отсутствующий необязательный пакет; `--sources` по умолчанию включает Discogs при токене.
* Отчёт: CSV плана экранирует формулы; сравнение называет лучший вариант по (тяжесть нарушений, цель).
* QC выходов `render`: нарушения плана — WARNING `plan_violation`; якорь `position -N` считается от
  конца; энергетический скачок запланированного контраста в миксе — INFO.
* Обратная совместимость 2.0: теги `role:` не участвуют ни в сходстве тегов секвенсора 2.0, ни в
  соответствии дейпарту.
* Research: альбом и год из тегов — тай-брейк при сопоставлении (не повод принять слабое
  совпадение); жанры Wikidata — теги сообщества, а не факты; режим `cache` видит закэшированные
  конфликты Discogs; просроченные записи MusicBrainz/Wikidata в `cache` используются с пометкой
  `stale` (хранятся не дольше года после истечения); `--report` без просроченных данных Discogs
  и с атрибуцией у каждого значения.
* AI: во вход модели не попадают артист, название и имя файла (`interpret-v2`); только
  описательные факты (белый список); доказательства по белым спискам для настроений и ритма;
  «пустые» блоки сегментного анализа не отправляются и доказательством не считаются; спор,
  держащийся только на Discogs, не меняет вход (стабильный ключ кэша); `PRIVACY_NOTE` в консоли.
* `suggest`: ASCII snake_case имена фаз (транслитерация кириллицы), проверка формы ответа по
  схеме, починка неверных типов с предупреждением, любой неисправимый ответ — понятная ошибка
  сценария (без падений на неверных типах и огромных числах).

## Automix 2.0 (2026-10-19)

Прототип переработан в пакет `veg_mixer` по ТЗ `ТЗ/task_01.md`: плейлист из отдельных
нормализованных треков (EBU R128, проверка на MP3), редакционный отбор по профилю клиента,
дейпарты, новый движок переходов, кэш с версиями, QC и отчёты. Исходный скрипт сохранён в
`legacy/automix_v1.py`. Подробности — `docs/EXECUTION_REPORT.md`, `README.md`.

---

## v1 — automix.py refactor

Structural cleanup, bug fixes, and footgun removal. **No audible/output behavior
changed** except the two explicitly accepted items below (P1.3 tracklist text, P1.4
uncached-BPM window). CLI interface, output filenames, mix format/bitrate unchanged.

> Note on state at handoff: when this pass was verified, the source already reflected
> P0.1–P1.5. This document records what each task changed (before → after) and the
> verification performed. The only source edit made in this pass was completing the
> P0.1 `select_tracks` comment; `CHANGES.md` was authored as the missing deliverable.

## P0.1 — Delete dead code
- Removed the unreachable SECOND full pipeline (analysis → harmonic sort →
  `select_tracks` → `build_mix` → export → `write_tracklist`) that followed
  `print("=== ALL PARTS COMPLETE ===")`. `main()` now stops at that print; the
  per-part loop above is the only pipeline.
- `select_tracks()` is now unused but kept, tagged `# reserved for --target-duration mode`.

## P0.2 — Fix non-deterministic seed
- `build_mix()` seed changed from process-randomized `hash(...) % (2**32)` to
  `int(hashlib.md5(output_file.name.encode()).hexdigest()[:8], 16)`.
- Same variable name (`seed_value`) and the same `print(f"  Random seed: {seed_value}")`.
- Reproducible across processes (no longer affected by `PYTHONHASHSEED`).

## P0.3 — Simplify camelot_compatible()
- Rewritten for readability; behavior identical for valid codes (1A–12B).
- Verified cases: (8A,8A)=T, (8A,9A)=T, (8A,7A)=T, (12A,1A)=T (wrap-around),
  (8A,8B)=T (relative), (8A,10A)=F, (8A,9B)=F — plus reverse wrap (1A,12A)=T
  and B-mode wrap (12B,1B)=T. All pass.

## P1.1 — Kill globals
- Removed module-level state globals `MORNING_DIR`, `OUTPUT_FILE`, `BPM_CACHE_FILE`.
- Introduced `@dataclass MixConfig(source_dir, output_file, cache_file, bitrate)`,
  built in `main()` from parsed args.
- `build_mix()` and `write_tracklist()` now take `output_file` as an explicit
  parameter (used for the seed, fades, and the `_tracklist.txt` name).
- `detect_bpm()` / `detect_key()` receive a `BPMCache` instance (see P1.5).
- **No `global` statements remain** in the file (verified: 0 occurrences).
- Tuning constants kept at module level: `TARGET_DURATION_S`, `TOLERANCE_S`,
  `FADE_IN_S`, `FADE_OUT_S`, `SILENCE_THRESH_DBFS`, `LIBROSA_SR`, `CAMELOT_MAP`,
  `MAJOR_PROFILE`, `MINOR_PROFILE`, `FX_OPTIONS`, `FX_WEIGHTS`, and the silence-scan
  constants. (Note: the old `OUTPUT_BITRATE` constant is now the `--bitrate` CLI arg,
  default `320k`, stored on `MixConfig` — CLI default unchanged.)

## P1.3 — Rename misleading FX function
- `apply_reverb_tail` → `apply_echo_tail` (it is a 3-tap delay/echo, not reverb).
- Updated everywhere: `FX_OPTIONS` (`"reverb_tail"`→`"echo_tail"`,
  `"filter+reverb"`→`"filter+echo"`), `apply_fx()` dispatch branches, and the FX
  strings written to the tracklist / used as `fx_counts` keys. `FX_WEIGHTS` unchanged
  (same weight at the same index).
- **Tracklist TEXT output changes** (`echo_tail`/`filter+echo` labels). Accepted:
  honest labeling, not audible behavior. Everything else byte-identical.

## P1.4 — Faster BPM detection (analysis window only)
- `detect_bpm()` no longer loads the entire track. It now loads 90 s from offset 30 s
  (mirroring `detect_key`), with fallback to the first 90 s when the offset load yields
  < 5 s of audio:
  ```python
  y, sr = librosa.load(str(path), sr=LIBROSA_SR, mono=True, offset=30, duration=90)
  if len(y) < sr * 5:   # NOTE: threshold later raised to sr*30 — see "Post-review hardening"
      y, sr = librosa.load(str(path), sr=LIBROSA_SR, mono=True, duration=90)
  ```
- **CAVEAT (accepted exception to "no output change"):** for **UNCACHED** tracks this
  can yield a slightly different BPM than full-track analysis (intro/outro excluded),
  which can change the harmonic sort order and therefore the resulting mix. The BPM
  detection *algorithm* is unchanged — only the analysis window. **Already-cached
  tracks are unaffected** (their BPM is read from cache).

## P1.5 — Encapsulate BPM cache with batched saves
- Replaced the dict + free functions (`load_bpm_cache`, `save_bpm_cache`, `_entry`)
  with a `BPMCache` class:
  - `__init__(path)` loads via `_load()`.
  - `_load()` returns parsed JSON if the file exists, else `{}` (no migration here).
  - `save()` writes `self._data` to disk once (JSON, `indent=2`, `ensure_ascii=False`).
  - `get_entry(key)` returns the per-track dict, **migrating legacy plain-float entries
    in memory** (`float → {"bpm": float}`) without saving.
  - `set_field(key, field, value)` ensures the entry exists (via `get_entry`) and
    mutates in memory only.
- `detect_bpm()` / `detect_key()` READ via `get_entry()` and WRITE via `set_field()`;
  **neither calls `.save()`**.
- `.save()` is called **exactly once**, in `main()`, in a `try/finally` around the
  analysis loop — so a crash mid-analysis still persists whatever was computed.
- **Legacy float cache entries still load and migrate transparently** (backward compat
  preserved). No references to `load_bpm_cache` / `save_bpm_cache` / `_entry` remain.

---

## Verification performed
- `python -m py_compile automix.py` → OK.
- `python automix.py --help` → exit 0; args `--folder`, `--output`, `--cache`,
  `--parts` (default 1), `--bitrate` (default `320k`) all present.
- `camelot_compatible` — all P0.3 cases + reverse/other-mode wrap-arounds pass.
- `BPMCache` — legacy float migrates in-memory; disk untouched until `save()`;
  `set_field` mutates in-memory only; `save()` persists migration; existing dict
  entries preserved; JSON written once.
- Static: `global ` occurrences = 0; no `hash(` used for the seed; no `reverb` /
  `reverb_tail` / `filter+reverb`; `save()` defined once, called once (in `main`).

## Post-review hardening (follow-up)

Three follow-up fixes after the initial pass.

### 1. Accurate fallback comments (was: misleading)
- `detect_bpm`: `# shorter than ~30s` → `# < 30 s captured from offset — ... analyze whole track`
  (comment now states the real audio-duration threshold, and matches the new logic below).
- `detect_key`: `# shorter than ~31s` → `# < 1 s captured from offset — retry from start`
  (real threshold is `len(y) < sr`, i.e. 1 s; **logic unchanged**).

### 2. Lazy `pydub` import — `--help` / `import automix` work without the audio backend
- Top-level `from pydub import AudioSegment` is now wrapped:
  ```python
  try:
      from pydub import AudioSegment
  except ImportError:   # allow `import automix` / `--help` without the audio backend
      AudioSegment = None
  ```
- Added `from __future__ import annotations` so the `AudioSegment` type hints on function
  signatures are not evaluated at import time (they stay as strings).
- Added a fast, clear guard at the top of `main()` (after `parse_args`, so `--help` still
  exits first): if `AudioSegment is None`, print
  `ERROR: pydub is required to run the mixer (pip install pydub; ffmpeg must be in PATH).`
  and `sys.exit(1)` — instead of a cryptic `AttributeError` deep in the pipeline.
- Verified: in an env **without** pydub, `python automix.py --help` → exit 0 with the full,
  unchanged usage; `import automix` succeeds; running the pipeline → the clean error above.

### 3. BPM fallback threshold `sr*5` → `sr*30` (silent quality bug)
- **Symptom:** with `sr*5` (5 s), any uncached track roughly **35–60 s** long analyzed only
  the short tail from offset 30 s (e.g. a 40 s track → just its last 10 s), discarding most
  of the audio. If that tail is unrepresentative (breakdown/outro), BPM is wrong.
- **Evidence (synthetic 40 s track: 30 s @128 BPM groove + 10 s @90 BPM outro):**
  - old window (offset=30 → 10 s tail): **89.1 BPM** (locks onto the outro)
  - new window (whole track from start): **129.2 BPM** (locks onto the main groove)
  - a ~40 BPM error that would misplace the track in the harmonic sort.
- **Fix:** raise the fallback threshold to `sr * 30`. Now a track whose offset-30 window
  captures **< 30 s** of audio is analyzed whole from the start; tracks ≥ 60 s still use the
  30 s intro-skipping window (unchanged). The beat-tracking *algorithm* is untouched — this
  is still an analysis-**window** change, same category as P1.4.
- **Scope:** affects only **uncached** tracks in the ~35–60 s band, and only toward *more*
  representative analysis — squarely inside the P1.4 accepted-exception envelope. Cached
  tracks unaffected.
- **Known adjacent (deliberately NOT changed):** `detect_key` has the same latent pattern but
  a `sr` (1 s) threshold, so a 40 s track still keys off its 10 s tail. Lower severity (chroma
  is averaged over the window and key only feeds a sort tie-break). **Decision: left as-is** —
  a past experiment where a track's detected key changed produced very bad mixes, so the
  key-detection window is kept conservative on purpose.

## Out of scope (not done, per instructions)
ffmpeg streaming; unifying silence-trim with `preprocess_audio.py`; dry-run/preview;
parallel analysis; tests; exhaustive type hints; tracklist format changes beyond the
P1.3 FX rename.
