# VEgroove Automix 2.0 / VEmix 3.0 — архитектура и контракты модулей

Документ — контракт между модулями пакета `veg_mixer`. Общие типы — в
`veg_mixer/models.py`, настройки — в `veg_mixer/config.py`. Сигнатуры ниже
обязательны: на них опираются CLI (`cli.py`) и оркестрация (`pipeline.py`).

## Поток данных

```
library folder
   │  analysis.scan_library / TrackAnalyzer.analyze   (cache.AnalysisCache)
   ▼
AnalyzedTrack (TrackFeatures + AudioProbe + LoudnessMeasurement pass-1)
   │  metadata.apply_metadata  (ручные правки куратора поверх авто-значений)
   ▼
TrackFeatures[]
   │  curation.curate(profile)  → CurationResult (eligible / excluded + причины)
   ▼
eligible TrackFeatures[]
   │  sequencing.build_sequence / plan_dayparts
   ▼
SequenceResult (порядок + оценки переходов + флаги)
   ├── Mode A: playlist.export_playlist → tracks/NNN_*.mp3, playlist.m3u8, manifest.json, tracklist.txt
   └── Mode B: loudness.normalize_to_wav (каждый трек) → renderer.render_mix → WAV (RF64 float)
                → loudness.normalize_to_mp3 (весь микс, проверка на MP3) → mix.mp3 + manifest
   ▼
quality_control.qc_playlist / qc_mix → QCReport → reports.write_reports (JSON + TXT)
```

## Инварианты (для всех модулей)

1. **Исходники неизменны.** Ни один модуль не пишет в исходные аудиофайлы и не
   перезаписывает их; выходной путь, совпадающий с исходным, — ошибка.
2. **Неизвестное остаётся неизвестным.** `None` не превращается в 0, «совместимо»,
   «английский» или «популярный». Оценки без надёжного источника помечаются
   источником (`feature_sources`) и пониженной уверенностью.
3. **Никаких тихих пропусков.** Битый файл → запись с ошибкой в отчёте, а не
   исчезновение из выборки.
4. **Один lossy-шаг.** MP3 кодируется один раз из lossless-промежуточного файла;
   MP3 никогда не перекодируется из MP3.
5. **Атомарная запись.** Выходные файлы пишутся во временный файл в той же
   папке и публикуются `os.replace`.
6. **Детерминизм.** Любая «случайность» (вариативность секвенсора, FX random)
   — только от явного `seed`.
7. **Одна реализация правил.** Нормализация тегов, артистов, языков и путей — только
   в `textnorm.py`; QC проверяет исключения, повторно вызывая `curation.curate`, а не
   собственной копией правил.
8. **Никогда PASS при сбое.** Трек, не прошедший нормализацию, и неопубликованный микс
   дают FAIL; пропущенная критичная проверка не даёт PASS.
9. **Правило ключа (память проекта).** Окно анализа тональности (`detect_key`:
   offset 30 s, duration 60 s, повтор с начала только если захвачено < 1 s)
   **не менять** — прошлый эксперимент со сменой определённых тональностей дал
   очень плохие миксы. Алгоритмы BPM/key перенесены из `legacy/automix_v1.py`
   без изменений и закреплены регрессионным тестом.

## Модули и обязательные API

### `ffmpeg_tools.py` (готово)
`find_ffmpeg()`, `run_ffmpeg()`, `probe_audio()`, `decode_check()`,
`measure_loudness()`, парсеры `parse_loudnorm_json()` / `parse_ebur128_*()`.
Исключения: `FFmpegNotFoundError` (нет бинарников — пробрасывать до CLI),
`FFmpegError`, `LoudnessMeasurementError`.

### `loudness.py` (готово)
`LoudnessNormalizer(tools, LoudnessConfig, EncodeConfig, QCConfig, work_dir=)`:
`measure(path)`, `plan(m)`, `normalize_to_mp3(src, dest, input_measurement=, source_probe=, metadata=)`,
`normalize_to_wav(src, dest, ...)`, `verify_output(path, plan=, expected_duration_s=, kind=)`.
Чистые функции: `plan_normalization`, `evaluate_output`, `compare_adjacent_short_term`.
Статусы: `passed` / `review` (выдано безопасно по пику, цель не достигнута) / `failed`.

### `textnorm.py`
`normalize_tag`, `split_values`, `label_keys`, `split_artists` (полная подпись + части:
`A/B`, `A & B`, `A x B`, `feat.`), `primary_artist`, `parse_languages` → (ISO 639-1, нераспознанные),
`normalize_language` (`'ru;en'`), `normalize_rel_path`, `rel_path_key`.

### `cache.py`
```python
ANALYSIS_CACHE_SCHEMA = "vegroove.analysis_cache"
class AnalysisCache:
    def __init__(self, path: Path, *, legacy_path: Path | None = None,
                 use_content_hash: bool = False, read_only: bool = False) -> None: ...
    def get(self, source: Path, section: str, *, version: str, params: dict | None = None) -> dict | None: ...
    def put(self, source: Path, section: str, value: dict, *, version: str, params: dict | None = None,
            expected_fingerprint: dict | None = None) -> bool: ...   # False: файл менялся во время анализа
    def save(self) -> None: ...            # атомарно; no-op если не было изменений
    stats: dict[str, int]                  # hits, misses, invalidated, migrated
    warnings: list[str]                    # напр. «повреждённый кэш сохранён как *.corrupt-…»
```
* Идентичность: абсолютный путь + размер + `mtime_ns` (+ sha256 при `use_content_hash`).
  Смена отпечатка инвалидирует **все** секции файла.
* Секции независимы (`probe`, `bpm`, `key`, `energy`, `loudness`): у каждой своя
  `version` и хэш `params`. Смена цели нормализации меняет только `loudness`.
* Повреждённый/обрезанный JSON, который **доказуемо является кэшем VEgroove** (наша schema,
  обрезанный документ, начинающийся как наш, пустой файл) → переименовывается в `*.corrupt-<n>`,
  работа продолжается с пустым кэшем + предупреждение. Чужой файл по этому пути не трогается
  (запись отключается).
* Миграция легаси `bpm_cache.json` (`{abs_path: float}` или
  `{abs_path: {"bpm", "camelot", "key_confidence"}}`): читается, **не изменяется**;
  значения переносятся в секции `bpm`/`key` текущих версий с пометкой
  `migrated_from`, если файл существует. Отпечаток снимается в момент миграции.
* Потокобезопасность (lock).
* Кэш никогда не переименовывает и не перезаписывает файл, который не является (повреждённым)
  кэшем VEgroove: чужой JSON, `bpm_cache.json` v1, CSV куратора или аудио по `--cache`
  открываются только на чтение (запись отключена, предупреждение); не-`.json` путь — ошибка.
* `read_only=True` (dry run): ничего не пишет и не переименовывает.
* Отпечаток файла снимается ДО анализа; если файл изменился во время анализа, результат не
  сохраняется (`put(..., expected_fingerprint=)`), трек получает причину ревью.

### `analysis.py`
```python
BPM_ALGO_VERSION = "bpm-v1"; KEY_ALGO_VERSION = "key-v1"; ENERGY_ALGO_VERSION = "energy-heuristic-v1"
def scan_library(folder: Path, *, recursive: bool = False, extensions: tuple[str, ...] = AUDIO_EXTENSIONS,
                 exclude: Iterable[Path] = ()) -> list[Path]: ...   # отсортировано; свои выходы пропускаются
def detect_bpm(path: Path, cache: AnalysisCache, cfg: AnalysisConfig) -> float: ...
def detect_key(path: Path, cache: AnalysisCache, cfg: AnalysisConfig) -> tuple[str, float | None]: ...
def estimate_energy(path: Path, cache: AnalysisCache, cfg: AnalysisConfig) -> float | None: ...
def detect_script_hint(*texts: str | None) -> str | None: ...   # "cyrillic" | None
def parse_artist_title(stem: str) -> tuple[str | None, str | None]: ...
class TrackAnalyzer:
    def __init__(self, tools, cache: AnalysisCache, cfg: AnalysisConfig, normalizer: LoudnessNormalizer | None = None): ...
    def analyze(self, path: Path, library_root: Path, *, measure_loudness: bool = True) -> AnalyzedTrack: ...
```
* Артист/название: теги (`artist`, `title`) → иначе имя файла `Artist - Title` (источник `filename`).
* `genre` из тега → `feature_sources["genre"]="id3"`; `language` из тегов `language`/`TLAN`.
* `energy` — эвристика (источник `auto:energy-heuristic-v1`, `energy_confidence` = `cfg.energy_confidence`),
  не зависит от громкости мастера. `danceability`, `vocal_presence`, `mood`, `familiarity` — `None`.
* Кириллица в артисте/названии → `script_hint="cyrillic"` + причина в `review_reasons`
  (это подсказка, **не** утверждение «язык = ru»).

### `metadata.py` (редакционные метаданные куратора)
```python
def export_metadata(tracks: list[TrackFeatures], path: Path, *, existing: dict | None = None) -> None   # .csv | .json
def load_metadata(path: Path) -> dict[str, dict]                       # ключ = rel_path (или имя файла)
def apply_metadata(tracks: list[TrackFeatures], overrides: dict[str, dict]) -> tuple[list[TrackFeatures], list[str]]
```
* Колонки = `models.EDITABLE_FIELDS` + `rel_path` (+ справочные авто-значения в `auto_*`).
* Пустая ячейка = «не переопределять». Ручные значения выигрывают, источник `manual`.
* `vibe_tags` в CSV — через `;`. Булевы: `true/false/yes/no/1/0`. Ошибки значений —
  понятное исключение `MetadataError` с номером строки.
* Второй элемент результата — предупреждения (ключи без совпадений и т. п.).

### `curation.py` (профиль клиента и правила)
```python
DEFAULT_DAYPARTS: dict[str, DaypartProfile]          # morning / afternoon / evening
class ProfileError(ValueError)
def load_profile(path: Path) -> ClientProfile          # YAML или JSON, строгая валидация
def default_profile() -> ClientProfile                 # без профиля: ничего не исключает, но помечает
                                                       # неизвестный язык и кириллицу на ревью
def curate(tracks: list[TrackFeatures], profile: ClientProfile) -> CurationResult
def get_daypart(profile: ClientProfile, name: str) -> DaypartProfile
```
Приоритет правил: blocklist (жёстко) → `excluded` в метаданных → исключённый язык — если
**любой** из языков трека исключён (не перекрывается allowlist) → `allowlist.only` → мягкие
правила (язык вне `allowed`, неизвестный язык, кириллица, `avoid`-предпочтения,
`require_approved`) — allowlist возвращает треку право на эфир, но причины ревью остаются.
`editorial_score` (−1..+1) из `preferences` × совпадений `preference_tags`
(по `vibe_tags`, `genre`, `mood`); `familiarity` учитывается только если задана куратором и
монотонно: известная высокая > неизвестная > известная низкая. Причины ревью из анализа
(не найдены ключ/темп, усечённый файл, файл менялся во время анализа) ставят `requires_review`;
кириллица в названии при языке из тега (не от куратора), не являющемся кириллическим, — конфликт на ревью.

### `sequencing.py`
```python
def camelot_compatible(a: str, b: str) -> bool                       # семантика legacy
def transition_score(a: TrackFeatures, b: TrackFeatures, cfg: SequencingConfig) -> TransitionScore
def build_sequence(pool: list[TrackFeatures], cfg: SequencingConfig, *,
                   profile: ClientProfile | None = None, daypart: DaypartProfile | None = None,
                   editorial_scores: dict[str, float] | None = None, seed: int = 0,
                   target_duration_s: float | None = None, max_tracks: int | None = None,
                   exclude_track_ids: frozenset[str] = frozenset(),
                   protected_track_ids: frozenset[str] = frozenset()) -> SequenceResult
def plan_dayparts(pool, cfg, *, profile, dayparts: list[str], editorial_scores=None, seed=0,
                  target_duration_s=None, max_tracks=None, protected_track_ids=frozenset()) -> dict[str, SequenceResult]
```
* Компоненты нормированы в 0..1; неизвестные — `None`, вес перераспределяется
  с учётом уверенности, итог «стягивается» к `neutral_score` пропорционально покрытию.
* Приоритет: редакционные ограничения → настроение → энергия → жанр → темп → гармония.
* Жёсткие ограничения при наличии альтернатив: артист не ближе `artist_min_gap`,
  скачок энергии ≤ `max_energy_jump`, не больше `max_similar_run` слишком похожих подряд.
* Дейпарт влияет на отбор (диапазон энергии/BPM, предпочитаемые теги/жанры/настроения)
  и на порядок (кривая энергии, мягкое начало). Треки между дейпартами не повторяются.

### `transitions.py` + `renderer.py` (Mode B)
```python
def plan_transition(prev: MixInput, nxt: MixInput, prev_dur_s: float, next_dur_s: float,
                    cfg: TransitionConfig, *, prev_tail_db: np.ndarray | None = None,
                    next_head_db: np.ndarray | None = None, rng=None,
                    fx_history: Sequence[str] = ()) -> TransitionPlan
def fade_curves(n: int, curve: str) -> tuple[np.ndarray, np.ndarray]           # (fade_out, fade_in)
def apply_fx(region: np.ndarray, sr: int, fx: str) -> np.ndarray                # только регион уходящего трека
def render_mix(inputs: list[MixInput], out_wav: Path, cfg: TransitionConfig, *,
               ceiling_dbtp: float, sample_rate: int = 44100, seed: int = 0,
               progress=None) -> MixRenderResult
def write_mix_tracklist(result: MixRenderResult, inputs: list[MixInput], path: Path, *,
                        mix_name: str | None = None) -> None
```
* Потоковый рендер: в памяти — текущий трек, следующий (просмотр вперёд для планирования
  перехода) и удержание под финальный fade-out; отрендеренное записано в файл и больше не меняется.
* FX применяется **только** к копии перекрываемого хвоста уходящего трека.
* Пиковая защита: true peak перекрытия (oversampling ×4); превышение потолка →
  плавное снижение усиления только в зоне перекрытия, фиксируется в `TransitionReport`.

### `playlist.py` (Mode A)
```python
def safe_filename(index: int, artist: str | None, title: str | None, fallback: str, *, max_len: int = 120) -> str
def export_playlist(sequence: SequenceResult, output_dir: Path, normalizer: LoudnessNormalizer, *,
                    profile_name: str | None, excluded: list[dict], input_measurements: dict[str, LoudnessMeasurement] | None = None,
                    source_probes: dict[str, AudioProbe] | None = None, jobs: int = 1,
                    dry_run: bool = False, progress=None, not_selected: list[dict] | None = None) -> PlaylistExportResult
def write_m3u8(path: Path, entries: list[ManifestTrack], title: str | None = None) -> None
def read_m3u8(path: Path) -> list[tuple[float | None, str, str]]                 # (duration, title, relpath)
```
`output/playlist.m3u8`, `output/manifest.json`, `output/tracklist.txt`, `output/tracks/NNN_artist_title.mp3`.
Треки со статусом `failed` в плейлист **не попадают** (перечислены в manifest `warnings`/`excluded`
со `stage='normalization'`). Пустая последовательность не бросает исключение: результат без
опубликованных файлов (`manifest_path is None`) и с причиной. Тег `album` исходника не
перезаписывается: имя плейлиста пишется в TXXX `VEGROOVE_PLAYLIST`; `album` ставится, только
если у исходника его нет. Чужие `playlist.m3u8`/`tracklist.txt` без манифеста VEgroove не
перезаписываются (`PlaylistExportError`). Если откат публикации не удался, файлы сохраняются в
`tracks/.veg-recovery-<время>/` (с `RECOVERY.txt`) и не удаляются автоматически.

### `quality_control.py` + `reports.py`
```python
def qc_playlist(output_dir: Path, normalizer: LoudnessNormalizer, *, profile: ClientProfile | None = None,
                qc_cfg: QCConfig | None = None) -> QCReport
def qc_mix(manifest_path: Path, normalizer: LoudnessNormalizer, *, profile: ClientProfile | None = None,
           qc_cfg: QCConfig | None = None) -> QCReport
def scan_silence(tools, path: Path, cfg: QCConfig, *, boundaries_s: list[float] = (),
                 fade_in_s: float = 0.0, fade_out_s: float = 0.0, sample_rate: int | None = None) -> list[dict]
def overall_status(issues: list[QCIssue], checks_skipped: list[str]) -> OverallStatus
# reports.py
def render_text_report(report: QCReport) -> str
def write_reports(report: QCReport, output_dir: Path, *, stem: str = "qc_report") -> tuple[Path, Path]
```
PASS только если все критические проверки выполнены и нет WARNING/FAIL.
Предупреждения содержат таймкоды.

### `pipeline.py` / `cli.py`
* До любой работы все запланированные выходы (MP3, манифесты, трек-листы, отчёты) сверяются
  со всеми найденными исходниками (`fsutil.assert_outputs_safe`); совпадение или коллизия
  имён — ошибка ввода (код 2). Собственные выходы исключаются из сканирования.
* `manifest.excluded[*].stage` ∈ {`curation`, `analysis`, `normalization`}; у выходов 3.0
  `render` — также `scenario` (треки, исключённые сценарием, см. `plan_report.py` /
  `plan_pipeline.py`); `manifest.not_selected` — подходящие треки, которые секвенсор никуда не поставил, с причиной.
* Микс: `mix.published`, при публикации — `sha256` и `size_bytes` MP3; неопубликованный микс → FAIL.
  Финальный проход громкости микса никогда не использует динамический режим `loudnorm`.
* `qc` берёт цели (громкость, частота, битрейт) из манифеста; ключ командной строки
  переопределяет только своё поле.
* Коды выхода: 0 PASS, 1 FAIL, 2 ввод, 3 окружение (FFmpeg; для 3.0 — и необязательный пакет,
  например SDK `anthropic`: `pipeline.OptionalDependencyError`), 4 WARNING, 5 ошибка выполнения
  (в т.ч. `enrich`/`interpret`, у которых не удался ни один запрос, — частичный сбой даёт 4, — и
  `suggest`, которому провайдер так и не ответил: сеть, ключ, лимит, сбой сервера, таймаут —
  `SuggestProviderError`), 130 — прервано пользователем (Ctrl+C, в stderr `Interrupted.`). У
  `suggest` отказ модели, обрезанный или неразбираемый ответ и черновик, невалидный и после
  повтора, — `ScenarioError`, код 2; неверная настройка провайдера (неизвестное имя, не заданная
  модель, не заданный или неверный URL — `AIConfigError`) — тоже 2.

---

## VEmix 3.0 — драматургия, внешние факты, AI-интерпретация

План и аудит — [VEMIX3_PLAN.md](VEMIX3_PLAN.md). Всё ниже **добавлено рядом** с 2.0:
команды `analyze/normalize/playlist/mix/qc` без новых аргументов работают как раньше,
форматы кэша анализа, манифестов и отчётов 2.0 не меняются.

### Поток данных 3.0

```
library ─► analysis (2.0) ─► TrackFeatures ─────────────────────────────┐
   └─► audio_intel.analyze_segments  (секция кэша "segments", только по запросу)
research/  enrich    ─► veg_research_cache.json  (факты с источником; online | cache | off)
ai/        interpret ─► veg_ai_cache.json        (строгая схема; anthropic | openai-compatible)
interpret_rules.derive (роли из измерений, без сети)                     │
                                                                         ▼
scenario.load_scenario ─► resolve_scenario(curated library) ─► dramaturgy.build_context
                                                               (по треку, с происхождением значений)
                                                                         ▼
dramaturgy.plan ─► PlanResult (варианты, слагаемые цели, нарушения, объяснения)
   ├─► plan_report: dry-run отчёт, сравнение, plan.json (vegroove.plan), M3U8, CSV
   └─► plan_pipeline.run_render ─► export_playlist / _render_one_mix (2.0) ─► manifest.plan ─► qc
```

### Дополнительные инварианты

10. **Категории сведений не смешиваются.** Измеренное (`auto:*`), теги файла (`id3`,
    `filename`), куратор (`manual`), внешние факты (`research:*`), правила над измерениями
    (`rules:*`) и AI (`ai:*`) хранятся раздельно (`TrackContext.provenance`). Внешние данные и
    AI **никогда** не записываются в поля `TrackFeatures` (их читают curation и QC, а
    `confidence()` дал бы незнакомому источнику 0.8).
11. **Жёсткое важнее мягкого.** Оптимизатор сравнивает планы лексикографически:
    (суммарная тяжесть нарушений, целевая функция). Позиционные якоря и обязательные треки —
    структурные: ходы, которые их ломают, не делаются.
12. **План проверяем детерминированно.** Тот же вход + seed → тот же план; бюджет поиска —
    число вычислений цели, не время. `evaluate_order` пересчитывает цель сохранённого порядка.
13. **Сеть — только в `enrich`/`interpret` в режиме `online` и в `suggest`** (запрос к AI с
    брифом и обезличенной статистикой библиотеки; если ответ неразбираемый, не JSON-объект или не
    даёт валидного сценария либо ошибка провайдера повторяемая — ещё один запрос с перечнем проблем,
    итого не больше двух; сетевые повторы SDK/HTTP (`max_retries`) — сверх этого). `plan` и
    `render` читают кэши. Аудио никуда не отправляется; AI получает только текстовые признаки без
    путей, имён файлов и названий.
15. **Выход не перезаписывает вход.** Каждый выход сверяется не только с аудио-исходниками, но
    и со входами самой команды (сценарий, CSV куратора, профиль, план, кэши анализа/research/AI):
    совпадение — ошибка ввода (код 2).
14. **Рендер — только отдельной командой.** `plan` ничего не нормализует и не рендерит;
    `render` воспроизводит **сохранённый** порядок (без повторной сортировки) и отказывается
    работать с устаревшим планом (файл изменился после планирования).

### `scenario.py`
```python
class ScenarioError(ValueError)
ROLES = ("opener", "tension_builder", "groove_builder", "release", "breather", "peak",
         "transition", "comedown", "closer")
DEFAULT_WEIGHTS: dict[str, float]            # веса слагаемых цели (см. dramaturgy)
def load_scenario(path: Path) -> Scenario    # YAML/JSON, строгая валидация (тот же ридер, что у профиля)
def parse_scenario(data, *, source="<scenario>") -> Scenario
def with_overrides(sc, *, target_duration_s=None, max_tracks=None) -> Scenario  # --duration-min/--max-tracks + повторная проверка
def match_reference(ref: str, tracks) -> tuple[list[TrackFeatures], str]   # 'exact' | 'relaxed' | 'none'
def song_key(track) -> str | None            # одна песня в разных версиях — один ключ
def resolve_scenario(scenario, eligible, *, curation_excluded=None, library=None) -> ResolvedScenario
AI_POSITIONS = ("intro", "build", "first_peak", "breather", "second_build", "climax", "descent", "closer")
```
* Ключи верхнего уровня: `scenario, description, target_duration_min, duration_tolerance_min,
  max_tracks, min_tracks, use_all_tracks, alternatives, seed, pool{tracks, include_tags,
  exclude_tags}, anchors[{track, position | phase}], required, excluded,
  adjacency{forbid, avoid, prefer}, curve | phases[{name, share, energy, energy_range, roles,
  moods{prefer, avoid}, contrast, ai_positions, description}], curve_axis (time | tracks),
  moods{prefer, avoid}, transitions{max_bpm_jump,
  max_key_steps, max_energy_jump, contrast_energy_jump, contrast_bpm_jump}, diversity{artist_min_gap,
  max_tracks_per_artist, avoid_same_song}, weights{...}`. Ошибки — `'<файл>: <ключ>: <проблема>'`.
* Ссылка на трек: путь в библиотеке, имя файла или «Исполнитель - Название» (правило
  блок-/аллоулиста `curation.track_match_keys`); если точного совпадения нет — без скобочных
  уточнений версии, но **весь** исполнитель должен совпасть (`Fire` ≠ `Earth, Wind & Fire`).
  Якорь или обязательный трек, которого нет, который неоднозначен или исключён профилем, — ошибка.
  `excluded` сопоставляется **только точно**: исключение «Song (Live)» не исключит студийную
  «Song»; `pool.tracks` и `adjacency` допускают совпадение без уточнений версии и пишут об этом заметку.
* `with_overrides`: `--duration-min` ограничен (0, 1440] минутами, как поле `target_duration_min`,
  и сбрасывает `duration_tolerance_min` сценария (действует допуск по умолчанию); `--max-tracks` проверяется против якорей,
  обязательных треков и `min_tracks`, как поле `max_tracks`. Явные длина или число треков из CLI
  важнее сценария: оба флага отключают его `use_all_tracks` (в файле такое сочетание — ошибка), и
  план строится как подмножество.
* `ai_positions` фазы — какие позиции словаря AI (`AI_POSITIONS`) считаются подсказкой для неё;
  по умолчанию — собственное имя фазы, если оно из словаря.
* Кривая: `target_energy(t)` — по относительному **времени** середины трека (длительности
  исходников) или, при `curve_axis: tracks`, по номеру позиции. Доли фаз нормируются к сумме 1.

### `dramaturgy.py`
```python
def build_context(track, *, segments=None, research=None, interpretations=(), editorial=None,
                  use_community_tags=False) -> TrackContext
def plan(contexts, resolved, *, seed=0, alternatives=None, seq_cfg=None, profile=None,
         settings: SearchSettings | None = None) -> PlanResult
def evaluate_order(model: Model, track_ids, weights=None) -> Evaluation     # проверка сохранённого плана
def explain(model, order, weights, ev=None, *, counterfactuals=True, own=None, preset=None) -> list[PlanEntry]
```
* Целевая функция — сумма нормированных (0..1) слагаемых × веса (формулы — в docstring
  модуля; кривая насыщается при |отклонении| 0.5, так что большие промахи различимы;
  исключения: `diversity` суммирует близкие пары и может быть > 1, `adjacency` = −1 для желательной пары):
  позиционные `curve, band, role, mood, editorial, uncertainty`, переходные
  `transition (оценка 2.0), edge_energy, tempo, harmony, timbre, mood_flow, diversity,
  adjacency`, глобальное `duration`.
* Нарушения (`Violation.kind`): `anchor, required, phase_anchor, forbidden_pair, energy_jump,
  tempo_jump, key_jump, artist_gap, artist_count, same_song, duration, track_count`. Лимит темпа
  жёсткий при уверенности обоих BPM ≥ `HARD_LIMIT_CONFIDENCE` (0.45), лимит тональности — при
  ≥ `KEY_HARD_LIMIT_CONFIDENCE` (0.6): авто-тональность (≤ 0.5 после калибровки 2.0) жёсткой не
  бывает никогда — переход помечается «не проверено, не применено». Пропущенный позиционный
  якорь — одно нарушение `anchor` (не дублируется `required`).
* Темп сверяется с сегментным анализом (`summary.tempo`, если он посчитан): авто-BPM, которому
  сегментный анализ противоречит (другой пульс, не просто x2/x0.5, при его уверенности ≥ 0.3) или
  для которого он не нашёл пульса, получает уверенность `TEMPO_DOUBT_CONFIDENCE` (0.3) — лимит
  темпа для него не жёсткий, отчёт пишет почему. Двусмысленность half/double-time лишь
  показывается: лимиты сравнивают «сложенную» разницу BPM. Ручной BPM не перепроверяется.
  Кэш анализа при этом не меняется.
* Идентичность артиста/песни — объединение ключей: имя (`artist_key`, `song_key`) и, у
  сопоставленных записей, MBID артиста / произведения; треки с общим ключом — один артист/одна
  песня (частичный research не отключает правила разнообразия).
* Настроения и роли по уровням, **отдельно для каждого вида**: метки **куратора** (`manual`; для
  настроений — колонка `mood` и слова-настроения в `vibe_tags`, для ролей — теги `role:`)
  **заменяют** интерпретации того же вида (отложенные видны в происхождении как `mood_set_aside` /
  `role_set_aside`). AI-подсказки позиций откладываются (`position_hint_set_aside`), только когда
  куратор назначил роль тегом `role:`; кураторское настроение без роли не отключает ни AI-роли, ни
  подсказки позиций. Без кураторских меток данного вида объединяются AI (≤ 0.8), правила, теги
  файла и теги сообщества (по флагу).
* Стоимость роли решается по фазе целиком: если ни одну из ролей фазы не мог оценить ни один слой
  (`role_judged`: AI оценивает все роли, правила — только поддержанные данными, тег куратора —
  только названную) и у трека нет AI-подсказки позиции для этой фазы, роль «неизвестна»
  (`UNKNOWN_ROLE_COST`), а не «не подходит». Если оценена хотя бы одна роль фазы (или есть
  подсказка), неоценённые роли фазы — несовпадение: берётся лучшая из оценённых ролей и подсказки.
  Пример (без AI и сегментного анализа): фаза `[tension_builder]` — «неизвестно», фаза
  `[tension_builder, peak]` у трека с известной энергией, которому правила не дали `peak`, —
  «не подходит».
* Вход в фазу `contrast: true` поднимает лимит скачка энергии до `contrast_energy_jump`, снимает
  (или задаёт `contrast_bpm_jump`) лимит темпа и снижает стоимость `transition`/`edge_energy`/
  `tempo`/`mood_flow`: запланированный контраст — не флаг.
* Нарушение якоря (позиционного или фазового) и пропуск обязательного трека весят
  `ANCHOR_SEVERITY` = 10 единиц, общие допуски — около 1: явное указание куратора о конкретном
  треке сильнее общих лимитов.
* Поиск: стартовые решения — beam по относительному времени (якоря на местах, обязательные и
  фазовые треки до дедлайна) и назначение треков на равные по времени слоты кривой (венгерский
  алгоритм по позиционной стоимости, одна версия песни); предыдущие варианты — тоже старты.
  Улучшение — итеративный локальный поиск (swap, relocate, перенос/обмен блоков, reverse,
  replace, insert, remove) с «отжигом» штрафа: первые стадии оценивают нарушения ценой
  `λ / длина плана` и могут проходить через невыполнимые планы, последняя — лексикографическая;
  ремонтные ходы целятся в места нарушений (в т.ч. «мостик» — трек промежуточного темпа/энергии
  между двумя) и сцепляются до трёх шагов. Бюджет — число вычислений цели (по умолчанию
  10000 + 200 × длина плана, ≤ 30000). После поиска — «полировка»: лучшая замена/перестановка
  для каждой позиции применяется, пока улучшает собственную цель варианта (та же окрестность,
  что у контрфактов отчёта). Кандидатов — до `POOL_CAP` (600) лучших по соответствию фазам плюс
  все якоря и обязательные треки: они остаются всегда, даже вне этих 600, поэтому кандидатов
  может быть чуть больше `POOL_CAP`. Отсечение **печатается** в заметках отчёта с фактическим
  числом кандидатов; `use_all_tracks` при пуле больше `POOL_CAP` — ошибка сценария, а не
  молчаливое отсечение.
* Альтернативы: пресеты весов `balanced, dramatic, smooth, harmonic, fresh, varied` + штраф за
  похожесть на предыдущие варианты; все варианты пересчитываются в весах сценария.
* Объяснения строятся только из вычисленных слагаемых: фаза и цель, отклонение, роли,
  AI-подсказки позиций и настроения с категорией источника, переход (оценка 2.0, ΔBPM, шаги
  Camelot, ΔE) и контрфакт — лучшая замена или перестановка для позиции: изменение цели J и
  тяжести нарушений **раздельно** (с допуском на сумму тяжестей), какие ограничения ход нарушил
  бы (`violations_added`), изменение каждого слагаемого. Ход, улучшающий цель сценария, но
  ухудшающий цель, с которой искался вариант (веса пресета + новизна), подписан как компромисс
  пресета, а не как «не найдено поиском».

### `interpret_rules.py`
`derive(track, segments=None) -> dict` — роли-гипотезы из энергии (кураторской — доказательство
`curator.energy`, или измеренной — `measured.energy`), траектории, плотности, секций; уверенность
≤ 0.5, источник `rules:interpret-v2`. Энергии начала/конца из сегментов переносятся на шкалу
энергии трека (как в планировщике). `judged_roles` — роли, которые данные вообще позволяли
оценить (`tension_builder`/`comedown`/`release` — только с сегментным анализом). Настроения
правилами не выводятся. Куратор может назначить роль тегом `role:<роль>` в `vibe_tags`; эти
теги не участвуют в сходстве тегов секвенсора 2.0.

### `moods.py`
Словарь настроений на плоскости валентность/возбуждение (редакционная конвенция, не
измерение): `label_affinity`, `best_affinity`, `mood_similarity`. Таблица `MOOD_FAMILIES`
секвенсора 2.0 **не меняется** — оценка перехода 2.0 остаётся прежней.

### `audio_intel.py` (сегментный анализ, `analyze --segments`)
```python
SEGMENTS_VERSION = "segments-v1"; SEGMENTS_SECTION = "segments"
@dataclass(frozen=True) class SegmentConfig      # свой sample_rate (22050), segment_s=5, max_analysis_s=900 ...
def segments_params(cfg) -> dict
def compute_segments(path, cfg=None, *, tools=None) -> dict          # без кэша; не бросает для битых/тихих файлов
def analyze_segments(path, cache: AnalysisCache, cfg=None, *, tools=None, compute=True,
                     expected_fingerprint=None) -> tuple[dict | None, bool]   # (value, from_cache); compute=False — только кэш
def timbre_similarity(a, b) -> float | None
```
* Отдельный декод (не общий с BPM/key — они закреплены бит-в-бит с v1), секция кэша `segments`
  со своими `version`/`params`; `bpm`/`key`/`energy`/`loudness` не меняются.
* Значение: по сегментам — `intensity` (эвристика энергии 2.0 по сегменту, не зависит от
  громкости), `rms_db` (уровень относительно медианы трека), `onset_rate`, `percussive_ratio`,
  `bass_ratio`, `centroid_hz`, `flatness`; `summary` — `energy_in/out/mean/peak`, `trajectory`,
  `density`, `percussive_density`, `bass_density`, `dynamic_range_db`, `tempo` (кандидаты +
  half/double-time; уверенность учитывает конкурирующий метрический уровень; не заменяет секцию
  `bpm`), `vocal_presence_proxy` (уверенность ≤ 0.3), `timbre` (6 чисел), `harmony` (тональность
  по STFT-хроме и профилям Крумхансла: `key_estimate`, `clarity`, `stability` — доля сегментов в
  пределах шага Camelot, `modulations` — устойчивые смены ключа, `segment_keys`; отдельное
  измерение, `key-v1` и его окно не меняются, в план попадает как предупреждение о нестабильной
  тональности); `sections` — интро/билд/пик/брейк/тело/аутро-**кандидаты** (новизна Фута,
  уверенность ≤ 0.5, `repeat_group` — кандидат припева, не утверждение); `notes`.
* Ничего не пишется в `TrackFeatures` и в `review_reasons`.

### `net.py`
`HttpClient(user_agent, timeout_s, max_retries, min_interval_s={host: s}, transport=...)` —
`get_json`/`post_json`/`request`; повтор 408/425/429/500/502/503/504/529 и сетевых ошибок с `Retry-After`
(секунды или HTTP-дата, ≤ 60 с), лимит размера ответа (и после gzip), только http/https.
Все сбои — `NetError` (`NetTimeout`, `NetHTTPError`, `NetDecodeError`); запросы логируются без
query-строки, заголовки с ключами — `<redacted>`.

### `research/` (Music Research Agent, `enrich`)
```python
RESEARCH_CACHE_NAME = "veg_research_cache.json"; RESEARCH_MODES = ("off", "cache", "online")
class ResearchCache(path, *, read_only=False)       # schema vegroove.research_cache v1, TTL по источнику
def default_client(contact=None, *, timeout_s=20.0) -> HttpClient     # MB 1.1 с, Wikidata 1.0 с, Discogs 1.1 с
def enrich_tracks(tracks, *, probes=None, mode="cache", cache, client=None, sources=None,
                  progress=None, now=None, max_consecutive_errors=5, match_config=None,
                  checkpoint_every=25, discogs_token=None) -> EnrichResult   # records / stats / warnings
def default_sources(env=None) -> tuple[str, ...]    # musicbrainz, wikidata (+ discogs при VEG_DISCOGS_TOKEN)
def export_review_csv(result, path) -> None
def report_document(records, *, now=None, stats=None, warnings=None) -> dict   # JSON для enrich --report
def format_conflict(conflict, *, now=None) -> str   # строка конфликта с «Data provided by Discogs.»
@dataclass class ResearchRecord   # status, match_confidence, chosen, candidates, facts, user_tags, editorial,
                                  # conflicts, errors, stale
```
* Порядок: MBID/ISRC из тегов файла → поиск записи MusicBrainz → собственная оценка кандидатов
  (сходство артиста/названия, длительность, штрафы live/remix/instrumental/360/edit, бонус за
  official; альбом и год из тегов файла — только «тай-брейк» между уже приемлемыми кандидатами:
  порог принятия проверяется по оценке без них; бонус «самый ранний релиз» — только если у
  файла нет ни альбома, ни года; штраф за несовпавший альбом — только при полном списке релизов)
  → `matched` только при пороге **и** отрыве от второго кандидата, иначе `ambiguous` (ничего не
  применяется). Lookup записи → факты CC0 (MBID записи/артиста/произведения, ISRC, дата первого
  релиза, релиз, группа релиза).
* Wikidata — перекрёстная проверка по MBID (дата публикации, CC0); жанры Wikidata — правимые
  сообществом классификации: они идут в `user_tags` (`scope` recording | work), не в факты.
  Discogs — с `VEG_DISCOGS_TOKEN` (по умолчанию добавляется в источники `online`), TTL 6 ч,
  каждое значение несёт `attribution` и `expires_at`; противоречия → `conflicts` (спорные поля
  планировщик не использует, а показывает как `disputed:*`).
* Теги/жанры MusicBrainz → `user_tags` с `license: CC BY-NC-SA 3.0, non_commercial: true`.
* Режимы: `off` — ничего; `cache` — только кэш (`not_cached` для промахов), сети нет, читаются все
  закэшированные источники (Discogs — без токена, пока не истекли 6 ч); `online` — кэш, затем
  сеть для промахов/просроченного; предохранитель после N подряд сетевых ошибок. Ни одна сетевая
  ошибка не выходит наружу исключением.
* Просроченные записи MusicBrainz/Wikidata (TTL 90 дней для найденного — `DEFAULT_TTL_S`; 14 дней
  для ответов «не найдено»: поиск/ISRC без записей, MBID `not_found`, Wikidata без `qid` —
  `NEGATIVE_TTL_S`) в режиме `cache` **используются**, а
  запись помечается `stale` (с заметкой о возрасте) — план остаётся воспроизводимым; в кэше они
  хранятся до обновления, но не дольше года после истечения (`STALE_RETENTION_S`). Просроченные
  данные Discogs не используются и не хранятся никогда.
* `enrich --report` пишется через `report_document`: просроченные значения Discogs удаляются, у
  каждого остаётся атрибуция, в блоке `discogs` — уведомление о 6-часовом сроке и `expires_at`.

### `ai/` (AI Track Understanding, `interpret`)
```python
PROMPT_VERSION = "interpret-v2"; SCHEMA_VERSION = "interpretation-v2"; AI_CACHE_NAME = "veg_ai_cache.json"
PRIVACY_NOTE; DESCRIPTIVE_FACTS; EVIDENCE_WHITELISTS        # что уходит модели; что может быть доказательством
INTERPRETATION_SCHEMA; def validate_interpretation(data, payload) -> tuple[dict | None, list[str]]
class AIProvider(Protocol): name; model; def complete_json(*, system, user, schema, max_tokens=4000) -> AIResult
AnthropicProvider | OpenAICompatibleProvider | FakeProvider; def get_provider(name=None, *, model=None, base_url=None, ...)
class AICache(path, *, read_only=False)             # schema vegroove.ai_cache v1; ключ — sha256(промпт, схема, провайдер, модель, вход)
def build_payload(track, *, segments=None, research=None, use_community_tags=False) -> dict
def interpret_tracks(tracks, *, provider, mode="cache", cache, segments=None, research=None,
                     retry_invalid=False, ...) -> InterpretResult
```
* Вход модели — без путей, имён файлов, идентификаторов, **артиста и названия** и аудио (чтобы
  модель не «вспоминала» песню): измеренное (`measured.*`, включая сегменты; блоки сегментного
  анализа без определённого значения — темп без BPM, тональность без оценки, траектория/вокал
  без значения — не отправляются), проверенные факты только при `matched`, только описательные
  (`DESCRIPTIVE_FACTS`: год, дата первого релиза, тип группы релиза, дата публикации Wikidata) и
  без спорных полей (спор, который держится только на Discogs, не прячет поле: иначе вход —
  и ключ кэша — менялся бы при истечении 6 ч), теги сообщества только по флагу (без тегов,
  повторяющих артиста/название, и без жанров уровня «произведение»), ручные поля куратора.
* Схема — подмножество structured outputs (все поля обязательны, `additionalProperties:false`, без
  `minimum/maximum`); диапазоны и семантика проверяются валидатором. **Каждое утверждение обязано
  ссылаться на реально переданные поля** (`evidence`); иначе оно отбрасывается (`dropped`).
  Траектория и плотность — только по сегментному анализу (`measured.segments.*`, кроме блоков
  `tempo` и `notes`; сегментный блок `harmony` допускается): общая энергия трека
  (`measured.energy`), BPM и тональность трека (`measured.bpm`, `measured.key`) основанием не
  считаются, без сегментов ответ — `unknown`; ссылки только на поля-провенанс
  (`*_source`, `*_confidence`) или на блок, где кроме провенанса и уточнений
  (`half_double_ambiguous`) ничего нет, доказательством не считаются; BPM/тональность в одиночку
  не основание для ролей. Белые списки (`EVIDENCE_WHITELISTS`): настроение — энергия, динамика,
  траектория/плотность/вокал сегментов, тональность/лад, темп или метки куратора (факты и теги
  сообщества — никогда); `danceability`/`groove` — только ритм (BPM, детали темпа сегментов,
  ударная плотность, ручные `danceability`/`bpm`).
* Anthropic — официальный SDK (`requirements-ai.txt`), модель по умолчанию `claude-opus-5-5`,
  `output_config.format = json_schema` + `effort`, серверный fallback при отказе (`betas`
  `server-side-fallback-2026-07-01`, `fallbacks: "default"`; ответ берётся после блока `fallback`,
  модель — из `usage.iterations`), `stop_reason` проверяется до чтения ответа. OpenAI-совместимый
  (Ollama, LM Studio, llama.cpp) — stdlib, `response_format: json_schema` с откатом на `json_object`;
  ключ `VEG_AI_API_KEY`, а `OPENAI_API_KEY` уходит только на `*.openai.com`.
* Невалидный ответ — один повторный запрос с ошибками, затем отрицательная запись в кэше; у того же
  провайдера/модели она переспрашивается только по `interpret --mode online --retry-invalid`
  (`retry_invalid`), иначе — предупреждение с этой командой (его печатает и `plan`), у `interpret`
  код 4; предохранитель после N подряд ошибок; фатальные ошибки (ключ, доступ, модель) останавливают
  вызовы сразу. Интерпретация хранит `source = ai:<провайдер>/<модель>`, `inputs` (какие разделы
  входа использованы) и `license_note`, если во входе были теги сообщества (`community_tags` —
  любые: теги MusicBrainz под CC BY-NC-SA или жанры Wikidata под CC0; лицензия не проверяется,
  примечание говорит, что лицензии у источников разные и часть из них только некоммерческие).
  Уверенность AI в планировщике ограничена 0.8.

### `scenario_suggest.py` (`suggest`)
AI предлагает **черновик** сценария (фазы, кривая, роли, настроения, допуски, разнообразие) по
брифу пользователя и обезличенной статистике библиотеки (`library_summary`: число треков,
длительность, распределения энергии и BPM, счётчики настроений — без названий и путей).
Имена фаз запрашиваются как ASCII snake_case и дополнительно транслитерируются
(`ascii_identifier`, кириллица → латиница, иначе `phase_<n>`); ответ сначала сверяется со
схемой, части неверного типа отбрасываются или приводятся с записью в предупреждения
(`note: draft repaired`), неисправимый ответ — `ScenarioError` после одного повтора.
Якорей и треков черновик не содержит. Ответ проверяется обычным `parse_scenario` (один повтор с
ошибками), пишется YAML с пометкой «ЧЕРНОВИК» и обоснованием AI; план по нему строит
детерминированный планировщик. Если провайдер так и не дал ответа (сеть, ключ, лимит, сбой
сервера, таймаут) — `SuggestProviderError` (`RuntimeError`, код 5), а не `ScenarioError`.

### `plan_report.py` / `plan_pipeline.py`
* `render_plan_text` — dry-run: `# | Phase | Start | Track | E | Tgt | Dev | BPM | Key | Role |
  Trans | Warnings` + причины по позициям + нарушения + компромиссы + источники данных.
* План-файл `vegroove.plan` v1: порядок вариантов, seed, веса, sha256 сценария, папка
  библиотеки и `rel_path` + `size` + `mtime_ns` каждого трека, который входит хотя бы в один
  вариант. `render` проверяет устаревание только по трекам выбранного варианта: трека нет по
  `rel_path` (или он не анализируется) либо изменились `size`/`mtime_ns`. Новые файлы в библиотеке
  и изменения треков вне варианта план устаревшим не делают.
* `manifest.plan` (только у выходов `render`): сценарий, вариант, запланированный порядок с
  фазами/целями/якорями. QC при его наличии выполняет `plan_order`, `plan_anchors` (якорь
  `position N` считается от начала, `position -N` — от конца опубликованного порядка, так что
  выпавший при нормализации трек не «сдвигает» последний) и `plan_constraints`: каждое
  нетранзитное нарушение плана (`anchor`, `required`, `duration`, `same_song`, ...) — WARNING
  `plan_violation`, поэтому план с нарушениями не получает PASS. В миксе неуверенный переход,
  вызванный только скачком энергии на входе в запланированный контраст, — INFO
  `planned_contrast_transition`; темповая неуверенность остаётся WARNING (слушать). У выходов
  2.0 раздела нет, и проверки не запускаются. `manifest.excluded`/`not_selected` у `render`
  заполняются как в 2.0 (исключения профиля, сбои анализа, исключения сценария — `stage:
  scenario`, подходящие треки вне варианта).
* `render` без `--variant` берёт вариант, который `plan` показал и экспортировал
  (`settings.variant_shown`), без `--cache` — кэш анализа, с которым строился план (абсолютный путь
  в `settings.cache`), если он существует.
* Сравнение (`render_comparison`) называет лучший вариант по (сумма тяжестей нарушений, цель)
  и отдельно — вариант с наименьшей целью, если это другой вариант.
* CSV плана экранирует текст, начинающийся с `= + - @ \t \r` (формулы электронных таблиц).
* Кэши research/AI по умолчанию лежат рядом с кэшем анализа (в папке библиотеки, если `--cache`
  не задан).
* Команды: `analyze --segments`, `enrich`, `interpret`, `suggest`, `plan`, `compare` (без FFmpeg), `render`.
