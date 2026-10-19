# VEgroove Automix 2.0 — архитектура и контракты модулей

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
    def put(self, source: Path, section: str, value: dict, *, version: str, params: dict | None = None) -> None: ...
    def save(self) -> None: ...            # атомарно; no-op если не было изменений
    stats: dict[str, int]                  # hits, misses, invalidated, migrated
    warnings: list[str]                    # напр. «повреждённый кэш сохранён как *.corrupt-…»
```
* Идентичность: абсолютный путь + размер + `mtime_ns` (+ sha256 при `use_content_hash`).
  Смена отпечатка инвалидирует **все** секции файла.
* Секции независимы (`probe`, `bpm`, `key`, `energy`, `loudness`): у каждой своя
  `version` и хэш `params`. Смена цели нормализации меняет только `loudness`.
* Повреждённый/обрезанный JSON → файл переименовывается в `*.corrupt-<n>`,
  работа продолжается с пустым кэшем + предупреждение.
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
def detect_key(path: Path, cache: AnalysisCache, cfg: AnalysisConfig) -> tuple[str, float]: ...
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
                    next_head_db: np.ndarray | None = None, rng=None) -> TransitionPlan
def fade_curves(n: int, curve: str) -> tuple[np.ndarray, np.ndarray]           # (fade_out, fade_in)
def apply_fx(region: np.ndarray, sr: int, fx: str) -> np.ndarray                # только регион уходящего трека
def render_mix(inputs: list[MixInput], out_wav: Path, cfg: TransitionConfig, *,
               ceiling_dbtp: float, sample_rate: int = 44100, seed: int = 0,
               progress=None) -> MixRenderResult
def write_mix_tracklist(result: MixRenderResult, inputs: list[MixInput], path: Path) -> None
```
* Потоковый рендер: в памяти — текущий трек + хвост предыдущего; отрендеренное
  записано в файл и больше не меняется.
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
def scan_silence(tools, path: Path, cfg: QCConfig, *, boundaries_s: list[float] = ()) -> list[dict]
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
* `manifest.excluded[*].stage` ∈ {`curation`, `analysis`, `normalization`};
  `manifest.not_selected` — подходящие треки, которые секвенсор никуда не поставил, с причиной.
* Микс: `mix.published`, при публикации — `sha256` и `size_bytes` MP3; неопубликованный микс → FAIL.
  Финальный проход громкости микса никогда не использует динамический режим `loudnorm`.
* `qc` берёт цели (громкость, частота, битрейт) из манифеста; ключ командной строки
  переопределяет только своё поле.
* Коды выхода: 0 PASS, 1 FAIL, 2 ввод, 3 окружение, 4 WARNING, 5 ошибка выполнения.
