# VEgroove Automix 2.0
## Engineering specification for Claude Code

You are acting as a senior Python audio/DSP engineer and software architect.

Your task is to refactor an existing `automix.py` prototype into a production-oriented B2B music playlist and mixing pipeline.

The product is VEgroove, a curated background music service for commercial spaces.

**This is an implementation task, not a request for architectural suggestions only.**

Inspect the existing repository, preserve working functionality where reasonable, implement the changes, add automated tests, and provide a final execution report.

Do not replace the entire codebase blindly. Identify reusable components first.

Do not claim that audio quality has been verified unless you actually ran the relevant tests and measurements.

---

# 1. BUSINESS CONTEXT

A commercial client reported the following problems:

1. Tracks cannot be skipped because the playlist is delivered as a single MP3.
2. Loudness varies significantly between songs.
3. Adjacent tracks sometimes have dramatically different moods and energy levels.
4. Morning, afternoon and evening playlists contain inconsistent musical vibes.
5. Too many obscure instrumental or experimental compositions.
6. The client prefers recognizable international pop music.
7. Russian-language songs should be excluded.
8. Desired listening experience resembles Yandex Wave: coherent, accessible, pleasant, with enough variety to avoid repetition.

The client likes the existing music selection overall. The objective is to improve consistency, usability and audio quality, not replace human curation with arbitrary AI recommendations.

Critical priority:

**LOUDNESS CONSISTENCY MUST BE IMPLEMENTED AND TESTED BEFORE ANY OPTIONAL FEATURES.**

---

# 2. TARGET ARCHITECTURE

Refactor the existing monolithic script into logical modules.

Suggested structure:

```text
veg_mixer/
    __init__.py
    cli.py
    config.py
    models.py
    cache.py
    analysis.py
    loudness.py
    sequencing.py
    transitions.py
    renderer.py
    playlist.py
    quality_control.py
    reports.py

tests/
    test_loudness.py
    test_sequencing.py
    test_transitions.py
    test_cache.py
    test_playlist.py
    test_quality_control.py

automix.py
requirements.txt
README.md
```

The existing `automix.py` should remain available as a backward-compatible CLI entry point where practical.

Avoid unnecessary abstraction and heavyweight infrastructure.

Use Python 3.11+.

Preferred dependencies:

- FFmpeg / ffprobe
- NumPy
- librosa
- soundfile
- scipy
- pydub only where appropriate
- pytest

Do not introduce cloud dependencies.

---

# 3. P0: LOUDNESS NORMALIZATION

Implement a dedicated `LoudnessNormalizer` service.

Use FFmpeg's EBU R128 loudnorm filter.

Do NOT use peak normalization as a substitute for perceptual loudness normalization.

## Default targets

```python
TARGET_LUFS = -16.0
TARGET_TRUE_PEAK_DBTP = -1.5
TARGET_LRA = 11.0

LUFS_TOLERANCE = 1.0
TRUE_PEAK_TOLERANCE_DB = 0.2
```

These parameters must be configurable.

LRA is a processing target, not a guarantee that every track must have exactly 11 LU.

## Required processing

Implement two-pass normalization.

First pass:

- Measure input integrated LUFS.
- Measure input True Peak.
- Measure input LRA.
- Measure input threshold.
- Parse FFmpeg JSON output.
- Validate all returned numerical values.

Second pass:

- Supply measured values to loudnorm.
- Enable linear normalization when feasible.
- Allow dynamic processing when required.
- Explicitly account for loudnorm falling back to dynamic processing.
- Use an appropriate internal sample rate for accurate True Peak measurement.
- Preserve stereo.
- Avoid unnecessary intermediate lossy encoding.

Use temporary PCM/WAV or another suitable lossless intermediate format.

Never repeatedly encode MP3 during processing.

## Important constraints

True Peak safety takes precedence over exact LUFS matching.

Do not aggressively compress or limit a highly dynamic master merely to achieve the target number.

If the target cannot be reached without violating peak constraints, flag the track for review.

Do not silently declare it normalized.

## Post-export verification

Re-analyze the actual encoded output file.

Check:

- Integrated LUFS.
- True Peak.
- Duration.
- Channel count.
- Sample rate.
- Encoding success.
- File integrity.

A successful FFmpeg exit code is not sufficient evidence of successful normalization.

Compare actual measured values against configured targets.

If normalization fails, report the specific reason.

Do not mark failed files as successfully processed.

## Required output metadata

```json
{
  "target_lufs": -16.0,
  "measured_lufs": -16.2,
  "true_peak_dbtp": -1.7,
  "normalization_status": "passed",
  "requires_review": false
}
```

These numbers are illustrative, not hardcoded expected results.

## Additional QC

Measure short-term loudness and detect major differences between adjacent tracks.

Flag suspicious transitions for listening review.

Do not automatically apply excessive dynamic compression to fix short-term loudness differences.

---

# 4. P0: PLAYLIST MODE

The system must support two separate output modes.

## Mode A: Playlist

This is the primary mode.

Export every selected composition as a separate normalized audio file.

Generate:

```text
output/
    playlist.m3u8
    manifest.json
    tracklist.txt

    tracks/
        001_artist_title.mp3
        002_artist_title.mp3
        003_artist_title.mp3
```

Requirements:

- Every song remains individually playable.
- Users can skip, repeat and seek individual tracks.
- Track order is preserved.
- Generate valid extended M3U entries with duration and title.
- Use relative paths.
- Preserve useful metadata when available.
- Handle Unicode filenames correctly.
- Avoid filename collisions.
- Do not overwrite source files.
- Use safe temporary files and atomic replacement where possible.

Important:

M3U8 is a playlist file, not a standalone playback application.

Actual skipping and playback controls depend on the player.

Do not claim that generating M3U8 alone implements a player interface.

Also export a JSON manifest suitable for future integration with a web player.

## Mode B: Continuous Mix

Preserve the existing ability to generate a single continuous MP3.

However, this must use the improved transition engine and quality-control pipeline.

---

# 5. P0: MUSIC METADATA AND VIBE MODEL

The current BPM + Camelot approach is insufficient.

Introduce a structured metadata model.

Each track should support:

```python
@dataclass
class TrackFeatures:
    path: Path
    duration_s: float

    bpm: float | None
    camelot: str | None
    key_confidence: float | None

    energy: float | None
    danceability: float | None
    vocal_presence: float | None

    genre: str | None
    mood: str | None
    vibe_tags: list[str]

    language: str | None
    familiarity: float | None

    approved: bool
    excluded: bool
    requires_review: bool
```

Energy, danceability, familiarity, mood and genre must not be presented as reliable measurements unless a suitable analysis or metadata source actually exists.

Support manually curated metadata via CSV or JSON.

Manual editorial metadata takes precedence over automatic estimates.

Do not fabricate familiarity scores or artist popularity.

Unknown values must remain unknown.

Provide an import/export mechanism so a human curator can edit track classifications without modifying Python code.

---

# 6. P0: CLIENT-SPECIFIC CURATION RULES

Create an editable client profile.

Example:

```yaml
client: commercial_pop

language:
  allowed:
    - en
    - fr
    - es
    - it
  excluded:
    - ru

preferences:
  recognizable_pop: high
  accessible_melodic_music: high
  experimental_instrumental: low
  aggressive_rap: low

vibe_consistency: high
energy_jumps: low

manual_review_required: true
```

This configuration expresses editorial preferences, not an instruction to download or discover music automatically.

The system must never claim that an unknown track is popular, familiar, or suitable without supporting metadata.

Implement exclusion rules for explicitly banned tracks, artists and tags.

Support manual allowlists and blocklists.

Unknown language should be flagged for review, not silently treated as English.

---

# 7. P1: SMART SEQUENCING

Replace the current greedy harmonic sorter.

Current implementation:

```python
score = bpm_diff - (50 if cam_ok else 0)
```

This creates excessive preference for harmonic compatibility and ignores mood.

Implement a weighted sequencing system.

Suggested initial weights:

```python
WEIGHTS = {
    "vibe_similarity": 0.35,
    "energy_continuity": 0.30,
    "bpm_continuity": 0.15,
    "harmonic_compatibility": 0.10,
    "genre_continuity": 0.10,
}
```

These are configurable starting values.

Normalize individual scoring components before combining them.

Unknown features must not be interpreted as zero similarity or maximum compatibility.

Use available features with appropriate confidence handling.

The algorithm must prioritize:

1. Editorial restrictions.
2. Mood compatibility.
3. Energy continuity.
4. Genre consistency.
5. Tempo continuity.
6. Harmonic compatibility.

Prevent major unexplained energy jumps.

Avoid consecutive repetition of the same artist where alternatives exist.

Avoid long runs of excessively similar songs.

Introduce controlled variation without destroying the overall playlist identity.

The output should resemble a thoughtfully curated radio station, not an ascending BPM spreadsheet.

## Daypart profiles

Implement three configurable profiles:

Morning:
- Gentle to moderate energy.
- Accessible melodic music.
- Avoid aggressive openings.

Afternoon:
- Moderate energy.
- Recognizable pop.
- Consistent working atmosphere.

Evening:
- Moderate to moderately high energy.
- More groove and rhythmic material.
- Avoid abrupt stylistic changes.

Daypart labels must actually affect selection and sequencing.

Do not simply split one global tracklist into three equal parts.

---

# 8. P1: TRANSITION ENGINE

Fix the existing transition implementation.

Current critical problem:

```python
mix = apply_fx(mix, fx)
```

This applies effects to the accumulated mix.

Effects must operate only on the outgoing track's transition region.

Previously rendered audio must remain immutable.

## Crossfade

- Calculate crossfade using actual adjacent track durations.
- Respect configured maximum overlap.
- Prevent invalid crossfade lengths.
- Support disabling crossfades.
- Support configurable fade curves.
- Consider equal-power crossfades where appropriate.
- Measure the resulting transition to detect peak overshoot.

Do not select crossfade duration using BPM difference alone.

## FX

Keep these options:

- clean
- filter_sweep
- echo_tail
- filter+echo

But disable random FX by default for commercial playlists.

FX must be optional and configurable.

Do not apply effects merely to make a transition sound more like a DJ mix.

A clean transition is preferable to a distracting effect.

Fix echo-tail implementation so delay taps have correct offsets, gain staging and tail handling.

Ensure transition processing cannot unexpectedly modify previously rendered material.

## Phrase awareness

If reliable phrase detection is unavailable, do not pretend to have it.

Provide conservative transition behavior and flag uncertain transitions for manual review.

---

# 9. P1: CACHE

Replace path-only cache identity.

Cache validity must account for:

- Absolute source path.
- File size.
- Modification timestamp.
- Analysis algorithm version.
- Relevant analysis parameters.

Optional content hashing may be supported.

Separate cached audio analysis from cached editorial metadata.

Changing the normalization target must not force unnecessary BPM re-analysis.

Use atomic cache writes.

Handle malformed or interrupted cache files gracefully.

Support migration of legacy cache entries.

---

# 10. P1: QUALITY CONTROL

Implement automated QC for both output modes.

## Playlist QC

Verify:

- Every referenced file exists.
- Every file is decodable.
- All files have acceptable loudness.
- True Peak is within configured limits.
- Track order matches the manifest.
- No source tracks are accidentally duplicated.
- No excluded tracks appear.
- All files have nonzero duration.

## Continuous Mix QC

Verify:

- Final integrated loudness.
- Final True Peak.
- Total duration.
- Unexpected silence.
- Unexpected clipping.
- Transition timing.
- Tracklist timestamps.

Do not assume that normalizing individual tracks guarantees correct loudness after crossfading.

Analyze the final rendered mix separately.

Silence detection should distinguish intentional quiet musical passages from suspicious gaps where possible.

Warnings must include timestamps.

---

# 11. P1: REPORTING

Generate a machine-readable JSON report and a human-readable text report.

Required summary:

```text
VEgroove Audio QC

Tracks processed:
Tracks passed:
Tracks failed:
Tracks requiring review:

Target LUFS:
Measured LUFS range:
Maximum measured True Peak:

Excluded tracks:
Unknown metadata:
Suspicious energy transitions:
Silence warnings:

Overall status:
```

Use explicit statuses:

- PASS
- WARNING
- FAIL

Do not report PASS when critical checks were skipped.

For individual tracks, report:

- Original loudness.
- Final loudness.
- Original True Peak.
- Final True Peak.
- Processing status.
- Warnings.

For sequencing, report the reason for each flagged transition.

---

# 12. CLI REQUIREMENTS

Implement commands equivalent to:

```bash
python automix.py analyze \
  --folder ./music

python automix.py normalize \
  --folder ./music \
  --output ./normalized \
  --target-lufs -16 \
  --true-peak -1.5

python automix.py playlist \
  --folder ./music \
  --profile ./profiles/client.yaml \
  --output ./output

python automix.py mix \
  --folder ./music \
  --profile ./profiles/client.yaml \
  --output ./output/mix.mp3

python automix.py qc \
  --folder ./output
```

The exact CLI implementation may differ if a better design is justified.

Provide `--dry-run`.

Provide useful progress reporting.

Provide clear errors when FFmpeg is missing.

Avoid silently skipping broken files.

---

# 13. AUTOMATED TESTS

Use pytest.

Generate synthetic audio fixtures using NumPy and WAV.

Do not depend on copyrighted commercial music in automated tests.

Required tests:

### Loudness

1. Generate two signals with different initial loudness.
2. Normalize both.
3. Verify their measured integrated LUFS approaches the target within tolerance when feasible.
4. Verify True Peak constraints.
5. Verify behavior when loudness and peak targets cannot both be achieved.
6. Verify malformed FFmpeg measurement handling.
7. Verify final MP3 measurement, not only intermediate PCM.

### Sequencing

1. Test harmonic compatibility.
2. Test that mood and energy can outweigh harmonic compatibility.
3. Test exclusion rules.
4. Test unknown metadata handling.
5. Test artist repetition constraints.
6. Test daypart profiles.

### Transitions

1. Verify transition duration.
2. Verify output duration accounting.
3. Verify that FX does not alter earlier rendered audio.
4. Verify peak handling during overlap.
5. Verify short-track behavior.

### Playlist

1. Validate M3U8 syntax.
2. Validate relative paths.
3. Validate Unicode filenames.
4. Validate manifest ordering.
5. Verify source files remain untouched.

### Cache

1. Verify cache reuse.
2. Verify invalidation after source modification.
3. Verify legacy migration.
4. Verify recovery from invalid cache content.

---

# 14. ACCEPTANCE CRITERIA

The implementation is complete only when:

- Individual normalized MP3 tracks can be exported.
- A valid M3U8 playlist is generated.
- Loudness is measured before and after normalization.
- True Peak is verified on encoded outputs.
- Loudness failures are explicitly reported.
- Existing source files remain untouched.
- Playlist sequencing respects editorial constraints.
- Daypart profiles affect the actual result.
- Continuous Mix mode remains available.
- Transition FX no longer modifies the accumulated mix.
- Automated tests pass.
- README documents installation and CLI usage.
- A sample client profile is provided.
- An example QC report is generated.

Do not use mocked FFmpeg results as the only evidence that audio normalization works.

Include at least one real FFmpeg integration test.

If FFmpeg is unavailable in the execution environment, clearly state which integration tests could not run.

---

# 15. IMPLEMENTATION ORDER

Implement in this exact order:

Phase 1:
- Inspect existing code.
- Identify reusable functionality.
- Establish test baseline.
- Implement loudness normalization.
- Implement post-export verification.

Phase 2:
- Implement individual track export.
- Implement M3U8 and manifest generation.
- Implement QC reporting.

Phase 3:
- Implement metadata model.
- Implement client profiles.
- Implement vibe-aware sequencing.

Phase 4:
- Fix continuous mixing and transition engine.
- Fix caching.
- Add regression tests.

Phase 5:
- Run tests.
- Perform an end-to-end smoke test.
- Update documentation.

Do not spend time polishing optional transition effects while loudness normalization remains unimplemented.

---

# 16. FINAL DELIVERY

At the end, provide:

1. Summary of changes.
2. Files created and modified.
3. Exact commands to install and run.
4. Test results.
5. Audio measurement results from integration tests.
6. Known limitations.
7. Any features requiring manual music curation.
8. Confirmation of whether every acceptance criterion was satisfied.

Distinguish implemented functionality from proposed functionality.

Do not invent test results.

Begin by inspecting the existing repository and implementing Phase 1.
