"""
automix.py — BPM-sorted crossfade mixer
-----------------------------------------
Dependencies:
    pip install librosa pydub
    ffmpeg in PATH
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
try:
    from pydub import AudioSegment
except ImportError:  # allow `import automix` / `--help` without the audio backend
    AudioSegment = None

# ---------------------------------------------------------------------------
# CONFIG (tuning constants — not runtime state)
# ---------------------------------------------------------------------------

TARGET_DURATION_S   = 3.5 * 3600   # 12600 s
TOLERANCE_S         = 10 * 60      # ±10 min
SILENCE_THRESH_DBFS = -40.0
LIBROSA_SR          = 22050
FADE_IN_S           = 3
FADE_OUT_S          = 5

# ---------------------------------------------------------------------------
# CAMELOT / KEY
# ---------------------------------------------------------------------------

CAMELOT_MAP = {
    (0,  1): '8B',  (0,  0): '5A',
    (1,  1): '3B',  (1,  0): '12A',
    (2,  1): '10B', (2,  0): '7A',
    (3,  1): '5B',  (3,  0): '2A',
    (4,  1): '12B', (4,  0): '9A',
    (5,  1): '7B',  (5,  0): '4A',
    (6,  1): '2B',  (6,  0): '11A',
    (7,  1): '9B',  (7,  0): '6A',
    (8,  1): '4B',  (8,  0): '1A',
    (9,  1): '11B', (9,  0): '8A',
    (10, 1): '6B',  (10, 0): '3A',
    (11, 1): '1B',  (11, 0): '10A',
}

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---------------------------------------------------------------------------
# MIX CONFIG
# ---------------------------------------------------------------------------

@dataclass
class MixConfig:
    source_dir: Path
    output_file: Path
    cache_file: Path
    bitrate: str


# ---------------------------------------------------------------------------
# BPM CACHE
# ---------------------------------------------------------------------------

class BPMCache:
    def __init__(self, path: Path):
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def get_entry(self, key: str) -> dict:
        """Return per-track dict; migrates legacy float entries."""
        val = self._data.get(key)
        if isinstance(val, (int, float)):
            return {"bpm": float(val)}
        return val if isinstance(val, dict) else {}

    def set_field(self, key: str, field: str, value) -> None:
        entry = self.get_entry(key)
        entry[field] = value
        self._data[key] = entry


# ---------------------------------------------------------------------------
# BPM + KEY DETECTION  (mutate cache; caller saves once)
# ---------------------------------------------------------------------------

def detect_bpm(path: Path, cache: BPMCache) -> float:
    key = str(path)
    entry = cache.get_entry(key)
    if "bpm" in entry:
        return entry["bpm"]
    y, sr = librosa.load(str(path), sr=LIBROSA_SR, mono=True, offset=30, duration=90)
    if len(y) < sr * 30:  # < 30 s captured from offset — too little / unrepresentative; analyze whole track
        y, sr = librosa.load(str(path), sr=LIBROSA_SR, mono=True, duration=90)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    cache.set_field(key, "bpm", bpm)
    return bpm


def detect_key(path: Path, cache: BPMCache) -> tuple[str, float]:
    """Detect key via chroma + Krumhansl-Kessler profiles.
    Returns (camelot_code, confidence).
    """
    key = str(path)
    entry = cache.get_entry(key)
    if "camelot" in entry:
        return entry["camelot"], entry.get("key_confidence", 0.0)

    y, sr = librosa.load(str(path), sr=LIBROSA_SR, mono=True, offset=30, duration=60)
    if len(y) < sr:  # < 1 s captured from offset — retry from start
        y, sr = librosa.load(str(path), sr=LIBROSA_SR, mono=True, duration=60)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)

    best_corr = -1.0
    best_pitch, best_mode = 0, 1
    for shift in range(12):
        shifted = np.roll(chroma_avg, -shift)
        maj_corr = float(np.corrcoef(shifted, MAJOR_PROFILE)[0, 1])
        min_corr = float(np.corrcoef(shifted, MINOR_PROFILE)[0, 1])
        if maj_corr > best_corr:
            best_corr, best_pitch, best_mode = maj_corr, shift, 1
        if min_corr > best_corr:
            best_corr, best_pitch, best_mode = min_corr, shift, 0

    camelot = CAMELOT_MAP.get((best_pitch, best_mode), '1A')
    cache.set_field(key, "camelot", camelot)
    cache.set_field(key, "key_confidence", round(best_corr, 3))
    return camelot, best_corr


# ---------------------------------------------------------------------------
# SILENCE TRIMMING
# ---------------------------------------------------------------------------

def trim_silence(
    audio: AudioSegment,
    silence_thresh: float = SILENCE_THRESH_DBFS,
    min_silence_len: int = 200,
) -> tuple[AudioSegment, float, float]:
    """Trim leading and trailing silence from an AudioSegment.
    Returns (trimmed_segment, head_trimmed_ms, tail_trimmed_ms).
    """
    chunk_size = 50  # ms

    # --- trim head ---
    start_trim = 0
    while start_trim < len(audio):
        chunk = audio[start_trim: start_trim + chunk_size]
        if chunk.dBFS > silence_thresh:
            break
        start_trim += chunk_size

    # --- trim tail ---
    end_trim = len(audio)
    while end_trim > start_trim:
        chunk = audio[end_trim - chunk_size: end_trim]
        if chunk.dBFS > silence_thresh:
            break
        end_trim -= chunk_size

    # 100 ms safety buffer so we don't clip the attack/release
    start_trim = max(0, start_trim - 100)
    end_trim = min(len(audio), end_trim + 100)

    head_ms = start_trim
    tail_ms = len(audio) - end_trim

    # only apply trim if at least min_silence_len was found
    if head_ms < min_silence_len:
        head_ms = 0
        start_trim = 0
    if tail_ms < min_silence_len:
        tail_ms = 0
        end_trim = len(audio)

    return audio[start_trim:end_trim], head_ms, tail_ms


# ---------------------------------------------------------------------------
# CROSSFADE SELECTION (BPM-based, post-trim)
# ---------------------------------------------------------------------------

def pick_crossfade(bpm_a: float, bpm_b: float, dur_a_ms: int, dur_b_ms: int) -> int:
    """Return crossfade ms based on BPM difference between adjacent tracks."""
    diff = abs(bpm_a - bpm_b)
    if diff > 20:
        xfade_ms = 7000
    elif diff < 5:
        xfade_ms = 4000
    else:
        xfade_ms = 5000
    # cap at 30% of the shorter track
    max_ms = int(min(dur_a_ms, dur_b_ms) * 0.30)
    return min(xfade_ms, max_ms)


# ---------------------------------------------------------------------------
# TRACK SELECTION
# ---------------------------------------------------------------------------

# reserved for --target-duration mode
def select_tracks(sorted_tracks: list[dict]) -> list[dict]:
    """
    Pick the first N tracks (in BPM order) whose total duration ≈ TARGET_DURATION_S.
    Stops when adding the next track would exceed target + tolerance.
    """
    selected = []
    total = 0.0
    for t in sorted_tracks:
        if total + t["duration_s"] > TARGET_DURATION_S + TOLERANCE_S:
            break
        selected.append(t)
        total += t["duration_s"]
        if total >= TARGET_DURATION_S - TOLERANCE_S:
            break
    return selected


# ---------------------------------------------------------------------------
# HARMONIC SORT (Camelot-aware greedy chain)
# ---------------------------------------------------------------------------

def camelot_compatible(cam_a: str, cam_b: str) -> bool:
    na, la = int(cam_a[:-1]), cam_a[-1]
    nb, lb = int(cam_b[:-1]), cam_b[-1]
    if na == nb and la == lb:          # same key
        return True
    if la == lb and (na - nb) % 12 in (1, 11):  # adjacent on wheel, same mode
        return True
    if na == nb and la != lb:          # relative major/minor
        return True
    return False


def sort_tracks_harmonic(tracks: list[dict]) -> list[dict]:
    """Greedy chain starting from the slowest BPM track.
    Each step picks the candidate with lowest (bpm_diff − 50 if camelot_ok).
    """
    remaining = sorted(tracks, key=lambda t: t["bpm"])
    chain = [remaining.pop(0)]
    while remaining:
        current = chain[-1]
        best_idx, best_score = 0, float("inf")
        for i, candidate in enumerate(remaining):
            bpm_diff = abs(current["bpm"] - candidate["bpm"])
            cam_ok = camelot_compatible(
                current.get("camelot", "1A"),
                candidate.get("camelot", "1A"),
            )
            score = bpm_diff - (50 if cam_ok else 0)
            if score < best_score:
                best_score, best_idx = score, i
        chain.append(remaining.pop(best_idx))
    return chain


# ---------------------------------------------------------------------------
# TRANSITION FX
# ---------------------------------------------------------------------------

FX_OPTIONS = ["clean", "filter_sweep", "echo_tail", "filter+echo"]
FX_WEIGHTS = [0.25, 0.35, 0.25, 0.15]


def apply_filter_sweep(audio_segment: AudioSegment, duration_ms: int = 4000) -> AudioSegment:
    """Low-pass sweep on the tail of audio (8 steps, 10 kHz → 200 Hz)."""
    from pydub.effects import low_pass_filter
    if len(audio_segment) <= duration_ms:
        return audio_segment
    tail = audio_segment[-duration_ms:]
    body = audio_segment[:-duration_ms]
    chunk_len = duration_ms // 8
    frequencies = [10000, 6000, 3500, 2000, 1200, 700, 400, 200]
    filtered_tail = AudioSegment.empty()
    for i, freq in enumerate(frequencies):
        filtered_tail += low_pass_filter(tail[i * chunk_len:(i + 1) * chunk_len], freq)
    return body + filtered_tail


def apply_echo_tail(audio_segment: AudioSegment, tail_ms: int = 2000, decay_db: int = 15) -> AudioSegment:
    """3-tap delay echo on the tail with increasing attenuation."""
    if len(audio_segment) <= tail_ms:
        return audio_segment
    tail = audio_segment[-tail_ms:]
    result = audio_segment
    for delay, decay in zip([80, 160, 300], [decay_db, decay_db + 6, decay_db + 12]):
        echo_shifted = AudioSegment.silent(duration=delay) + (tail - decay)
        overlay_pos = len(audio_segment) - tail_ms
        if overlay_pos > 0:
            result = result.overlay(echo_shifted, position=overlay_pos)
    return result


def choose_transition_fx() -> str:
    return random.choices(FX_OPTIONS, weights=FX_WEIGHTS)[0]


def apply_fx(mix: AudioSegment, fx: str) -> AudioSegment:
    if fx == "filter_sweep":
        return apply_filter_sweep(mix)
    if fx == "echo_tail":
        return apply_echo_tail(mix)
    if fx == "filter+echo":
        return apply_echo_tail(apply_filter_sweep(mix))
    return mix  # 'clean'


# ---------------------------------------------------------------------------
# MIX
# ---------------------------------------------------------------------------

def build_mix(tracks: list[dict], output_file: Path) -> tuple[AudioSegment, list[dict], int]:
    """Build the mix: trim silence, apply transition FX, BPM-based crossfade.
    Returns (mix_segment, tracklist, seed) where seed makes FX reproducible.
    """
    seed_value = int(hashlib.md5(output_file.name.encode()).hexdigest()[:8], 16)
    random.seed(seed_value)
    print(f"  Random seed: {seed_value}")

    mix = None
    tracklist = []
    mix_pos_ms = 0
    fx_history: list[str] = []

    for i, t in enumerate(tracks):
        print(f"  [{i+1}/{len(tracks)}] Loading + trimming: {t['name']}")
        seg = AudioSegment.from_mp3(str(t["path"]))
        seg, head_ms, tail_ms = trim_silence(seg)
        print(
            f"    Trimmed: {head_ms/1000:.2f}s head / {tail_ms/1000:.2f}s tail  "
            f"| New dur: {len(seg)/1000:.1f}s"
        )

        if mix is None:
            xfade_ms = 0
            fx = None
            mix = seg.fade_in(FADE_IN_S * 1000)
        else:
            prev = tracks[i - 1]
            xfade_ms = pick_crossfade(
                prev["bpm"], t["bpm"],
                len(mix),
                len(seg),
            )
            fx = choose_transition_fx()
            while len(fx_history) >= 2 and fx_history[-1] == fx and fx_history[-2] == fx:
                fx = choose_transition_fx()
            fx_history.append(fx)
            print(f"    FX: {fx}  |  Crossfade: {xfade_ms // 1000}s  |  Key: {t.get('camelot', '?')}")
            mix = apply_fx(mix, fx)
            mix_pos_ms -= xfade_ms
            mix = mix.append(seg, crossfade=xfade_ms)

        tracklist.append({
            "num": i + 1,
            "time_ms": mix_pos_ms,
            "bpm": t["bpm"],
            "camelot": t.get("camelot", "?"),
            "fx": fx or "—",
            "xfade_ms": xfade_ms,
            "name": t["name"],
        })
        mix_pos_ms += len(seg)

    mix = mix.fade_out(FADE_OUT_S * 1000)
    return mix, tracklist, seed_value


# ---------------------------------------------------------------------------
# POST-MIX SILENCE DETECTOR
# ---------------------------------------------------------------------------

SILENCE_SCAN_WINDOW_MS = 500       # analysis window size
SILENCE_SCAN_THRESH_DBFS = -35.0   # below this = silent window
SILENCE_SCAN_MIN_WINDOWS = 3       # 3 × 500 ms = 1.5 s minimum gap


def find_between_tracks(gap_start_ms: int, tracklist: list[dict]) -> tuple[int, int]:
    """Return (track_num_before, track_num_after) for a gap at gap_start_ms."""
    before_num = 1
    after_num = 1
    for entry in tracklist:
        if entry["time_ms"] <= gap_start_ms:
            before_num = entry["num"]
        else:
            after_num = entry["num"]
            break
    else:
        after_num = tracklist[-1]["num"]
    return before_num, after_num


def detect_silence_gaps(mix: AudioSegment, tracklist: list[dict]) -> list[dict]:
    """
    Scan mix in SILENCE_SCAN_WINDOW_MS windows.
    Collect runs of SILENCE_SCAN_MIN_WINDOWS+ consecutive silent windows.
    Returns list of gap dicts: {start_ms, end_ms, duration_ms, before, after}.
    """
    gaps = []
    mix_len = len(mix)
    run_start_ms = None
    run_count = 0

    pos = 0
    while pos + SILENCE_SCAN_WINDOW_MS <= mix_len:
        window = mix[pos: pos + SILENCE_SCAN_WINDOW_MS]
        is_silent = window.dBFS < SILENCE_SCAN_THRESH_DBFS

        if is_silent:
            if run_start_ms is None:
                run_start_ms = pos
            run_count += 1
        else:
            if run_start_ms is not None and run_count >= SILENCE_SCAN_MIN_WINDOWS:
                gap_end_ms = pos
                before, after = find_between_tracks(run_start_ms, tracklist)
                gaps.append({
                    "start_ms": run_start_ms,
                    "end_ms": gap_end_ms,
                    "duration_ms": gap_end_ms - run_start_ms,
                    "before": before,
                    "after": after,
                })
            run_start_ms = None
            run_count = 0

        pos += SILENCE_SCAN_WINDOW_MS

    # flush any trailing run
    if run_start_ms is not None and run_count >= SILENCE_SCAN_MIN_WINDOWS:
        before, after = find_between_tracks(run_start_ms, tracklist)
        gaps.append({
            "start_ms": run_start_ms,
            "end_ms": mix_len,
            "duration_ms": mix_len - run_start_ms,
            "before": before,
            "after": after,
        })

    return gaps


def print_silence_report(gaps: list[dict]):
    if not gaps:
        print("  ✓ No silence gaps detected.")
        return
    for g in gaps:
        dur_s = g["duration_ms"] / 1000
        print(
            f"  ⚠ SILENCE GAP: {dur_s:.1f}s at "
            f"{ms_to_hms(g['start_ms'])} — {ms_to_hms(g['end_ms'])} "
            f"(between track {g['before']} and {g['after']})"
        )


# ---------------------------------------------------------------------------
# TRACKLIST WRITER
# ---------------------------------------------------------------------------

def ms_to_hms(ms: int) -> str:
    s = ms // 1000
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def write_tracklist(
    tracklist: list[dict],
    mix_dur_ms: int,
    seed: int,
    gaps: list[dict] | None,
    output_file: Path,
):
    log_path = output_file.with_name(output_file.stem + "_tracklist.txt")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    total_h = mix_dur_ms // 3600000
    total_m = (mix_dur_ms % 3600000) // 60000
    total_s = (mix_dur_ms % 60000) // 1000

    # --- camelot stats ---
    transitions = len(tracklist) - 1
    compatible = sum(
        1 for i in range(1, len(tracklist))
        if camelot_compatible(
            tracklist[i - 1].get("camelot", "1A"),
            tracklist[i].get("camelot", "1A"),
        )
    )
    incompatible = transitions - compatible

    # --- fx counts ---
    fx_counts: dict[str, int] = {fx: 0 for fx in FX_OPTIONS}
    for entry in tracklist[1:]:  # first track has no FX
        fx = entry.get("fx", "—")
        if fx in fx_counts:
            fx_counts[fx] += 1

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"TRACKLIST: {output_file.name}\n")
        f.write(f"Generated: {now}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Total duration: {total_h}h {total_m:02d}m {total_s:02d}s\n")
        f.write(f"Total tracks: {len(tracklist)}\n")
        f.write("\n")
        f.write(f"{'#':<4} | {'Time':<10} | {'BPM':<6} | {'Key':<5} | {'Xfade':<6} | {'FX':<14} | Track\n")
        f.write("-" * 4 + "-|-" + "-" * 10 + "-|-" + "-" * 6 + "-|-" + "-" * 5 + "-|-" + "-" * 6 + "-|-" + "-" * 14 + "-|-" + "-" * 30 + "\n")
        for entry in tracklist:
            xfade_str = f"{entry['xfade_ms']//1000}s" if entry["xfade_ms"] else "—"
            f.write(
                f"{entry['num']:<4} | "
                f"{ms_to_hms(entry['time_ms']):<10} | "
                f"{entry['bpm']:<6.1f} | "
                f"{entry.get('camelot', '?'):<5} | "
                f"{xfade_str:<6} | "
                f"{entry.get('fx', '—'):<14} | "
                f"{entry['name']}\n"
            )

        f.write("\n=== CAMELOT STATS ===\n")
        cam_pct = (compatible / transitions * 100) if transitions else 0
        f.write(f"Compatible transitions: {compatible}/{transitions} ({cam_pct:.0f}%)\n")
        f.write(f"Incompatible transitions: {incompatible}/{transitions}\n")

        if gaps is not None:
            f.write("\n=== SILENCE WARNINGS ===\n")
            if not gaps:
                f.write("✓ No silence gaps detected.\n")
            else:
                for g in gaps:
                    dur_s = g["duration_ms"] / 1000
                    f.write(
                        f"⚠ {dur_s:.1f}s gap at {ms_to_hms(g['start_ms'])} "
                        f"(between #{g['before']} and #{g['after']})\n"
                    )
                f.write(f"Total gaps found: {len(gaps)}\n")

        f.write("\n=== FX SUMMARY ===\n")
        fx_total = sum(fx_counts.values())
        for fx_name, count in fx_counts.items():
            fx_pct = (count / fx_total * 100) if fx_total else 0
            f.write(f"{fx_name}: {count} ({fx_pct:.0f}%)\n")

    print(f"  Tracklist: {log_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BPM-sorted crossfade mixer")
    parser.add_argument("--folder", required=True,
                        help="Path to the folder containing MP3 files to mix")
    parser.add_argument("--output",
                        help="Output MP3 base path (parts will be numbered automatically)")
    parser.add_argument("--cache",
                        help="BPM/key cache JSON path (default: <folder>/bpm_cache.json)")
    parser.add_argument("--parts", type=int, default=1,
                        help="Split into N equal mixes (default: 1)")
    parser.add_argument("--bitrate", default="320k",
                        help="Output bitrate (default: 320k)")
    args = parser.parse_args()

    if AudioSegment is None:
        print("ERROR: pydub is required to run the mixer "
              "(pip install pydub; ffmpeg must be in PATH).", file=sys.stderr)
        sys.exit(1)

    source_dir = Path(args.folder).resolve()
    if not source_dir.is_dir():
        print(f"ERROR: --folder '{source_dir}' does not exist or is not a directory.",
              file=sys.stderr)
        sys.exit(1)

    bitrate = args.bitrate if args.bitrate.endswith("k") else args.bitrate + "k"
    base_output = (
        Path(args.output).resolve()
        if args.output
        else source_dir.parent / f"{source_dir.name}_mix.mp3"
    )
    cache_file = Path(args.cache).resolve() if args.cache else source_dir / "bpm_cache.json"

    cfg = MixConfig(
        source_dir=source_dir,
        output_file=base_output,
        cache_file=cache_file,
        bitrate=bitrate,
    )

    mp3_files = sorted(cfg.source_dir.glob("*.mp3"))
    if not mp3_files:
        print(f"No MP3 files found in {cfg.source_dir}")
        return

    print(f"Found {len(mp3_files)} MP3 files in {cfg.source_dir}")
    print(f"Cache: {cfg.cache_file}")
    cache = BPMCache(cfg.cache_file)

    # --- analyse each file; save cache once at the end (or on crash) ---
    print("\n=== Analysis (BPM + Key) ===")
    tracks = []
    try:
        for idx, path in enumerate(mp3_files):
            print(f"  [{idx+1}/{len(mp3_files)}] Analysing: {path.name}")
            bpm = detect_bpm(path, cache)
            camelot, conf = detect_key(path, cache)
            seg = AudioSegment.from_mp3(str(path))
            dur_s = len(seg) / 1000.0
            print(
                f"    BPM: {bpm:6.1f}  Key: {camelot:<4} (conf: {conf:.2f})  "
                f"Dur: {dur_s/60:.1f} min"
            )
            tracks.append({
                "path": path,
                "name": path.name,
                "bpm": bpm,
                "camelot": camelot,
                "duration_s": dur_s,
            })
    finally:
        cache.save()
        print(f"  Cache saved → {cfg.cache_file}")

    # --- harmonic sort ---
    tracks = sort_tracks_harmonic(tracks)

    total_dur = sum(t["duration_s"] for t in tracks)
    print(f"\nTotal library: {len(tracks)} tracks, {total_dur/3600:.1f}h")
    print(f"Splitting into {args.parts} part(s)\n")

    # --- split into N equal parts by track count ---
    n = args.parts
    chunk_size = math.ceil(len(tracks) / n)
    parts = [tracks[i * chunk_size: (i + 1) * chunk_size] for i in range(n)]
    parts = [p for p in parts if p]

    for part_idx, part_tracks in enumerate(parts):
        part_num = part_idx + 1
        part_dur = sum(t["duration_s"] for t in part_tracks)

        if n == 1:
            output_file = cfg.output_file
        else:
            stem = cfg.output_file.stem
            output_file = cfg.output_file.with_name(
                f"{stem}_part{part_num}{cfg.output_file.suffix}"
            )

        print(f"{'='*60}")
        print(f"=== PART {part_num}/{len(parts)}  |  {len(part_tracks)} tracks  |  {part_dur/3600:.1f}h ===")
        print(f"{'='*60}")

        mix, tracklist, seed = build_mix(part_tracks, output_file)

        print("\n=== Silence Gap Detection ===")
        gaps = detect_silence_gaps(mix, tracklist)
        print_silence_report(gaps)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nExporting → {output_file}  ({cfg.bitrate})")
        mix.export(str(output_file), format="mp3", bitrate=cfg.bitrate)

        file_size = os.path.getsize(str(output_file))
        print(f"File size: {file_size / 1024 / 1024:.1f} MB")
        if file_size < 1024:
            print("WARNING: File appears empty!")

        write_tracklist(tracklist, len(mix), seed, gaps, output_file)

        final_dur_s = len(mix) / 1000.0
        print(f"\n  Part {part_num} done: {final_dur_s/3600:.2f}h  |  {output_file}\n")

        del mix  # free RAM between parts

    print("=== ALL PARTS COMPLETE ===")


if __name__ == "__main__":
    main()
