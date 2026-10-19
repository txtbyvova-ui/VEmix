"""Generate a small SYNTHETIC demo library for smoke tests (no copyrighted music).

    python examples/make_demo_library.py ./demo_music [--seconds 70]

Creates fictional "songs" (chord progressions in known keys, kick/hat at known tempi,
different mastering loudness), tagged with artist/title/genre/language, plus edge
cases: a Cyrillic-titled track without a language tag, a track tagged ``ru``, a
track whose loudness target conflicts with its true peak, and a corrupt file.
Also writes ``demo_metadata.csv`` (curator overrides) next to the folder.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from veg_mixer.ffmpeg_tools import find_ffmpeg  # noqa: E402

SR = 44100
NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

# artist, title, genre, language, bpm, key root, mode, energy (0..1), peak dBFS, mastering drive
TRACKS = [
    ("Demo Aurora", "Morning Glow", "pop", "en", 92, "C", "major", 0.25, -9.0, 1.5),
    ("Demo Aurora", "Paper Boats", "pop", "en", 100, "G", "major", 0.35, -3.0, 2.0),
    ("Demo Harbor", "City Walk", "pop", "en", 110, "A", "minor", 0.50, -1.0, 2.5),
    ("Demo Harbor", "Window Seat", "pop", "fr", 104, "F", "major", 0.45, -2.0, 2.0),
    ("Demo Coral", "Weekend Drive", "dance-pop", "en", 122, "D", "minor", 0.75, -0.1, 4.0),
    ("Demo Coral", "Neon Avenue", "dance-pop", "en", 126, "A", "minor", 0.85, -0.1, 5.0),
    ("Demo Delta", "Lemon Tree Cafe", "pop", "es", 98, "E", "major", 0.40, -4.0, 2.0),
    ("Demo Echo", "Slow Sunday", "soft-pop", "it", 86, "A#", "major", 0.20, -12.0, 1.0),
    ("Demo Field", "Drone Study No. 3", "experimental", None, 70, "D", "minor", 0.10, -6.0, 0.0),
    ("Демо Тест", "Утренний трек", "pop", None, 108, "G", "major", 0.40, -2.0, 2.0),
    ("Demo Mira", "Night City", "pop", "ru", 115, "B", "minor", 0.55, -1.0, 2.5),
]


def chord_progression(root: int, mode: str) -> list[list[int]]:
    """I-V-vi-IV (major) or i-VI-III-VII (minor) as semitone offsets from C."""
    if mode == "major":
        degrees = [(0, 4, 7), (7, 11, 14), (9, 12, 16), (5, 9, 12)]
    else:
        degrees = [(0, 3, 7), (8, 12, 15), (3, 7, 10), (10, 14, 17)]
    return [[root + d for d in chord] for chord in degrees]


def synth_track(seconds: float, bpm: float, root: str, mode: str, energy: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    beat = 60.0 / bpm
    bar = 4 * beat
    out = np.zeros(n)
    # harmonic bed: one chord per bar, three partials per note
    chords = chord_progression(NOTE[root], mode)
    for k in range(int(np.ceil(seconds / bar))):
        s, e = int(k * bar * SR), min(n, int((k + 1) * bar * SR))
        if s >= n:
            break
        tt = t[s:e] - t[s]
        env = np.minimum(1.0, tt / 0.05) * np.exp(-tt / (bar * 1.5))
        for semi in chords[k % 4]:
            f = 220.0 * 2 ** ((semi - 9) / 12)
            for h, amp in ((1, 1.0), (2, 0.4), (3, 0.2)):
                out[s:e] += 0.12 * amp * np.sin(2 * np.pi * f * h * tt) * env
    # rhythm: kick every beat (louder for higher energy), hats on off-beats when energetic
    phase = t % beat
    kick_f = 50.0 + 100.0 * np.exp(-phase / 0.02)          # pitch-swept (un-keyed) kick
    out += (0.3 + 0.7 * energy) * np.sin(2 * np.pi * np.cumsum(kick_f) / SR) * np.exp(-phase / 0.08)
    if energy >= 0.4:
        hat_phase = (t + beat / 2) % beat
        out += energy * 0.25 * rng.standard_normal(n) * np.exp(-hat_phase / 0.015)
    # intro / outro fades like real songs
    fade = np.minimum(1.0, np.minimum(t / 4.0, (seconds - t) / 6.0))
    out *= np.clip(fade, 0, 1)
    stereo = np.stack([out, np.roll(out, 64)], 1)
    return stereo / (np.max(np.abs(stereo)) + 1e-12)


def master(x: np.ndarray, peak_dbfs: float, drive: float) -> np.ndarray:
    if drive > 0:
        x = np.tanh(drive * x) / np.tanh(drive)
    return (x * 10 ** (peak_dbfs / 20)).astype(np.float32)


def encode(ffmpeg: str, wav: Path, mp3: Path, tags: dict[str, str]) -> None:
    args = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", str(wav)]
    for k, v in tags.items():
        args += ["-metadata", f"{k}={v}"]
    args += ["-c:a", "libmp3lame", "-b:a", "192k", "-id3v2_version", "3", str(mp3)]
    subprocess.run(args, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--seconds", type=float, default=70.0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tools = find_ffmpeg()
    folder = Path(args.folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, (artist, title, genre, lang, bpm, root, mode, energy, peak, drive) in enumerate(TRACKS):
            data = master(synth_track(args.seconds, bpm, root, mode, energy, seed=i), peak, drive)
            wav = Path(tmp) / f"{i}.wav"
            sf.write(str(wav), data, SR, subtype="FLOAT")
            tags = {"artist": artist, "title": title, "genre": genre}
            if lang:
                tags["language"] = lang
            encode(tools.ffmpeg, wav, folder / f"{artist} - {title}.mp3", tags)
            print(f"  {artist} - {title}.mp3  ({bpm} BPM, {root} {mode}, peak {peak} dBFS)")
        # loudness/peak conflict: quiet bed + full-scale clicks
        x = synth_track(args.seconds, 90, "E", "minor", 0.2, seed=99) * 10 ** (-38 / 20)
        for k in range(SR, len(x), 2 * SR):
            x[k:k + 20] = 0.97
        wav = Path(tmp) / "spiky.wav"
        sf.write(str(wav), x.astype(np.float32), SR, subtype="FLOAT")
        encode(tools.ffmpeg, wav, folder / "Demo Quiet - Click Garden.mp3",
               {"artist": "Demo Quiet", "title": "Click Garden", "genre": "pop", "language": "en"})
        print("  Demo Quiet - Click Garden.mp3  (loudness/true-peak conflict)")
    (folder / "Demo Broken - Corrupt File.mp3").write_bytes(b"ID3" + bytes(range(256)) * 64)
    print("  Demo Broken - Corrupt File.mp3  (corrupt on purpose)")

    meta = folder.parent / "demo_metadata.csv"
    meta.write_text(
        "rel_path,mood,energy,vibe_tags,approved,notes\n"
        "Demo Aurora - Morning Glow.mp3,calm,0.25,melodic;acoustic,true,curated\n"
        "Demo Aurora - Paper Boats.mp3,warm,0.35,melodic;recognizable,true,\n"
        "Demo Harbor - City Walk.mp3,uplifting,0.5,recognizable,true,\n"
        "Demo Coral - Weekend Drive.mp3,energetic,0.75,groove;dance,true,\n"
        "Demo Coral - Neon Avenue.mp3,energetic,0.85,groove;dance,true,\n"
        "Demo Field - Drone Study No. 3.mp3,dark,0.1,experimental;instrumental,false,obscure\n",
        encoding="utf-8")
    print(f"Curator metadata example: {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
