"""Reproducible loudness measurements on synthetic signals (no copyrighted music).

    python examples/measure_loudness_demo.py [--out ./loudness_demo]

For each synthetic case: measure the source, normalize it to MP3 with the default
configuration, and print the verifier's numbers next to an INDEPENDENT re-measurement of
the published MP3 with FFmpeg's ebur128 filter alone. Prints a Markdown table.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from conftest import SR, encode_mp3, music_like, spiky  # noqa: E402
from veg_mixer.ffmpeg_tools import find_ffmpeg  # noqa: E402
from veg_mixer.loudness import LoudnessNormalizer  # noqa: E402


def ebur128(tools, path: Path) -> tuple[float, float]:
    p = subprocess.run([tools.ffmpeg, "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, encoding="utf-8", errors="replace")
    i = float(re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", p.stderr)[-1])
    tp = float(re.findall(r"Peak:\s*(-?[\d.]+)\s*dBFS", p.stderr)[-1])
    return i, tp


def cases() -> dict[str, np.ndarray]:
    t = np.arange(15 * SR) / SR
    tone = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return {
        "quiet master (peak -20 dBFS)": music_like(20, peak_dbfs=-20, seed=1),
        "loud master (peak -0.5 dBFS)": music_like(20, peak_dbfs=-0.5, seed=2),
        "hot, heavily clipped-style master": music_like(20, peak_dbfs=-0.1, seed=5, drive=6),
        "unmastered, high crest": music_like(20, peak_dbfs=-6, seed=4, drive=0),
        "mono source": music_like(15, peak_dbfs=-12, seed=6, channels=1),
        "steady tone (LRA 0)": np.stack([tone, tone], 1),
        "loudness/true-peak conflict": spiky(20),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="keep outputs here (default: temporary folder)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tools = find_ffmpeg()
    norm = LoudnessNormalizer(tools)
    work = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="veg_measure_"))
    print(f"FFmpeg: {tools.version}\nTargets: {norm.cfg.target_lufs} LUFS ±{norm.cfg.lufs_tolerance}, "
          f"TP <= {norm.cfg.target_true_peak_dbtp} dBTP (+{norm.cfg.true_peak_tolerance_db})\n")
    print("| Case | Source LUFS / TP | Mode | Status | Output LUFS / TP (verifier) | Output LUFS / TP (ebur128 only) | Attempts |")
    print("|---|---|---|---|---|---|---|")
    for i, (name, data) in enumerate(cases().items()):
        wav = work / "src" / f"{i}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav), data, SR, subtype="FLOAT")
        src = encode_mp3(tools, wav, work / "src" / f"{i}.mp3")
        r = norm.normalize_to_mp3(src, work / "out" / f"{i}.mp3")
        m_in = r.input_measurement
        if r.output is not None:
            ind_i, ind_tp = ebur128(tools, r.output)
            ind = f"{ind_i:.1f} / {ind_tp:+.1f}"
            ver = f"{r.measured_lufs:.2f} / {r.true_peak_dbtp:+.2f}"
        else:
            ind = ver = "not published"
        print(f"| {name} | {m_in.integrated_lufs:.1f} / {m_in.true_peak_dbtp:+.1f} | "
              f"{r.plan.mode.value if r.plan else '-'} | {r.status.value} | {ver} | {ind} | {r.encode_attempts} |")
    print(f"\nOutputs: {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
