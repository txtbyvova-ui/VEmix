"""
automix.py — VEgroove Automix 2.0 entry point (backward compatible).
----------------------------------------------------------------------
The implementation lives in the ``veg_mixer`` package; the original prototype is
kept unchanged in ``legacy/automix_v1.py``.

    python automix.py analyze   --folder ./music
    python automix.py normalize --folder ./music --output ./normalized --target-lufs -16 --true-peak -1.5
    python automix.py playlist  --folder ./music --profile ./profiles/client.yaml --output ./output
    python automix.py mix       --folder ./music --profile ./profiles/client.yaml --output ./output/mix.mp3
    python automix.py qc        --folder ./output

    # automix v1 syntax still works (runs `mix`):
    python automix.py --folder ./music --output ./mix.mp3 --parts 2 --bitrate 320k

Dependencies: see requirements.txt; FFmpeg + ffprobe on PATH (or VEG_FFMPEG / VEG_FFPROBE).
"""

import sys

from veg_mixer.cli import main

if __name__ == "__main__":
    sys.exit(main())
