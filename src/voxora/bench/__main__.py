"""``voxora`` console entry point — benchmark CLI.

Examples:

    voxora list
    voxora run --engine sensevoice --audio-dir data/fixtures -o sense.json
    voxora run --engine piper --text "Hello world." --language en -o piper.json
    voxora run --engine zipformer --audio-dir data/fixtures --repeat 5 -o zf.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..engines import EngineError, engine_classes
from .runner import run_asr, run_tts, write_result


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voxora",
                                     description="voxora benchmark CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list registered engines")

    p_run = sub.add_parser("run", help="run one engine over a workload")
    p_run.add_argument("--engine", required=True)
    p_run.add_argument("--models-dir", default="models")
    p_run.add_argument("--threads", type=int, default=8)
    p_run.add_argument("--audio-dir", default=None, help="directory of wav files (ASR)")
    p_run.add_argument("--text", action="append", default=None,
                       help="input text (repeatable; TTS)")
    p_run.add_argument("--language", default=None)
    p_run.add_argument("--voice", default=None)
    p_run.add_argument("--wav-dir", default=None, help="write synthesized wavs here")
    p_run.add_argument("--repeat", type=int, default=1,
                       help="run the workload N times and report mean/std/cv (>=2 "
                            "exposes cold-vs-warm spread; pass 1 is cold)")
    p_run.add_argument("-o", "--output", required=True, help="output JSON path")

    args = parser.parse_args(argv)

    if args.cmd == "list":
        for name, cls in sorted(engine_classes().items()):
            print(f"{name:20s} {cls.kind.value:3s}  {cls.description}")
        return 0

    if args.engine not in engine_classes():
        print(f"unknown engine '{args.engine}' (see voxora list)", file=sys.stderr)
        return 2
    kind = engine_classes()[args.engine].kind

    try:
        if kind.value == "asr":
            if not args.audio_dir:
                print("--audio-dir is required for ASR engines", file=sys.stderr)
                return 2
            files = sorted(Path(args.audio_dir).glob("*.wav"))
            if not files:
                print(f"no wav files under {args.audio_dir}", file=sys.stderr)
                return 2
            result = run_asr(args.engine, files, models_dir=args.models_dir,
                             num_threads=args.threads, repeat=args.repeat)
        else:
            texts = args.text or []
            if not texts:
                print("--text is required for TTS engines", file=sys.stderr)
                return 2
            result = run_tts(args.engine, texts, models_dir=args.models_dir,
                             num_threads=args.threads, language=args.language,
                             voice=args.voice,
                             wav_dir=Path(args.wav_dir) if args.wav_dir else None,
                             repeat=args.repeat)
    except (EngineError, FileNotFoundError) as e:
        print(f"engine error: {e}", file=sys.stderr)
        return 3

    write_result(Path(args.output), result)
    agg = result["aggregate"]
    stats = agg.get("stats", {})
    print(f"{result['engine']}: rtf={agg['rtf']}"
          + (f" (passes={agg['passes']}, std={stats.get('std')}, cv={stats.get('cv')})"
             if len(agg.get("passes", [])) > 1 else "")
          + f" -> {args.output}")
    return 0


def main() -> None:
    raise SystemExit(_main())


if __name__ == "__main__":
    main()
