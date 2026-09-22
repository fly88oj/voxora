"""``voxora-api`` console entry point."""

import argparse

import uvicorn

from ..settings import get_settings
from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Voxora REST API server")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--models-dir", default=None,
                        help="directory containing model weights (overrides VOXORA_MODELS_DIR)")
    args = parser.parse_args()

    settings = get_settings()
    if args.models_dir:
        settings = settings.model_copy(update={"models_dir": args.models_dir})
    uvicorn.run(create_app(settings), host=args.host or settings.host,
                port=args.port or settings.port)


if __name__ == "__main__":
    main()
