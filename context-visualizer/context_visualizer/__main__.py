"""Entry point for ``python -m context_visualizer``."""

import sys

_MISSING = []
for _mod in ("flask", "yaml", "msgpack"):
    try:
        __import__(_mod)
    except ImportError:
        _MISSING.append({"flask": "flask", "yaml": "pyyaml", "msgpack": "msgpack"}[_mod])

if _MISSING:
    print(
        "ERROR: missing Python dependencies: " + ", ".join(_MISSING) + "\n"
        "\n"
        "Install them with one of:\n"
        "  pip install " + " ".join(_MISSING) + "\n"
        "  pip install iowarp-core[visualizer]\n"
        "  conda install " + " ".join(_MISSING),
        file=sys.stderr,
    )
    sys.exit(1)

import argparse

from .app import create_app


def main():
    parser = argparse.ArgumentParser(description="Chimaera runtime visualizer")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Listen port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
