
from __future__ import annotations

# Keep the public CLI entrypoint for backward compatibility.
# This forwards to src/cli.py:main()

from .cli import main

if __name__ == "__main__":
    main()
