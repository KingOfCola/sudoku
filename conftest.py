"""Make the paths listed in ``.env``'s ``PYTHONPATH`` importable during tests.

pytest does not read ``.env`` on its own, so we parse it here and prepend the
referenced directories (e.g. ``./src``) to ``sys.path``. This keeps ``.env`` as
the single source of truth shared with the editor / IDE.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_ENV_FILE = os.path.join(_ROOT, ".env")

if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            if key != "PYTHONPATH":
                continue
            for entry in value.split(os.pathsep):
                entry = entry.strip()
                if not entry:
                    continue
                abs_entry = entry if os.path.isabs(entry) else os.path.join(_ROOT, entry)
                abs_entry = os.path.normpath(abs_entry)
                if abs_entry not in sys.path:
                    sys.path.insert(0, abs_entry)
