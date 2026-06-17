from __future__ import annotations

import json

from molforge.generative._env_check import probe_reinvent_environment


def main() -> int:
    print(json.dumps(probe_reinvent_environment().to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
