from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

PRIOR_URL = "https://zenodo.org/records/15641297/files/reinvent.prior?download=1"
EXPECTED_SHA256 = "UNVERIFIED"


def sha256_for(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="vendor/reinvent_priors/reinvent.prior",
        help="Destination path for the prior file.",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.check and not output_path.exists():
        urlretrieve(PRIOR_URL, output_path)

    if not output_path.exists():
        raise SystemExit(f"Prior file not found: {output_path}")

    digest = sha256_for(output_path)
    print(f"path={output_path}")
    print(f"sha256={digest}")
    if EXPECTED_SHA256 != "UNVERIFIED" and digest != EXPECTED_SHA256:
        raise SystemExit("Prior SHA-256 mismatch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
