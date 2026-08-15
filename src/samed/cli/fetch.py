"""Stage 0: obtain a dataset and prove it is the one the paper describes.

    python -m samed.cli.fetch --dataset chaos --root $WORK/sam-medical-revisited/data
    python -m samed.cli.fetch --all --dry-run
    python -m samed.cli.fetch --dataset drive --verify-only

Downloading is the easy half. The half that matters is what happens afterwards:
every dataset is checked against the counts, resolutions and label encodings its
source publication reports, and anything that fails is rejected rather than
quietly used (see :mod:`samed.data.verify`).

Whatever is fetched is appended to a provenance lockfile - resolved URL, file
name, SHA-256, timestamp - so the report can state exactly which bytes produced
which numbers, and a reader can obtain the same ones.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import yaml

from samed.data.verify import sha256_of, verify_dataset

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "sources.yaml"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _ssl_context():
    """An SSL context that works on hosts without a populated system trust store.

    Python builds from python.org ship no root certificates on macOS until the
    bundled ``Install Certificates.command`` has been run, so ``urlopen`` fails
    with CERTIFICATE_VERIFY_FAILED on a machine where ``curl`` succeeds. Using
    certifi's bundle when it is available makes the fetcher behave the same on a
    laptop and on the cluster. Verification is never disabled.
    """
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def urlopen(url: str, **kwargs):
    """``urllib.request.urlopen`` with a usable trust store and a clear failure."""
    import urllib.error

    try:
        return urllib.request.urlopen(url, context=_ssl_context(), **kwargs)
    except urllib.error.URLError as error:
        raise RuntimeError(f"could not reach {url}: {error.reason}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", help="key from configs/sources.yaml")
    parser.add_argument("--all", action="store_true", help="every automatable source")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="describe, download nothing")
    parser.add_argument("--verify-only", action="store_true", help="check what is already there")
    return parser


# --------------------------------------------------------------------------- #
# Download routes
# --------------------------------------------------------------------------- #


def download(url: str, destination: Path, *, expect_md5: str | None = None) -> Path:
    """Stream a URL to disk, atomically, and check the publisher's own checksum."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"    have {destination.name}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"    get {url}")
    with urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    if expect_md5:
        import hashlib

        digest = hashlib.md5()
        with partial.open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
        if digest.hexdigest() != expect_md5:
            partial.unlink()
            raise ValueError(
                f"checksum mismatch for {destination.name}: "
                f"publisher says {expect_md5}, download is {digest.hexdigest()}"
            )
        print("    checksum matches the publisher's")

    partial.replace(destination)
    return destination


def fetch_zenodo(spec: dict, target: Path, *, dry_run: bool) -> list[Path]:
    """Resolve a Zenodo record through its API rather than hard-coding names.

    The API returns the current file list with official MD5s, so filenames never
    go stale and the integrity check uses the publisher's own digest.
    """
    api = f"https://zenodo.org/api/records/{spec['record']}"
    with urlopen(api) as response:
        record = json.load(response)

    wanted = set(spec.get("files") or [])
    files = [f for f in record["files"] if not wanted or f["key"] in wanted]
    if wanted and len(files) != len(wanted):
        missing = wanted - {f["key"] for f in files}
        raise ValueError(f"record {spec['record']} has no file(s) named {sorted(missing)}")

    if dry_run:
        for f in files:
            print(f"    would get {f['key']} ({f['size'] / 1e9:.2f} GB)")
        return []

    downloaded = []
    for f in files:
        md5 = f.get("checksum", "")
        downloaded.append(download(
            f["links"]["self"], target / f["key"],
            expect_md5=md5.removeprefix("md5:") if md5.startswith("md5:") else None,
        ))
    return downloaded


def fetch_kaggle(spec: dict, target: Path, *, dry_run: bool) -> list[Path]:
    slug = spec["slug"]
    if dry_run:
        print(f"    would run: kaggle datasets download -d {slug}")
        return []
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "the kaggle CLI is not on PATH; install it and place kaggle.json in ~/.kaggle"
        )
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(target)],
        check=True,
    )
    return sorted(target.glob("*.zip"))


def extract(archives: list[Path], target: Path) -> None:
    for archive in archives:
        stamp = target / f".extracted-{archive.name}"
        if stamp.exists():
            continue
        print(f"    unpack {archive.name}")
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                tf.extractall(target, filter="data")
        else:
            print(f"    (not an archive, left as is: {archive.name})")
            continue
        stamp.touch()


# --------------------------------------------------------------------------- #
# Verification and provenance
# --------------------------------------------------------------------------- #


def find_images(root: Path) -> tuple[list[Path], list[Path]]:
    """Split the extracted tree into probable images and probable label maps."""
    everything = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    labelish = ("mask", "label", "ground", "gt", "segmentation", "annotation")
    labels = [p for p in everything if any(k in str(p).lower() for k in labelish)]
    images = [p for p in everything if p not in set(labels)]
    return images, labels


def record_provenance(root: Path, name: str, spec: dict, files: list[Path]) -> None:
    lockfile = root / "provenance.json"
    entries = json.loads(lockfile.read_text()) if lockfile.exists() else {}
    entries[name] = {
        "type": spec.get("type"),
        "source": spec.get("url") or spec.get("slug") or spec.get("record"),
        "licence": spec.get("licence"),
        "retrieved": time.strftime("%Y-%m-%d"),
        "files": [{"name": f.name, "sha256": sha256_of(f)} for f in files],
    }
    lockfile.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    print(f"    provenance recorded in {lockfile}")


def handle(name: str, spec: dict, root: Path, *, dry_run: bool, verify_only: bool) -> bool:
    print(f"\n{name} [{spec.get('type')}]")
    target = root / name

    if licence_note := spec.get("licence_note"):
        print(f"    licence: {spec.get('licence')} - {' '.join(licence_note.split())}")

    archives: list[Path] = []
    if not verify_only:
        kind = spec.get("type")
        if kind == "manual":
            print(f"    manual download required: {spec['url']}")
            print(f"    unpack it into {target} and rerun with --verify-only")
            return True
        if kind == "zenodo":
            archives = fetch_zenodo(spec, target, dry_run=dry_run)
        elif kind == "kaggle":
            archives = fetch_kaggle(spec, target, dry_run=dry_run)
        else:
            raise ValueError(f"unknown source type {kind!r} for {name}")
        if dry_run:
            return True
        extract(archives, target)

    if not target.exists():
        print("    nothing on disk yet")
        return True

    images, labels = find_images(target)
    report = verify_dataset(
        name,
        image_paths=images,
        label_paths=labels,
        expect_count=spec.get("expect_count"),
        expect_resolutions=spec.get("expect_resolutions"),
        expect_label_values=spec.get("expect_label_values"),
    )
    print(report.render())

    if report.ok and archives:
        record_provenance(root, name, spec, archives)
    return report.ok


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = yaml.safe_load(args.config.read_text())["sources"]

    if args.all:
        names = [n for n, s in sources.items() if s.get("type") != "manual"]
    elif args.dataset:
        if args.dataset not in sources:
            print(f"unknown dataset {args.dataset!r}; known: {', '.join(sorted(sources))}")
            return 2
        names = [args.dataset]
    else:
        build_parser().print_help()
        return 2

    args.root.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in names:
        try:
            ok = handle(name, sources[name], args.root,
                        dry_run=args.dry_run, verify_only=args.verify_only)
        except (RuntimeError, ValueError, OSError) as error:
            # One unreachable host or one bad checksum must not abandon the
            # other datasets; the run reports what failed and exits non-zero.
            print(f"    ERROR {error}")
            ok = False
        if not ok:
            failed.append(name)

    if failed:
        print(f"\nrejected: {', '.join(failed)}")
        return 1
    print("\nall requested datasets pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
