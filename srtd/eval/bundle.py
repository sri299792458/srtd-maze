from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_file(tar: tarfile.TarFile, path: Path, arcname: Path, manifest: dict[str, str]) -> None:
    if not path.exists() or not path.is_file():
        return
    tar.add(path, arcname=str(arcname))
    manifest[str(arcname)] = _sha256(path)


def _add_tree(tar: tarfile.TarFile, root: Path, arcroot: Path, manifest: dict[str, str]) -> None:
    if not root.exists():
        return
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        _add_file(tar, path, arcroot / path.relative_to(root), manifest)


def create_repro_bundle(
    out: Path,
    dataset: Path | None,
    run_dirs: list[Path],
    report_dir: Path | None,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    with tarfile.open(out, "w:gz") as tar:
        if dataset is not None:
            _add_file(tar, dataset, Path("data") / dataset.name, manifest)
        for run_dir in run_dirs:
            arcroot = Path("runs") / run_dir.name
            for name in ["config.json", "checkpoint_last.pt", "losses.txt"]:
                _add_file(tar, run_dir / name, arcroot / name, manifest)
            for checkpoint in sorted(run_dir.glob("checkpoint_step_*.pt")):
                _add_file(tar, checkpoint, arcroot / checkpoint.name, manifest)
            _add_tree(tar, run_dir / "annotations", arcroot / "annotations", manifest)
            _add_tree(tar, run_dir / "figures", arcroot / "figures", manifest)
            _add_tree(tar, run_dir / "spectral", arcroot / "spectral", manifest)
        if report_dir is not None:
            _add_tree(tar, report_dir, Path("runs") / report_dir.name, manifest)
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo("SHA256SUMS.json")
        info.size = len(manifest_bytes)
        tar.addfile(info, fileobj=io.BytesIO(manifest_bytes))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args()
    out = create_repro_bundle(
        Path(args.out),
        Path(args.dataset) if args.dataset else None,
        [Path(p) for p in args.runs],
        Path(args.report_dir) if args.report_dir else None,
    )
    print(out)


if __name__ == "__main__":
    main()
