import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARCHIVE_PATH = REPO_ROOT / "data" / "archive.zip"
DEFAULT_EXTRACT_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_NAME = ".query-source-manifest.json"
MAX_ARCHIVE_MEMBERS = 10_000
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3


class SourceArchiveError(Exception):
    """Raised when the source archive cannot be read safely."""


@dataclass(frozen=True)
class CsvReport:
    path: str
    columns: tuple[str, ...]
    rows: int


@dataclass(frozen=True)
class SourceReport:
    archive_path: Path
    extraction_dir: Path
    files: tuple[CsvReport, ...]
    extracted: bool

    @property
    def total_rows(self) -> int:
        return sum(csv_file.rows for csv_file in self.files)


def read_source_archive(
    archive_path: Path = DEFAULT_ARCHIVE_PATH,
    extract_dir: Path = DEFAULT_EXTRACT_DIR,
) -> SourceReport:
    """Cache a matching archive extraction and report CSV schemas and row counts."""
    archive_path = Path(archive_path).resolve()
    extract_dir = Path(extract_dir).resolve()
    if not archive_path.is_file():
        raise SourceArchiveError(f"source archive does not exist: {archive_path}")
    if extract_dir == archive_path or extract_dir in archive_path.parents:
        raise SourceArchiveError("extraction directory cannot contain the source archive")

    temporary_dir = None
    previous_dir = None
    try:
        signature = _file_signature(archive_path)
        cached_files = _read_cached_report(extract_dir, signature)
        if cached_files is not None:
            return SourceReport(archive_path, extract_dir, cached_files, extracted=False)

        extract_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{extract_dir.name}-", dir=extract_dir.parent)
        )
        with zipfile.ZipFile(archive_path) as archive:
            _validate_members(archive, temporary_dir)
            archive.extractall(temporary_dir)
        files = _scan_csv_files(temporary_dir)
        if not files:
            raise SourceArchiveError(f"source archive contains no CSV files: {archive_path}")
        _write_manifest(temporary_dir, signature, files)
        if extract_dir.exists():
            if not extract_dir.is_dir():
                raise SourceArchiveError(f"extraction path is not a directory: {extract_dir}")
            if not (extract_dir / MANIFEST_NAME).is_file() and any(extract_dir.iterdir()):
                raise SourceArchiveError(
                    f"refusing to replace unmanaged extraction directory: {extract_dir}"
                )
            previous_dir = extract_dir.with_name(extract_dir.name + ".previous")
            if previous_dir.exists():
                shutil.rmtree(previous_dir, ignore_errors=True)
            extract_dir.rename(previous_dir)
        os.replace(temporary_dir, extract_dir)
        if previous_dir is not None:
            shutil.rmtree(previous_dir, ignore_errors=True)
    except (
        OSError,
        ValueError,
        RuntimeError,
        NotImplementedError,
        UnicodeError,
        zipfile.BadZipFile,
        csv.Error,
    ) as exc:
        raise SourceArchiveError(f"could not read source archive {archive_path}: {exc}") from exc
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)

    return SourceReport(archive_path, extract_dir, files, extracted=True)


def _file_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": stat.st_size, "sha256": digest.hexdigest()}


def _read_cached_report(
    extract_dir: Path, archive_signature: dict[str, int | str]
) -> tuple[CsvReport, ...] | None:
    manifest_path = extract_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
            return None
        if not manifest["files"]:
            return None
        if manifest.get("archive") != archive_signature:
            return None
        files = []
        for item in manifest["files"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not isinstance(item.get("columns"), list)
                or not item["columns"]
                or not all(isinstance(column, str) for column in item["columns"])
                or type(item.get("rows")) is not int
                or item["rows"] < 0
                or not isinstance(item.get("signature"), dict)
            ):
                return None
            relative_path = Path(item["path"])
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative_path.suffix.lower() != ".csv"
            ):
                return None
            file_path = (extract_dir / relative_path).resolve()
            try:
                file_path.relative_to(extract_dir)
            except ValueError:
                return None
            files.append(CsvReport(relative_path.as_posix(), tuple(item["columns"]), item["rows"]))
        files = tuple(files)
        for csv_file, item in zip(files, manifest["files"]):
            path = extract_dir / csv_file.path
            if not path.is_file() or _file_signature(path) != item["signature"]:
                return None
        return files
    except (OSError, TypeError, ValueError):
        return None


def _write_manifest(
    extract_dir: Path, archive_signature: dict[str, int | str], files: tuple[CsvReport, ...]
) -> None:
    manifest = {
        "archive": archive_signature,
        "files": [
            {
                "path": csv_file.path,
                "columns": list(csv_file.columns),
                "rows": csv_file.rows,
                "signature": _file_signature(extract_dir / csv_file.path),
            }
            for csv_file in files
        ],
    }
    (extract_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _scan_csv_files(extract_dir: Path) -> tuple[CsvReport, ...]:
    reports = []
    for path in sorted(extract_dir.rglob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                columns = tuple(next(reader))
            except StopIteration as exc:
                raise SourceArchiveError(f"CSV file has no header: {path}") from exc
            if not columns or any(not column for column in columns):
                raise SourceArchiveError(f"CSV file has an invalid header: {path}")
            rows = 0
            for row in reader:
                if not row:
                    continue
                rows += 1
                if len(row) != len(columns):
                    raise SourceArchiveError(
                        f"CSV row {rows + 1} has {len(row)} fields, expected {len(columns)}: {path}"
                    )
        reports.append(CsvReport(path.relative_to(extract_dir).as_posix(), columns, rows))
    return tuple(reports)


def _validate_members(archive: zipfile.ZipFile, extract_dir: Path) -> None:
    root = extract_dir.resolve()
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise SourceArchiveError(f"archive contains too many members: {len(members)}")
    uncompressed_size = sum(member.file_size for member in members)
    if uncompressed_size > MAX_UNCOMPRESSED_BYTES:
        raise SourceArchiveError(
            f"archive is too large to extract safely: {uncompressed_size} bytes"
        )
    for member in members:
        target = (extract_dir / member.filename).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise SourceArchiveError(
                f"archive member escapes extraction directory: {member.filename}"
            ) from exc
