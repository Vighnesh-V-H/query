import json
import zipfile
from pathlib import Path

from query import cli
from query import source


def _write_archive(path: Path) -> None:
    csv_data = (
        "tweet_id,author_id,inbound,text\n"
        "1,customer,true,Where is my receipt?\n"
        "2,brand,false,We sent it by email.\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("twcs/twcs.csv", csv_data)


def test_read_source_archive_extracts_and_reports_csv(tmp_path):
    archive_path = tmp_path / "archive.zip"
    extract_dir = tmp_path / "raw"
    _write_archive(archive_path)

    report = source.read_source_archive(archive_path, extract_dir)

    assert report.extracted is True
    assert report.total_rows == 2
    assert report.files[0].path == "twcs/twcs.csv"
    assert report.files[0].columns == ("tweet_id", "author_id", "inbound", "text")
    assert report.files[0].rows == 2
    assert (extract_dir / "twcs" / "twcs.csv").exists()


def test_read_source_archive_reuses_valid_cache(tmp_path, monkeypatch):
    archive_path = tmp_path / "archive.zip"
    extract_dir = tmp_path / "raw"
    _write_archive(archive_path)
    source.read_source_archive(archive_path, extract_dir)

    def fail_extract(*args, **kwargs):
        raise AssertionError("archive was extracted more than once")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", fail_extract)

    report = source.read_source_archive(archive_path, extract_dir)

    assert report.extracted is False
    assert report.total_rows == 2


def test_read_source_archive_rejects_extracting_over_archive(tmp_path):
    archive_path = tmp_path / "archive.zip"
    _write_archive(archive_path)

    try:
        source.read_source_archive(archive_path, tmp_path)
    except source.SourceArchiveError as exc:
        assert "cannot contain" in str(exc)
    else:
        raise AssertionError("expected overlapping extraction directory to be rejected")


def test_read_source_archive_does_not_replace_unmanaged_directory(tmp_path):
    archive_path = tmp_path / "archive.zip"
    extract_dir = tmp_path / "raw"
    _write_archive(archive_path)
    extract_dir.mkdir()
    marker = extract_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    try:
        source.read_source_archive(archive_path, extract_dir)
    except source.SourceArchiveError as exc:
        assert "unmanaged" in str(exc)
    else:
        raise AssertionError("expected unmanaged extraction directory to be rejected")
    assert marker.read_text(encoding="utf-8") == "keep"


def test_invalid_manifest_is_treated_as_cache_miss(tmp_path):
    archive_path = tmp_path / "archive.zip"
    extract_dir = tmp_path / "raw"
    _write_archive(archive_path)
    extract_dir.mkdir()
    (extract_dir / source.MANIFEST_NAME).write_text("null", encoding="utf-8")

    report = source.read_source_archive(archive_path, extract_dir)

    assert report.extracted is True
    assert report.total_rows == 2


def test_modified_cached_csv_is_reextracted(tmp_path):
    archive_path = tmp_path / "archive.zip"
    extract_dir = tmp_path / "raw"
    _write_archive(archive_path)
    source.read_source_archive(archive_path, extract_dir)
    (extract_dir / "twcs" / "twcs.csv").write_text(
        "tweet_id,author_id,inbound,text\n", encoding="utf-8"
    )

    report = source.read_source_archive(archive_path, extract_dir)

    assert report.extracted is True
    assert report.total_rows == 2


def test_malformed_csv_row_is_rejected(tmp_path):
    archive_path = tmp_path / "archive.zip"
    extract_dir = tmp_path / "raw"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data.csv", "id,text\n1\n")

    try:
        source.read_source_archive(archive_path, extract_dir)
    except source.SourceArchiveError as exc:
        assert "expected 2" in str(exc)
    else:
        raise AssertionError("expected malformed CSV row to be rejected")


def test_manifest_records_schema_and_counts(tmp_path):
    archive_path = tmp_path / "archive.zip"
    extract_dir = tmp_path / "raw"
    _write_archive(archive_path)

    source.read_source_archive(archive_path, extract_dir)

    manifest = json.loads((extract_dir / source.MANIFEST_NAME).read_text())
    assert manifest["files"][0]["rows"] == 2
    assert manifest["files"][0]["columns"] == ["tweet_id", "author_id", "inbound", "text"]


def test_read_source_command_prints_report(tmp_path, capsys):
    archive_path = tmp_path / "archive.zip"
    extract_dir = tmp_path / "raw"
    _write_archive(archive_path)

    assert cli.main(
        ["read-source", "--archive", str(archive_path), "--extract-dir", str(extract_dir)]
    ) == 0

    output = capsys.readouterr().out
    assert "schema: twcs/twcs.csv -> tweet_id, author_id, inbound, text" in output
    assert "rows: twcs/twcs.csv -> 2" in output
    assert "total rows across CSV files: 2" in output
