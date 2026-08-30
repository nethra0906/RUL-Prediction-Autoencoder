import pandas as pd
import pytest

from src.data.loaders import load_test, load_test_rul, load_train
from src.data.schema import COLUMN_NAMES


def _write_synthetic_train_file(path, n_engines=2, cycles_per_engine=5, trailing_space=True):
    """Mimic a real C-MAPSS file: 26 space-separated columns, optional trailing delimiter."""
    lines = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, cycles_per_engine + 1):
            values = [engine_id, cycle] + [0.0] * 24  # 3 settings + 21 sensors
            line = " ".join(str(v) for v in values)
            if trailing_space:
                line += " "  # trailing delimiter -> extra empty field on read
            lines.append(line)
    path.write_text("\n".join(lines) + "\n")


def test_load_train_assigns_canonical_columns(tmp_path):
    fd_dir = tmp_path / "FD001"
    fd_dir.mkdir()
    _write_synthetic_train_file(fd_dir / "train_FD001.txt")

    df = load_train("FD001", raw_dir=tmp_path)

    assert list(df.columns) == COLUMN_NAMES
    assert len(df) == 10  # 2 engines x 5 cycles


def test_load_train_handles_trailing_delimiter(tmp_path):
    fd_dir = tmp_path / "FD001"
    fd_dir.mkdir()
    _write_synthetic_train_file(fd_dir / "train_FD001.txt", trailing_space=True)

    df = load_train("FD001", raw_dir=tmp_path)

    assert df.shape[1] == 26
    assert not df.isna().any().any()


def test_load_train_without_trailing_delimiter_also_works(tmp_path):
    fd_dir = tmp_path / "FD001"
    fd_dir.mkdir()
    _write_synthetic_train_file(fd_dir / "train_FD001.txt", trailing_space=False)

    df = load_train("FD001", raw_dir=tmp_path)

    assert df.shape[1] == 26


def test_load_train_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_train("FD001", raw_dir=tmp_path)


def test_load_test_uses_same_schema(tmp_path):
    fd_dir = tmp_path / "FD002"
    fd_dir.mkdir()
    _write_synthetic_train_file(fd_dir / "test_FD002.txt", n_engines=1, cycles_per_engine=3)

    df = load_test("FD002", raw_dir=tmp_path)

    assert list(df.columns) == COLUMN_NAMES
    assert len(df) == 3


def test_load_test_rul_indexed_from_one(tmp_path):
    fd_dir = tmp_path / "FD001"
    fd_dir.mkdir()
    (fd_dir / "RUL_FD001.txt").write_text("112\n98\n45\n")

    rul = load_test_rul("FD001", raw_dir=tmp_path)

    assert rul.name == "RUL"
    assert list(rul.index) == [1, 2, 3]
    assert rul.loc[1] == 112
    assert rul.loc[3] == 45


def test_load_test_rul_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_test_rul("FD001", raw_dir=tmp_path)
