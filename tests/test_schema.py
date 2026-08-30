from src.data.schema import COLUMN_NAMES, SENSOR_COLUMNS, SETTING_COLUMNS


def test_column_count_is_26():
    assert len(COLUMN_NAMES) == 26


def test_column_order():
    assert COLUMN_NAMES[0] == "unit_id"
    assert COLUMN_NAMES[1] == "cycle"
    assert COLUMN_NAMES[2:5] == SETTING_COLUMNS
    assert COLUMN_NAMES[5:] == SENSOR_COLUMNS


def test_sensor_columns_count():
    assert len(SENSOR_COLUMNS) == 21


def test_setting_columns_count():
    assert len(SETTING_COLUMNS) == 3
