"""Canonical C-MAPSS column schema.

See AI_CONTEXT.md Section 4. Column names must be assigned explicitly —
never rely on unnamed / positional dataframe columns.
"""

SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

COLUMN_NAMES = ["unit_id", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]
