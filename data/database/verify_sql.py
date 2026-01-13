import sqlite3
import pandas as pd

conn = sqlite3.connect("data/database/nfl_pbp.db")

query = """
SELECT
    down,
    COUNT(*) AS plays,
    AVG(epa) AS avg_epa
FROM pbp_field_position
GROUP BY down
ORDER BY down;
"""

df = pd.read_sql(query, conn)
conn.close()

print(df)
