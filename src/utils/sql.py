import sqlite3
import pandas as pd

def read_sql(query, db_path):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(query, conn)

def write_sql(df, table_name, db_path, if_exists="replace"):
    with sqlite3.connect(db_path) as conn:
        df.to_sql(
            table_name,
            conn,
            if_exists=if_exists,
            index=False
        )
#SQL logic centralized
#No repeated connection code
#Easy to swap databases later