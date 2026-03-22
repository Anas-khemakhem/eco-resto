import sqlite3
import os

DB_FILE = 'database/ecoresto.db'

def get_db_connection():
    os.makedirs('database', exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row # Returns dict-like objects
    return conn

def init_db():
    conn = get_db_connection()
    with open('database/schema.sql', 'r') as f:
        conn.executescript(f.read())
    
    # Set default status if empty
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('food_status', 'available')")
    conn.commit()
    conn.close()