-- Core App Settings
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);

-- Menu & Daily Context
CREATE TABLE IF NOT EXISTS menu (id INTEGER PRIMARY KEY, date TEXT, main_dish TEXT, protein TEXT, side_dish TEXT, fruit TEXT, yogurt TEXT);
CREATE TABLE IF NOT EXISTS calendar (id INTEGER PRIMARY KEY, event_date TEXT, event_type TEXT, weather TEXT);

-- Student Interactions
CREATE TABLE IF NOT EXISTS rsvp (id INTEGER PRIMARY KEY, student_id TEXT, date TEXT, will_attend BOOLEAN);
CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, date TEXT, dish_name TEXT, review_text TEXT, sentiment_score REAL);

-- AI Vision Logs
CREATE TABLE IF NOT EXISTS tray_scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    date TEXT, 
    detected_item TEXT, 
    waste_pct INTEGER, 
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);