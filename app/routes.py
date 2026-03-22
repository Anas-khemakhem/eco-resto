import os 
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime
from .models import get_db_connection
# IMPORT OUR 3 NEW SUPER-AGENTS
from .services.waste_detection import vision_ai
from .services.forecasting import forecaster
from .services.nlp_agent import nlp_engine
from .services.ngo_agent import ngo_dispatcher

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/chef', methods=['GET', 'POST'])
def chef():
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    conn = get_db_connection()
    
    if request.method == 'POST':
        if 'main_dish' in request.form:
            conn.execute("INSERT INTO menu (date, main_dish, protein, side_dish, fruit, yogurt) VALUES (?, ?, ?, ?, ?, ?)", 
                      (today_str, request.form['main_dish'], request.form['protein'], request.form['side_dish'], request.form['fruit'], request.form['yogurt']))
        elif 'toggle_status' in request.form:
            status = conn.execute("SELECT value FROM settings WHERE key='food_status'").fetchone()['value']
            new_status = 'sold_out' if status == 'available' else 'available'
            conn.execute("UPDATE settings SET value=? WHERE key='food_status'", (new_status,))
        conn.commit()
        
    menu = conn.execute("SELECT * FROM menu WHERE date=? ORDER BY id DESC LIMIT 1", (today_str,)).fetchone()
    food_status = conn.execute("SELECT value FROM settings WHERE key='food_status'").fetchone()['value']
    exam_count = conn.execute("SELECT COUNT(*) as count FROM calendar WHERE event_type LIKE '%Exam%'").fetchone()['count']
    
    # --- LEVEL 1: DEEP FORECASTING AGENT ---
    is_exam = 1 if exam_count > 0 else 0
    # Provide the exact parameters the new HistGradientBoosting model expects
    predicted_qty = forecaster.predict_attendance(
        day=today.weekday(), 
        month=today.month, 
        is_exam=is_exam, 
        is_rain=0, 
        menu_score=7
    )

    # --- LEVEL 2: NLP SENTIMENT AGENT ---
    # The AI reads the database and gives the chef a live sentiment report
    nlp_alert = nlp_engine.generate_chef_report('database/ecoresto.db')
    
    # Combine the context for the chef UI
    context = f"HistGradientBoosting Active. {nlp_alert}"
    conn.close()

    return render_template('chef.html', menu=menu, predicted=predicted_qty, context=context, 
                           food_status=food_status, portions={}, ingredients={}, warnings=[])

@main_bp.route('/api/scan_tray', methods=['POST'])
def scan_tray():
    data = request.get_json()
    b64_image = data.get('image', '')
    
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    menu = conn.execute("SELECT * FROM menu WHERE date=? ORDER BY id DESC LIMIT 1", (today,)).fetchone()
    
    if not menu:
        return jsonify({"status": "error", "message": "No active menu to analyze today."})
    
    active_items = [menu['main_dish'], menu['protein'], menu['side_dish'], menu['fruit'], menu['yogurt']]
    active_items = [item for item in active_items if item != 'none']
    
    # 🔴 NOTE: We added nutrients to the return variables here
    detected_item, pct, reasoning, conf, nutrients = vision_ai.analyze_base64_image(b64_image, active_items)
    
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tray_scans (date, detected_item, waste_pct, confidence) VALUES (?, ?, ?, ?)", 
                 (today, detected_item, pct, conf))
    scan_id = cursor.lastrowid # Get the ID of this exact scan
    conn.commit()
    conn.close()
    
    return jsonify({
        "status": "success", 
        "scan_id": scan_id,       # Send ID back to frontend
        "detected_item": detected_item, 
        "percentage": pct, 
        "reasoning": reasoning,
        "confidence": conf,
        "nutrition": nutrients    # Send Nutrition data back
    })

@main_bp.route('/api/teach_ai', methods=['POST'])
def teach_ai():
    data = request.get_json()
    scan_id = data.get('scan_id')
    corrections = data.get('corrections', {})
    dominant_item = data.get('dominant_item', 'Clean Plate')
    total_pct = data.get('total_pct', 0)
    image_b64 = data.get('image', '')
    
    # 1. Format the database string (e.g. "Couscous: 71% | Escalope: 33%")
    if not corrections:
        corrected_str = "Clean Plate"
    else:
        corrected_str = " | ".join([f"{k.replace('_', ' ').title()}: {v}%" for k, v in corrections.items()])
    
    import os
    os.makedirs('training_data', exist_ok=True)
    
    if image_b64 and dominant_item != "Clean Plate":
        import base64
        # Save image for weekend retraining
        img_data = image_b64.split(",")[1] if "," in image_b64 else image_b64
        with open(f"training_data/correction_{scan_id}_{dominant_item}.jpg", "wb") as fh:
            fh.write(base64.b64decode(img_data))
            
        # 🔴 TRIGGER ZERO-SHOT COLOR MEMORY ON THE HIGHEST % ITEM
        from .services.waste_detection import vision_ai
        vision_ai.learn_new_signature(dominant_item, image_b64)

    conn = get_db_connection()
    conn.execute("CREATE TABLE IF NOT EXISTS ai_training_data (id INTEGER PRIMARY KEY, scan_id INTEGER, true_label TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("INSERT INTO ai_training_data (scan_id, true_label) VALUES (?, ?)", (scan_id, corrected_str))
    
    # Update the admin dashboard with the REAL data!
    conn.execute("UPDATE tray_scans SET detected_item = ?, waste_pct = ?, confidence = 1.0 WHERE id = ?", 
                 (f"[GROUND TRUTH] {corrected_str}", total_pct, scan_id))
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success"})

@main_bp.route('/student', methods=['GET', 'POST'])
def student():
    # Handle the RSVP and NLP Feedback logic here
    if request.method == 'POST':
        today = datetime.now().strftime("%Y-%m-%d")
        conn = get_db_connection()
        
        # Log RSVP
        if 'attendance' in request.form:
            intent = 1 if request.form['attendance'] == 'yes' else 0
            conn.execute("INSERT INTO rsvp (student_id, date, will_attend) VALUES (?, ?, ?)", 
                         ("student@fst.utm.tn", today, intent))
            
        # Log NLP Feedback
        elif 'feedback_dish' in request.form:
            dish = request.form['feedback_dish']
            # We simulate a text review based on their thumbs up/down for the NLP to analyze
            review_text = "This dish was amazing and fresh!" if request.form['rating'] == 'good' else "This was terrible, overcooked, and too salty."
            
            # --- LEVEL 2: NLP SENTIMENT SCORING ---
            score, emotion = nlp_engine.analyze_feedback(review_text)
            conn.execute("INSERT INTO feedback (date, dish_name, review_text, sentiment_score) VALUES (?, ?, ?, ?)", 
                         (today, dish, review_text, score))
            
        conn.commit()
        conn.close()
        return redirect(url_for('main.student'))
        
    return render_template('student.html')

@main_bp.route('/admin', methods=['GET', 'POST'])
def admin():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    
    # Get current RSVP counts
    rsvp_count = conn.execute("SELECT COUNT(*) as count FROM rsvp WHERE date=? AND will_attend=1", (today,)).fetchone()['count']
    
    # --- LEVEL 4: AUTONOMOUS NGO DISPATCHER ---
    # We pretend the kitchen cooked 150kg of food today. 
    # The AI calculates if the RSVPs will eat it all, or if we need to call the NGO.
    total_cooked_food_kg = 150.0 
    dispatch_needed, action_message = ngo_dispatcher.evaluate_surplus(
        predicted_attendance=500, 
        current_rsvps=rsvp_count, 
        total_cooked_kg=total_cooked_food_kg
    )
    
    if dispatch_needed:
        ai_insight = f"NGO DISPATCHED: {action_message}"
    else:
        ai_insight = action_message

    conn.close()
    return render_template('admin.html', rsvp_count=rsvp_count, ai_insight=ai_insight)