from textblob import TextBlob
import sqlite3

class SentimentAnalysisAgent:
    def __init__(self):
        print("[NLP Agent] Natural Language Processing Engine Online.")

    def analyze_feedback(self, text):
        """
        Uses Lexicon-based NLP to determine if a student's review is positive or negative.
        Returns a score between -1.0 (Terrible) and 1.0 (Amazing).
        """
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity
        
        # Determine human-readable emotion
        if sentiment_score > 0.3: emotion = "Positive 😊"
        elif sentiment_score < -0.2: emotion = "Negative 😠"
        else: emotion = "Neutral 😐"
            
        return sentiment_score, emotion

    def generate_chef_report(self, db_path):
        """Scans the database and finds the most hated dish to warn the chef."""
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT dish_name, AVG(sentiment_score) as avg_score FROM feedback GROUP BY dish_name ORDER BY avg_score ASC LIMIT 1")
        worst_dish = c.fetchone()
        conn.close()
        
        if worst_dish and worst_dish[1] is not None and worst_dish[1] < 0:
            return f"CRITICAL NLP ALERT: Students strongly dislike the {worst_dish[0].upper()}. Average Sentiment: {worst_dish[1]:.2f}"
        return "NLP STATUS: Student sentiment is stable."

nlp_engine = SentimentAnalysisAgent()