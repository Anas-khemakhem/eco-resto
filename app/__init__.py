from flask import Flask
from .routes import main_bp
from .models import init_db

def create_app():
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.secret_key = "research_grade_secret_key"
    
    # Initialize the single clean database
    init_db()
    
    # Register all our URLs
    app.register_blueprint(main_bp)
    
    return app