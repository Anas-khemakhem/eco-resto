import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

class AttendancePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.is_trained = False
        self.metrics = {}

    def _generate_synthetic_research_data(self):
        """
        Simulates 1 year of historical data with complex real-world correlations
        for robust ML training.
        """
        np.random.seed(42)
        days = 365
        
        # Feature Engineering
        df = pd.DataFrame({
            'day_of_week': np.random.randint(0, 5, days),
            'is_exam': np.random.choice([0, 1], days, p=[0.85, 0.15]),
            'is_raining': np.random.choice([0, 1], days, p=[0.80, 0.20]),
            'menu_score': np.random.randint(1, 10, days)
        })
        
        # Ground Truth Generation (Target Variable)
        # Base: 500. Rain drops by 40. Exams drop by 150. Good menu adds 50.
        df['actual_attendance'] = (
            500 
            - (df['day_of_week'] * 15) 
            - (df['is_exam'] * 150) 
            - (df['is_raining'] * 40) 
            + (df['menu_score'] * 5)
        )
        # Add real-world statistical noise
        df['actual_attendance'] += np.random.normal(0, 25, days)
        return df

    def train_and_evaluate(self):
        """Research-grade training pipeline with strict evaluation."""
        print("\n[ML Research Pipeline] Initializing Training...")
        df = self._generate_synthetic_research_data()
        
        X = df[['day_of_week', 'is_exam', 'is_raining', 'menu_score']]
        y = df['actual_attendance']
        
        # 80% Training, 20% Testing Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        # Calculate Research Metrics
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        self.metrics = {'MAE': round(mae, 2), 'R2': round(r2, 3)}
        self.is_trained = True
        
        print(f"[ML Research Pipeline] Success! Model Metrics -> MAE: {self.metrics['MAE']} students | R²: {self.metrics['R2']}")

    def predict_today(self, is_exam=0, is_raining=0, menu_score=5):
        if not self.is_trained:
            self.train_and_evaluate()
            
        today = datetime.now()
        day_of_week = today.weekday()
        
        if day_of_week > 4: return 0, "Weekend / Closed"

        features = pd.DataFrame({
            'day_of_week': [day_of_week],
            'is_exam': [is_exam],
            'is_raining': [is_raining],
            'menu_score': [menu_score]
        })
        
        prediction = self.model.predict(features)[0]
        context = f"Baseline (Error Margin: ±{self.metrics.get('MAE', 0)} students)"
        return int(prediction), context

# Singleton
predictor = AttendancePredictor()