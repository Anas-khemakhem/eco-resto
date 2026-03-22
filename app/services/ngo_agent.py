from datetime import datetime

class AutonomousNGOAgent:
    def __init__(self):
        print("[NGO Agent] Surplus Distribution Matrix Online.")
        self.trigger_threshold_kg = 15.0 # If we have > 15kg of surplus, call the NGO

    def evaluate_surplus(self, predicted_attendance, current_rsvps, total_cooked_kg):
        """
        Calculates if we have massive surplus and auto-dispatches help.
        """
        # Assume 300g (0.3kg) per student portion
        expected_consumption_kg = current_rsvps * 0.3
        projected_surplus_kg = total_cooked_kg - expected_consumption_kg

        if projected_surplus_kg >= self.trigger_threshold_kg:
            dispatch_time = datetime.now().strftime("%H:%M")
            action = (
                f"🚨 [AUTO-DISPATCH INITIATED at {dispatch_time}]\n"
                f"Projected Surplus: {projected_surplus_kg:.1f} kg of fresh food.\n"
                f"Action: Alerting local Food Bank for immediate pickup at 14:00."
            )
            return True, action
        
        return False, "Surplus is within normal parameters. No NGO dispatch required."

ngo_dispatcher = AutonomousNGOAgent()