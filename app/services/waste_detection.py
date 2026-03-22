import cv2
import numpy as np
import base64
from ultralytics import YOLO

class MenuAwareHybridVision:
    def __init__(self):
        print("[Vision Core] Booting Menu-Aware Hybrid AI + Nutritional Engine...")
        self.ai_model = YOLO("yolov8n-seg.pt") 
        
        self.hazards = ['fork', 'knife', 'spoon', 'bottle', 'cell phone']
        self.containers = ['bowl', 'cup', 'dining table']
        
        self.signatures = {
            'couscous': [((10, 50, 50), (35, 255, 255))],
            'spaghetti': [((0, 70, 50), (10, 255, 255)), ((160, 70, 50), (180, 255, 255))],
            'rice': [((0, 0, 150), (180, 40, 255))],
            'mra9': [((0, 50, 20), (15, 255, 150))],
            'meat': [((0, 50, 20), (15, 255, 150))],
            'escalope': [((10, 50, 50), (25, 255, 200))],
            'salad': [((35, 40, 40), (85, 255, 255))],
            'apple': [((0, 70, 50), (10, 255, 255)), ((160, 70, 50), (180, 255, 255))],
            'orange': [((10, 100, 100), (25, 255, 255))],
            'banana': [((20, 100, 100), (35, 255, 255))]
        }

        # AGENT 5: Nutritional Matrix (per 1% of plate waste)
        self.nutrition_db = {
            'Couscous': {'kcal': 5.2, 'protein': 0.15},
            'Spaghetti': {'kcal': 4.8, 'protein': 0.12},
            'Meat': {'kcal': 7.5, 'protein': 0.80},
            'Escalope': {'kcal': 6.0, 'protein': 0.90},
            'Salad': {'kcal': 0.5, 'protein': 0.02},
            'Mra9': {'kcal': 4.0, 'protein': 0.30},
        }

    def _normalize_lighting(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    def learn_new_signature(self, correct_item, b64_string):
        """
        ZERO-SHOT LEARNING: Analyzes the corrected image, finds the dominant color,
        and dynamically updates the AI's live memory for this session!
        """
        print(f"[Active Learning] Extracting live color signature for: {correct_item}")
        correct_item = correct_item.lower().replace(' ', '_')
        
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
            
        import base64
        image_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Look at the center 30% of the image to find the dominant color of the food
        h, w = frame.shape[:2]
        center_roi = frame[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
        
        if center_roi.size == 0: return
        
        hsv_roi = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
        
        # Calculate the average Hue, Saturation, and Value of the center
        avg_h = int(np.mean(hsv_roi[:,:,0]))
        avg_s = int(np.mean(hsv_roi[:,:,1]))
        avg_v = int(np.mean(hsv_roi[:,:,2]))
        
        # Create a new custom color range based on this exact food
        lower_bound = (max(0, avg_h - 10), max(50, avg_s - 40), max(50, avg_v - 40))
        upper_bound = (min(180, avg_h + 10), 255, 255)
        
        # Inject it into the live memory!
        if correct_item not in self.signatures:
            self.signatures[correct_item] = []
            
        self.signatures[correct_item].append((lower_bound, upper_bound))
        print(f"[Active Learning] Memory Updated! New signature mapped to {correct_item}: {lower_bound} to {upper_bound}")
    def analyze_base64_image(self, b64_string, active_menu):
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
            
        image_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        raw_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = self._normalize_lighting(raw_frame)

        results = self.ai_model(frame, verbose=False)
        plate_mask = None
        max_plate_area = 0
        hazard_found = None
        confidence = 0.0

        for r in results:
            if r.masks is None: continue
            masks = r.masks.data.cpu().numpy()
            for i, box in enumerate(r.boxes):
                cls_name = self.ai_model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                # 🔴 FIX: ONLY trigger hazard if confidence is > 65%
                if cls_name in self.hazards and conf > 0.65:
                    hazard_found = cls_name
                    
                if cls_name in self.containers and conf > 0.40:
                    mask_resized = cv2.resize(masks[i], (frame.shape[1], frame.shape[0]))
                    area = np.sum(mask_resized > 0)
                    if area > max_plate_area:
                        max_plate_area = area
                        plate_mask = (mask_resized > 0).astype(np.uint8)
                        confidence = conf

        if hazard_found:
            return "Contamination", 0, f"GUARD ALERT: {hazard_found.upper()} detected.", 0.99, {"kcal":0, "protein":0}

        if plate_mask is None:
            h, w = frame.shape[:2]
            plate_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(plate_mask, (w//2, h//2), min(h, w)//3, 1, -1)
            max_plate_area = np.sum(plate_mask > 0)
            confidence = 0.50

        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        plate_roi = cv2.bitwise_and(blurred_frame, blurred_frame, mask=plate_mask)
        hsv_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2HSV)
        
        breakdown = {}
        total_waste_pct = 0
        total_kcal = 0.0
        total_protein = 0.0
        
        valid_items = [item.lower().replace(' ', '_') for item in active_menu if item and item.lower() != 'none']

        for item in valid_items:
            if item in self.signatures:
                item_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
                for (lower, upper) in self.signatures[item]:
                    item_mask = cv2.bitwise_or(item_mask, cv2.inRange(hsv_roi, np.array(lower), np.array(upper)))
                
                kernel = np.ones((5, 5), np.uint8)
                item_mask = cv2.morphologyEx(item_mask, cv2.MORPH_OPEN, kernel)
                
                pixel_count = cv2.countNonZero(item_mask)
                item_pct = int((pixel_count / max_plate_area) * 100 * 1.5)
                
                if item_pct > 3: 
                    item_pct = min(item_pct, 100)
                    formatted_name = item.replace('_', ' ').title()
                    breakdown[formatted_name] = item_pct
                    total_waste_pct += item_pct
                    
                    # Add Nutritional Data
                    nutrients = self.nutrition_db.get(formatted_name, {'kcal': 2.0, 'protein': 0.05})
                    total_kcal += item_pct * nutrients['kcal']
                    total_protein += item_pct * nutrients['protein']

        total_waste_pct = min(total_waste_pct, 100)

        if not breakdown:
            return "Clean Plate", 0, "No organic waste detected.", confidence, {"kcal":0, "protein":0}

        detected_string = " | ".join([f"{name}: {pct}%" for name, pct in breakdown.items()])
        nutrition_data = {"kcal": int(total_kcal), "protein": round(total_protein, 1)}
        
        return detected_string, total_waste_pct, "Multi-Agent Scan Complete.", confidence, nutrition_data

vision_ai = MenuAwareHybridVision()