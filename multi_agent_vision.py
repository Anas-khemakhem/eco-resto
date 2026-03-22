import cv2
import numpy as np
from ultralytics import YOLO

class LocatorAgent:
    def __init__(self):
        print("[Agent 1] Booting Spatial Locator...")
        self.model = YOLO("yolov8n.pt")
        self.target_classes = ['bowl', 'dining table']

    def find_plate(self, frame):
        results = self.model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                if self.model.names[int(box.cls[0])] in self.target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Ensure coordinates are within bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    return True, frame[y1:y2, x1:x2], (x1, y1, x2, y2)
        return False, None, None

class SegmenterAgent:
    def __init__(self):
        print("[Agent 2] Booting Topological Segmenter...")
        self.model = YOLO("yolov8n-seg.pt")

    def extract_food_mass(self, plate_crop):
        if plate_crop is None or plate_crop.size == 0: return []
        results = self.model(plate_crop, verbose=False)
        food_masks = []
        for r in results:
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                for mask in masks:
                    food_masks.append(mask)
        return food_masks

class GuardAgent:
    def __init__(self):
        print("[Agent 3] Booting Contamination Guard...")
        self.model = YOLO("yolov8n.pt")
        self.hazards = ['fork', 'knife', 'spoon', 'bottle', 'cell phone', 'cup']

    def check_for_hazards(self, plate_crop):
        if plate_crop is None or plate_crop.size == 0: return []
        results = self.model(plate_crop, verbose=False)
        hazards_found = []
        for r in results:
            for box in r.boxes:
                item = self.model.names[int(box.cls[0])]
                if item in self.hazards:
                    hazards_found.append(item)
        return hazards_found

class PercentageAgent:
    def __init__(self):
        print("[Agent 4] Booting Spatial Percentage Engine...")

    def calculate_percentage(self, food_masks, plate_coords):
        x1, y1, x2, y2 = plate_coords
        plate_area = (x2 - x1) * (y2 - y1)
        
        if plate_area <= 0: return 0
        
        total_food_pixels = sum(np.sum(mask > 0) for mask in food_masks)
        
        # Calculate % based on plate bounding box
        pct = int((total_food_pixels / plate_area) * 100 * 1.5)
        return min(pct, 100)

class MultiAgentHiveMind:
    def __init__(self):
        self.locator = LocatorAgent()
        self.segmenter = SegmenterAgent()
        self.guard = GuardAgent()
        self.calculator = PercentageAgent()
        
    def process_camera_feed(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            plate_found, plate_crop, coords = self.locator.find_plate(frame)
            
            if plate_found:
                x1, y1, x2, y2 = coords
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(display_frame, "AGENT 1: PLATE LOCKED", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                hazards = self.guard.check_for_hazards(plate_crop)
                if hazards:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(display_frame, f"AGENT 3 ALERT: {hazards[0].upper()} DETECTED. DO NOT COMPOST!", 
                                (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                else:
                    food_masks = self.segmenter.extract_food_mass(plate_crop)
                    if food_masks:
                        waste_pct = self.calculator.calculate_percentage(food_masks, coords)
                        cv2.putText(display_frame, f"AGENT 2+4: {waste_pct}% ORGANIC WASTE", (x1, y2+25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            else:
                cv2.putText(display_frame, "AGENTS IDLE: AWAITING TRAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

            cv2.imshow("Eco-Resto Hive-Mind", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hive = MultiAgentHiveMind()
    hive.process_camera_feed()