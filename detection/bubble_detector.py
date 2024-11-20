from ultralytics import YOLO


class BubbleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_bubbles(self, image):
        try:
            results = self.model([image])
            for result in results:
                return result.boxes
        except Exception as e:
            print(f"Error during bubble detection: {e}")
            return []
