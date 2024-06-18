from ultralytics import YOLO

model = YOLO('models/comic-speech-bubble-detector.pt')  
def detect_bubbles(image):
    results = model([image])  
    for result in results:
        boxes = result.boxes  
        masks = result.masks  
        keypoints = result.keypoints  
        probs = result.probs  
        #result.save(filename='result.jpg')
        
    return boxes
