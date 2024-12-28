from ultralytics import YOLO
import cv2
import pytesseract
import os

# Load YOLOv8 model (pre-trained or custom ANPR model)
model = YOLO('yolov8n.pt')  # Replace with your ANPR-trained YOLO model

# Function to detect vehicles and number plates
def detect_objects(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Run YOLO inference
    results = model(image_path)
    
    for result in results[0].boxes:
        # Extract detection info
        box = result.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        class_id = int(result.cls[0].cpu().numpy())
        confidence = result.conf[0].cpu().numpy()
        label = model.names[class_id]
        
        # Check if the detection is a number plate (label should match your ANPR model class)
        if label == "number_plate":
            x1, y1, x2, y2 = map(int, box)
            # Draw bounding box for number plate
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Crop the detected number plate region
            number_plate_region = img[y1:y2, x1:x2]
            
            # Perform OCR on the cropped number plate
            plate_text = pytesseract.image_to_string(number_plate_region, config='--psm 7')
            print(f"Detected Number Plate: {plate_text.strip()}")
            
            # Display detected number plate text on the image
            cv2.putText(img, plate_text.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display the results
    cv2.imshow('Detected Objects', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


# Example usage
detect_objects("img1.jpg") 
detect_objects("image 2.jpg") 
detect_objects("img3.jpg") 
detect_objects("img4.jpg") 
detect_objects("image5.jpg") 
detect_objects("img6.jpg") 
detect_objects("img7.jpg") 
detect_objects("img8.jpg") 
detect_objects("img9.jpg") 
detect_objects("img10.jpg") 

