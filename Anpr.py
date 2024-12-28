from ultralytics import YOLO
import cv2
import os
import easyocr

# Load YOLO model
model = YOLO('yolov8n.pt')  # Use 'yolov8m.pt' or 'yolov8l.pt' for more accuracy
vehicle_classes = ["car", "motorcycle", "bus", "truck", "person"]

# Load Haar Cascade for license plate detection
plate_cascade = cv2.CascadeClassifier("model/haarcascade_russian_plate_number.xml")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def detect_objects_and_extract_text(image_path, cascade, save_dir='plates', min_area=500):
    if not os.path.isfile(image_path):
        print(f"File {image_path} not found.")
        return

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect vehicles using YOLO
    results = model(image_path)
    for result in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, confidence, class_id = result
        label = model.names[int(class_id)]
        if label in vehicle_classes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Detect license plates using Haar Cascade
    plates = cascade.detectMultiScale(img_gray, 1.1, 4)
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for (x, y, w, h) in plates:
        if w * h > min_area:
            # Extract plate image
            plate_img = img[y:y+h, x:x+w]

            # Recognize text using EasyOCR
            text_results = reader.readtext(plate_img)
            text = " ".join([result[1] for result in text_results])
            print(f"Extracted Text: {text}")

            # Annotate the image with the detected text
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Save the plate image
            cv2.imwrite(f"{save_dir}/plate_{count}.jpg", plate_img)
            count += 1

    # Display the processed image
    cv2.imshow("Combined Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Directory containing images
image_folder = r"C:\Users\sai\OneDrive\Desktop\ANPR\pictures"

# Process all images in the folder
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
for img_path in image_paths:
    detect_objects_and_extract_text(img_path, plate_cascade)