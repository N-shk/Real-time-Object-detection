from ultralytics import YOLO
import cv2
import math 

# Load YOLOv8 Nano pre-trained model
model = YOLO("yolo-Weights/yolov8n.pt")  # Automatically downloads if not present

# Load a video file instead of using the webcam
cap = cv2.VideoCapture('VID_20240702_110935.mp4')  # Specify your video file path

# Object classes (COCO dataset)
classNames = model.names

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Perform object detection
    results = model(img, stream=True)

    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence score
            confidence = round(box.conf[0].item(), 2)
            print("Confidence:", confidence)

            # Class name
            cls = int(box.cls[0].item())
            class_name = classNames[cls]
            print("Class Name:", class_name)

            # Display class name and confidence on the image
            cv2.putText(img, f'{class_name} {confidence}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display the result in a window
    cv2.imshow('YOLOv8 Object Detection', img)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
