import cv2
import numpy as np
from ultralytics import YOLO
import os

def load_model(model_path):
    """Load the YOLO model."""
    return YOLO(model_path)

def preprocess_image(image_path):
    """Read and return the input image."""
    img = cv2.imread(image_path)
    return img, img.shape

def perform_inference(model, img):
    """Perform object detection."""
    results = model(img)
    return results[0].boxes

def extract_detections(detections, image_shape):
    """Extract and filter detection results."""
    raw_height, raw_width, _ = image_shape
    boxes_no, confidences_score, class_ids = [], [], []

    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        confidence = detection.conf[0].tolist()
        class_id = int(detection.cls[0].tolist())

        if confidence > 0.3:
            x1, y1, x2, y2 = map(lambda v: max(0, min(v, raw_width if v in [x2, y2] else raw_height)), [x1, y1, x2, y2])
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            boxes_no.append([x, y, w, h])
            confidences_score.append(float(confidence))
            class_ids.append(class_id)

    return boxes_no, confidences_score, class_ids

def draw_boxes_and_labels(img, boxes_no, confidences_score, class_ids, classes, colors):
    """Draw bounding boxes and labels on the image."""
    indices = cv2.dnn.NMSBoxes(boxes_no, confidences_score, 0.3, 0.4)
    track = [class_ids[i] for i in indices.flatten()]

    for i in indices.flatten():
        x, y, w, h = boxes_no[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return track

def count_objects(track):
    """Count occurrences of each object class."""
    count_dict = {}
    for obj_id in track:
        if obj_id in count_dict:
            count_dict[obj_id] += 1
        else:
            count_dict[obj_id] = 1
    return count_dict

def display_counts(img, count_dict, classes, colors):
    """Display counts of objects on the image."""
    x_label, x_count, y = 20, 240, 30  # Adjusted x_count for spacing
    line_height = 30  # Height between lines for clear visibility

    for obj_id, count in count_dict.items():
        label = str(classes[obj_id])
        color = colors[obj_id]
        # Draw the label and count on the image at adjusted x positions
        cv2.putText(img, label, (x_label, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(img, f": {count}", (x_count, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        y += line_height  # Move down for the next label and count

def save_image(img):
    """Save the image to a specified path."""
    while True:
        save_path = input("Enter the path to save the image (e.g., 'output.jpg'): ").strip()
        if os.path.isdir(os.path.dirname(save_path)) or os.path.dirname(save_path) == "":
            try:
                if cv2.imwrite(save_path, img):
                    print(f"Image saved to {save_path}")
                    break
                else:
                    print("Failed to save image. Please try again.")
            except Exception as e:
                print(f"Error saving image: {e}. Please try again.")
        else:
            print("Invalid path. Please try again.")

def main():
    model_path = r"C:\Users\khkha\OneDrive\Bureau\Real-Time-Object-Counting-main\yolov8s.pt"
    image_path = "./1.jpg"

    # Load model and image
    model = load_model(model_path)
    img, image_shape = preprocess_image(image_path)

    # Perform inference and extract detections
    detections = perform_inference(model, img)
    boxes_no, confidences_score, class_ids = extract_detections(detections, image_shape)

    # Define colors and draw bounding boxes and labels
    classes = model.names
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    track = draw_boxes_and_labels(img, boxes_no, confidences_score, class_ids, classes, colors)

    # Count objects and display counts
    count_dict = count_objects(track)
    display_counts(img, count_dict, classes, colors)

    # Prompt user for action
    while True:
        print("Choose an option:")
        print("1. Display image")
        print("2. Save image")

        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == '1':
            # Display the image
            cv2.imshow('Image', img) #./path/output.jpg
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        elif choice == '2':
            # Save the image
            save_image(img)
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
