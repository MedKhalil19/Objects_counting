import cv2
import numpy as np
from ultralytics import YOLO
import threading
import torch


class YOLOv8ObjectCounter:
    def __init__(self, weights_path, confidence_threshold=0.4):  # Adjusted threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(weights_path)  # Model loaded
        self.confidence_threshold = confidence_threshold
        self.classes = self.model.names
        self.colors = self.generate_colors(len(self.classes))

    def generate_colors(self, num_colors):
        """Generate unique colors for each class."""
        np.random.seed(42)
        return [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                for _ in range(num_colors)]

    def detect_objects(self, img):
        """Detect objects in the image."""
        original_height, original_width = img.shape[:2]
        img_resized = cv2.resize(img, (640, 640))

        img_tensor = torch.from_numpy(img_resized).float().to(self.device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        results = self.model(img_tensor)
        detections = []

        for result in results:
            for box in result.boxes:
                confidence = box.conf.item()
                if confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    scale_x = original_width / 640
                    scale_y = original_height / 640
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    w, h = x2 - x1, y2 - y1
                    class_id = int(box.cls.item())
                    detections.append(((x1, y1, w, h), confidence, class_id))

        return detections

    def draw_labels(self, img, detections):
        """Draw bounding boxes and labels on the image."""
        for box, confidence, class_id in detections:
            x, y, w, h = box
            color = self.colors[class_id]
            label = f"{self.classes[class_id]} ({confidence:.2f})"
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def count_objects(self, detections):
        """Count occurrences of each detected class."""
        count_dict = {}
        for _, _, class_id in detections:
            count_dict[class_id] = count_dict.get(class_id, 0) + 1
        return count_dict

    def display_counts(self, img, counts):
        """Display counts of detected objects on the image."""
        y_offset = 30
        for class_id, count in counts.items():
            label = f"{self.classes[class_id]}: {count}"
            color = self.colors[class_id]
            cv2.putText(img, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            y_offset += 30
        return img


def process_frame(frame, detector):
    detections = detector.detect_objects(frame)
    frame = detector.draw_labels(frame, detections)
    counts = detector.count_objects(detections)
    frame = detector.display_counts(frame, counts)
    return frame


def video_capture(detector):
    """Capture and process video frames from the webcam."""
    cap = cv2.VideoCapture(0)

    # Set desired resolution
    desired_width, desired_height = 1280, 720  # Standard HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # Create a named window and resize it
    window_name = 'YOLOv8 Object Detection and Counting'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, desired_width, desired_height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, detector)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    weights_path = r"C:\Users\khkha\OneDrive\Bureau\Real-Time-Object-Counting-main\yolov8s.pt"
    detector = YOLOv8ObjectCounter(weights_path)

    # Start video capture in a separate thread
    capture_thread = threading.Thread(target=video_capture, args=(detector,))
    capture_thread.start()
