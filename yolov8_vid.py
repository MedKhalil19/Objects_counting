import cv2
import numpy as np
from ultralytics import YOLO
import os

class YOLOv8ObjectCounter:
    def __init__(self, weights_path, confidence_threshold=0.5, nms_threshold=0.4):
        # Initialize YOLO model on CPU
        self.device = 'cpu'
        self.model = YOLO(weights_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.classes = self.model.names
        self.colors = self.generate_colors(len(self.classes))

    def generate_colors(self, num_colors):
        """Generate unique colors for each class."""
        return [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in
                range(num_colors)]

    def detect_objects(self, img):
        """Detect objects in the frame."""
        input_size = (416, 416)  # Reduced input size for faster performance on CPU
        img_resized = cv2.resize(img, input_size)
        results = self.model(img_resized, device=self.device, conf=self.confidence_threshold, iou=self.nms_threshold)[0]

        detections = []
        scale_x = img.shape[1] / input_size[0]
        scale_y = img.shape[0] / input_size[1]

        for box in results.boxes:
            if box.conf.item() >= self.confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                detections.append(((x1, y1, x2 - x1, y2 - y1), box.conf.item(), int(box.cls.item())))

        return detections

    def draw_labels(self, img, detections):
        """Draw bounding boxes and labels on the frame."""
        for box, confidence, class_id in detections:
            x, y, w, h = box
            color = self.colors[class_id]
            label = f"{self.classes[class_id]}"
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
        """Display counts of detected objects on the frame."""
        y_offset = 30
        for class_id, count in counts.items():
            label = f"{self.classes[class_id]}: {count}"
            color = self.colors[class_id]
            cv2.putText(img, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            y_offset += 30
        return img

def process_frame(frame, detector, total_counts):
    """Process each frame to detect objects, draw labels, and update total counts."""
    detections = detector.detect_objects(frame)
    frame = detector.draw_labels(frame, detections)
    counts = detector.count_objects(detections)
    frame = detector.display_counts(frame, counts)

    # Update total counts
    for class_id, count in counts.items():
        total_counts[class_id] = total_counts.get(class_id, 0) + count

    return frame

def process_video(video_path, detector, skip_frames=2):
    """Process the video file frame by frame."""
    cap = cv2.VideoCapture(video_path)
    total_counts = {}  # Dictionary to store the total counts
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to improve processing speed
        if frame_index % skip_frames == 0:
            frame = process_frame(frame, detector, total_counts)
            cv2.imshow('YOLOv8 Object Detection and Counting', frame)

        frame_index += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return total_counts

def save_video(video_path, detector, output_path, skip_frames=2):
    """Process and save the video file."""
    cap = cv2.VideoCapture(video_path)
    total_counts = {}  # Dictionary to store the total counts
    frame_index = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video saving
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to improve processing speed
        if frame_index % skip_frames == 0:
            frame = process_frame(frame, detector, total_counts)
            out.write(frame)

        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print total counts after processing the video
    print("Total Object Counts:")
    for class_id, count in total_counts.items():
        print(f"{detector.classes[class_id]}: {count}")

def main():
    weights_path = r"C:\Users\khkha\OneDrive\Bureau\Real-Time-Object-Counting-main\yolov8s.pt"
    video_path = "1.mp4"  # Change to your video file path

    detector = YOLOv8ObjectCounter(weights_path)

    while True:
        print("Choose an option:")
        print("1. Display video")
        print("2. Save video")

        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == '1':
            # Process and display the video
            process_video(video_path, detector)
            break
        elif choice == '2':
            while True:
                output_path = input("Enter the path to save the video (e.g., 'output.avi'): ").strip()
                if os.path.isdir(os.path.dirname(output_path)) or os.path.dirname(output_path) == "":
                    try:
                        save_video(video_path, detector, output_path)
                        print(f"Video saved to {output_path}")
                        break
                    except Exception as e:
                        print(f"Error saving video: {e}. Please try again.")
                else:
                    print("Invalid path. Please try again.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
