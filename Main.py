import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch

class YOLOv8ObjectCounter:
    def __init__(self, weights_path, confidence_threshold=0.4, iou_threshold=0.4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(weights_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = self.model.names
        self.colors = self.generate_colors(len(self.classes))

    def generate_colors(self, num_colors):
        np.random.seed(42)
        return [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                for _ in range(num_colors)]

    def preprocess_image(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return cv2.resize(img_eq, (640, 640))

    def count_objects(self, img):
        original_height, original_width = img.shape[:2]
        img_resized = self.preprocess_image(img)
        img_tensor = torch.from_numpy(img_resized).float().to(self.device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        results = self.model(img_tensor)
        counts = []

        for result in results:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            class_ids = result.boxes.cls

            if len(boxes) == 0:
                continue

            for i, box in enumerate(boxes):
                confidence = confidences[i].item()
                if confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.tolist()
                    scale_x = original_width / 640
                    scale_y = original_height / 640
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    w, h = x2 - x1, y2 - y1
                    class_id = int(class_ids[i].item())
                    counts.append(((x1, y1, w, h), confidence, class_id))

        if counts:
            boxes, scores, class_ids = zip(*counts)
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)
            counts = [counts[i] for i in indices.flatten()] if indices is not None else []

        return counts

    def draw_labels(self, img, counts):
        for box, confidence, class_id in counts:
            x, y, w, h = box
            color = self.colors[class_id]
            label = f"{self.classes[class_id]} ({confidence:.2f})"
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def count_objects_in_frame(self, counts):
        count_dict = {}
        for _, _, class_id in counts:
            count_dict[class_id] = count_dict.get(class_id, 0) + 1
        return count_dict

    def display_counts(self, img, counts):
        y_offset = 30
        for class_id, count in counts.items():
            label = f"{self.classes[class_id]}: {count}"
            color = self.colors[class_id]
            cv2.putText(img, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            y_offset += 30
        return img

def process_frame(frame, counter):
    counts = counter.count_objects(frame)
    frame = counter.draw_labels(frame, counts)
    count_dict = counter.count_objects_in_frame(counts)
    frame = counter.display_counts(frame, count_dict)
    return frame

def video_capture(counter, save=False, output_path=None):
    cap = cv2.VideoCapture(0)
    desired_width, desired_height = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    window_name = 'YOLOv8 Object Counting'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, desired_width, desired_height)

    if save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (desired_width, desired_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, counter)

        if save:
            out.write(frame)
        else:
            cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save:
        out.release()
    cv2.destroyAllWindows()

def process_image(image_path, counter, save=False, output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        messagebox.showerror("Error", f"Error loading image: {image_path}")
        return

    img = process_frame(img, counter)

    if save:
        if output_path is None:
            output_path = filedialog.asksaveasfilename(title="Save Image As", defaultextension=".jpg",
                                                       filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if output_path:
            cv2.imwrite(output_path, img)
            messagebox.showinfo("Success", f"Image saved successfully: {output_path}")
    else:
        cv2.imshow('Processed Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_video(video_path, counter, save=False, output_path=None):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') if save else None
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4)))) if save else None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, counter)

        if save:
            out.write(frame)
        else:
            cv2.imshow('Processed Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save:
        out.release()
        # Show success message after saving the video
        messagebox.showinfo("Success", f"Video saved successfully: {output_path}")
    cv2.destroyAllWindows()

def start_real_time_counting(weights_path, save=False, output_path=None):
    counter = YOLOv8ObjectCounter(weights_path)
    video_capture(counter, save, output_path)

def start_image_counting(weights_path):
    counter = YOLOv8ObjectCounter(weights_path)
    image_path = filedialog.askopenfilename(title="Select Image File",
                                            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if image_path:
        choice = messagebox.askquestion("Save or Display", "Do you want to save the processed image?", icon='question')
        if choice == 'yes':
            output_path = filedialog.asksaveasfilename(title="Save Image As", defaultextension=".jpg",
                                                       filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
            process_image(image_path, counter, save=True, output_path=output_path)
        else:
            process_image(image_path, counter)

def start_video_counting(weights_path):
    counter = YOLOv8ObjectCounter(weights_path)
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4 *.avi")])
    if video_path:
        choice = messagebox.askquestion("Save or Display", "Do you want to save the processed video?", icon='question')
        if choice == 'yes':
            output_path = filedialog.asksaveasfilename(title="Save Video As", defaultextension=".avi",
                                                       filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4")])
            if output_path:
                process_video(video_path, counter, save=True, output_path=output_path)
        else:
            process_video(video_path, counter)

def load_icon(path):
    try:
        icon = Image.open(path)
        icon = icon.resize((32, 32), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(icon)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def on_real_time():
    weights_path = filedialog.askopenfilename(title="Select Weights File", filetypes=[("PyTorch files", "*.pt")])
    if weights_path:
        Thread(target=start_real_time_counting, args=(weights_path,)).start()

def on_image_counting():
    weights_path = filedialog.askopenfilename(title="Select Weights File", filetypes=[("PyTorch files", "*.pt")])
    if weights_path:
        start_image_counting(weights_path)

def on_video_counting():
    weights_path = filedialog.askopenfilename(title="Select Weights File", filetypes=[("PyTorch files", "*.pt")])
    if weights_path:
        start_video_counting(weights_path)

def create_gui():
    root = tk.Tk()
    root.title("YOLOv8 Object Counting")
    root.geometry("600x500")

    # Add a descriptive label
    tk.Label(root, text="Select a mode to start object counting:", font=("Arial", 12)).pack(pady=10)

    # Load icons
    real_time_photo = load_icon("icons/real_time.png")
    image_photo = load_icon("icons/image.png")
    video_photo = load_icon("icons/video.png")
    quit_photo = load_icon("icons/quit.png")

    # Create button frame to hold buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    # Add buttons with icons and text
    tk.Button(button_frame, text="Real-Time Counting", image=real_time_photo, compound=tk.LEFT, command=on_real_time,
              font=("Arial", 10), height=50, width=300).pack(pady=5, fill=tk.X)
    tk.Button(button_frame, text="Image Counting", image=image_photo, compound=tk.LEFT, command=on_image_counting,
              font=("Arial", 10), height=50, width=300).pack(pady=5, fill=tk.X)
    tk.Button(button_frame, text="Video Counting", image=video_photo, compound=tk.LEFT, command=on_video_counting,
              font=("Arial", 10), height=50, width=300).pack(pady=5, fill=tk.X)

    # Add the Quit button with icon
    tk.Button(button_frame, text="Quit", image=quit_photo, compound=tk.LEFT, command=root.quit, font=("Arial", 10),
              height=50, width=300).pack(pady=5, fill=tk.X)

    # Add a status bar
    global status_bar
    status_bar = tk.Label(root, text="Select an option to start", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # Add the watermark
    watermark = tk.Label(root, text="by Khlifi Med Khalil", font=("Arial", 11), fg="gray")
    watermark.pack(side=tk.BOTTOM, anchor=tk.SE, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
