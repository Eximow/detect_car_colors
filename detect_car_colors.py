import cv2
import numpy as np
from ultralytics import YOLO
import sys

def get_color_name_and_sample(bgr_roi):
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    mask_color = (V > 40) & (V < 230) & (S > 30)
    if np.count_nonzero(mask_color) < 0.1 * mask_color.size:
        mask_color = (V > 30) & (V < 245) & (S > 15)

    mean_S = float(np.mean(S))
    mean_V = float(np.mean(V))

    if mean_S <= 25:
        if mean_V >= 200:
            color_name = "White"
        elif mean_V <= 60:
            color_name = "Black"
        else:
            color_name = "Silver" if mean_V >= 140 else "Gray"
    else:
        dominant_hue = int(np.median(H[mask_color])) if np.count_nonzero(mask_color) > 0 else int(np.median(H))
        if (0 <= dominant_hue <= 10) or (160 <= dominant_hue <= 179):
            color_name = "Red"
        elif 11 <= dominant_hue <= 25:
            color_name = "Orange"
        elif 26 <= dominant_hue <= 35:
            color_name = "Yellow"
        elif 36 <= dominant_hue <= 85:
            color_name = "Green"
        elif 86 <= dominant_hue <= 125:
            color_name = "Blue"
        elif 126 <= dominant_hue <= 159:
            color_name = "Purple"
        else:
            color_name = "Unknown"

    return color_name

def detect_car_colors(input_image_path: str, output_image_path: str = "output.jpg", conf_threshold: float = 0.3):
    model = YOLO("yolov8n.pt")
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {input_image_path}")

    results = model(img, conf=conf_threshold)[0]
    vehicle_classes = {2, 3, 5, 7}
    count = 0

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        cls_id = int(cls.item())
        if cls_id in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.tolist())
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            cname = get_color_name_and_sample(roi)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 180, 60), 2)

            # Put text in bold red
            cv2.putText(img, cname, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            count += 1

    cv2.putText(img, f"Vehicles: {count}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 255), 2)

    cv2.imwrite(output_image_path, img)
    print(f"Number of vehicles: {count}")
    print(f"Output image saved: {output_image_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_car_colors.py <input_image_path> [output_image_path] [confidence_threshold]")
        print("Example: python detect_car_colors.py street.jpg output.jpg 0.35")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "output.jpg"
    conf = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.3

    detect_car_colors(input_path, output_path, conf)
