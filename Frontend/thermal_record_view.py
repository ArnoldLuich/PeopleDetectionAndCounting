import asyncio
import websockets
import numpy as np
import cv2
import json
import os
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO('C:/Users/arnol/Desktop/Inimeste loendamine/display/best_v3.pt')

# WebSocket URLs
URLS = {
    "cam3": "ws://172.20.10.4:85",
    "cam4": "ws://172.20.10.2:86",
    "thermal": "ws://172.20.10.5:81"
}

# Frame sizes
FRAME_WIDTH = 32
FRAME_HEIGHT = 24
FINAL_IMAGE_SIZE = (640, 640)

# Global dictionary for images
latest_images = {
    "cam3": None,
    "cam4": None,
    "thermal1": None,
    "thermal2": None
}

# Recording state
recording = False
save_counter = 0
save_base_path = "dataset"
os.makedirs(os.path.join(save_base_path, "images"), exist_ok=True)
os.makedirs(os.path.join(save_base_path, "labels"), exist_ok=True)

def digital_zoom(img, zoom_factor=1):
    h, w = img.shape[:2]
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    cropped = img[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, FINAL_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

def preprocess_frame(thermal_data):
    try:
        thermal_frame = thermal_data.reshape((FRAME_HEIGHT, FRAME_WIDTH))
        normalized = cv2.normalize(thermal_frame, None, 0, 255, cv2.NORM_MINMAX)
        resized = cv2.resize(normalized, (320, 240), interpolation=cv2.INTER_LINEAR)
        resized_uint8 = resized.astype(np.uint8)
        colored = cv2.applyColorMap(resized_uint8, cv2.COLORMAP_INFERNO)
        top_pad = (320 - 240) // 2
        padded = cv2.copyMakeBorder(
            colored, top=top_pad, bottom=320-240-top_pad, left=0, right=0,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return padded
    except Exception as e:
        print("Error processing frame:", e)
        return np.zeros((320, 320, 3), dtype=np.uint8)

def csv_to_image(csv_data):
    thermal_data = np.array(list(map(float, csv_data.split(','))), dtype=np.float32)
    return preprocess_frame(thermal_data)

def json_to_images(json_data):
    try:
        data = json.loads(json_data)
        img1 = csv_to_image(data["thermal1"])
        img2 = csv_to_image(data["thermal2"])
        return img1, img2
    except Exception as e:
        print("Error parsing JSON:", e)
        return None, None

async def listen_to_camera(name, url):
    async with websockets.connect(url) as websocket:
        print(f"Connected to {name} at {url}")
        while True:
            try:
                message = await websocket.recv()
                if name == "thermal":
                    img1, img2 = json_to_images(message)
                    if img1 is not None and img2 is not None:
                        latest_images["thermal1"] = img1
                        latest_images["thermal2"] = img2
                else:
                    img = csv_to_image(message)
                    if img is not None:
                        latest_images[name] = img
            except Exception as e:
                print(f"Connection to {name} lost: {e}")
                break

def save_yolo_format(results, filename, img_width, img_height):
    """Save YOLO format labels"""
    with open(filename, 'w') as f:
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                center_x = (x1 + x2) / 2.0 / img_width
                center_y = (y1 + y2) / 2.0 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

async def display_combined_window():
    global recording, save_counter
    names = ["cam3", "cam4", "thermal1", "thermal2"]
    zoom_factor = 1  # Set your zoom level here

    while True:
        images = [
            latest_images.get("cam3"),
            latest_images.get("cam4"),
            latest_images.get("thermal1"),
            latest_images.get("thermal2")
        ]

        # Replace None with black frames
        for i in range(len(images)):
            if images[i] is None:
                images[i] = np.zeros((320, 320, 3), dtype=np.uint8)

        clean_images = []
        display_images = []
        predictions = []

        # Process each image
        for img in images:
            zoomed_img = digital_zoom(img, zoom_factor=zoom_factor)
            clean_images.append(zoomed_img.copy())
            result = model.predict(zoomed_img, imgsz=640, conf=0.3, iou=0.4, device="cpu", verbose=False)
            predictions.append(result)

            # Annotate for display
            annotated_img = zoomed_img.copy()
            if result and len(result) > 0 and result[0].boxes is not None:
                for box in result[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            display_images.append(annotated_img)

        # Combine for display
        top_row = cv2.hconcat([cv2.resize(display_images[0], (320, 320)),
                               cv2.resize(display_images[1], (320, 320))])
        bottom_row = cv2.hconcat([cv2.resize(display_images[2], (320, 320)),
                                  cv2.resize(display_images[3], (320, 320))])
        combined_display = cv2.vconcat([top_row, bottom_row])

        cv2.imshow("Thermal Cameras View (YOLO Detection + Digital Zoom)", combined_display)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            recording = not recording
            print("Recording:", recording)

        if recording:
            for idx, img in enumerate(clean_images):
                img_filename = os.path.join(save_base_path, "images", f"{names[idx]}_{save_counter:04d}.jpg")
                label_filename = os.path.join(save_base_path, "labels", f"{names[idx]}_{save_counter:04d}.txt")
                cv2.imwrite(img_filename, img)
                save_yolo_format(predictions[idx], label_filename, 640, 640)
                print(f"Saved {img_filename} and {label_filename}")

            save_counter += 1

        await asyncio.sleep(0.01)

async def main():
    await asyncio.gather(
        listen_to_camera("cam3", URLS["cam3"]),
        listen_to_camera("cam4", URLS["cam4"]),
        listen_to_camera("thermal", URLS["thermal"]),
        display_combined_window()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
    cv2.destroyAllWindows()
