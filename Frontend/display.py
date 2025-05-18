import asyncio
import websockets
import numpy as np
import cv2
import json
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO('Frontend/best.pt')

# WebSocket URLs
#TODO: Add your IP
URLS = {
    "cam3": "ws://IP:85",
    "cam4": "ws://IP:86",
    "thermal": "ws://IP:81"
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

async def display_combined_window():
    names = ["cam3", "cam4", "thermal1", "thermal2"]
    
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

        display_images = []
        total_people = 0

        # Process each image
        for img in images:
            resized_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
            result = model.predict(resized_img, imgsz=640, conf=0.3, iou=0.4, device="cpu", verbose=False)

            # Annotate for display
            annotated_img = resized_img.copy()
            if result and len(result) > 0 and result[0].boxes is not None:
                boxes = result[0].boxes.xyxy
                total_people += len(boxes)
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            display_images.append(annotated_img)

        # Combine for display
        top_row = cv2.hconcat([cv2.resize(display_images[0], (320, 320)),
                               cv2.resize(display_images[1], (320, 320))])
        bottom_row = cv2.hconcat([cv2.resize(display_images[2], (320, 320)),
                                  cv2.resize(display_images[3], (320, 320))])
        combined_display = cv2.vconcat([top_row, bottom_row])

        # Add total people count at the top
        cv2.putText(combined_display, f"Total People: {total_people}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Thermal Cameras View (YOLO Detection)", combined_display)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

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
