import cv2
from ultralytics import YOLO
from pathlib import Path


model = YOLO("./runs/detect/figures/yolo/weights/best.pt")
camera = cv2.VideoCapture(0)
classes = {0: 'cube', 1: 'neither', 2: 'sphere'}


while camera.isOpened():
    ret, frame = camera.read()
    key = cv2.waitKey(10) & 0xFF

    result = model.predict(source=frame, conf=0.25, iou=0.1, imgsz=640)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    for box, label, score in zip(boxes, cls, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{classes[int(label)]}: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Camera", frame)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()