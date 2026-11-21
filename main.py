import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "8"

import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

path = "data/test.mp4"
cap = cv2.VideoCapture(path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
corrupted_in_a_row = 0
max_corrupted_streak = 30
skipped_total = 0

while cap.isOpened():
    ret, frame = cap.read()
    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if not ret or frame is None or frame.size == 0:
        corrupted_in_a_row += 1
        skipped_total += 1
        
        if (total_frames > 0 and current_pos >= total_frames) or corrupted_in_a_row >= max_corrupted_streak:
            print(f"Відео завершено на кадрі {current_pos}/{total_frames}. Пропущено битих кадрів: {skipped_total}")
            break
        continue

    corrupted_in_a_row = 0

    results = model(frame, classes=[0], conf=0.4, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("Human Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()