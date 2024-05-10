from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8m.pt")

def draw_boxes(frame, results):
    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            cls = r.names[box.cls[0].item()]

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, cls, org, font, fontScale, color, thickness)

    return frame


def video_detection(cap):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'h264'), fps, (frame_width, frame_height))

    count = 0
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        results = model(frame, stream=True, device='cuda', verbose=False)

        frame = draw_boxes(frame, results)

        out.write(frame)
        if not count % 10:
            yield frame, None
        # print(count)
        count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    yield None, 'output_video.mp4'
