from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import math
import threading

app = Flask(__name__)

# Model Initialization
model = YOLO("best.pt")

# Object Classes
classNames = ["drowning","swimming","person out of water"]

# Webcam Thread
class WebcamThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.stopped = False

    def run(self):
        while not self.stopped:
            success, img = self.cap.read()
            results = model(img, stream=True)

            # Count variable
            box_count = 0

            # coordinates
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    # class name
                    cls = int(box.cls[0])
                    print("Class name -->", classNames[cls])

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

                    # Increment box count
                    box_count += 1

            # Display box count
            cv2.putText(img, f"Detected_Box: {box_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def stop(self):
        self.stopped = True
        self.cap.release()



@app.route('/video_feed')
def video_feed():
    return Response(gen(WebcamThread()), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    for frame in camera.run():
        yield frame

if __name__ == '__main__':
    app.run(debug=True)
