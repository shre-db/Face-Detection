import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
p_time = 0

mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.75)

while True:
    success, img = cap.read()
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    print(results)

    if results.detections:
        for idx, detection in enumerate(results.detections):
            # mp_draw.draw_detection(img, detection)
            # print(idx, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)

            cv.rectangle(img, bbox, (255, 0, 255), 2)
            cv.putText(img, f'{int(detection.score[0]*100)}%',
                       (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv.imshow("Image", img)
    cv.waitKey(1)
