import cv2 as cv
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_conf=0.5):
        self.min_detection_conf = min_detection_conf
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_conf)
        self.results = None

    def find_faces(self, img, draw=True):

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for idx, detection in enumerate(self.results.detections):
                # mp_draw.draw_detection(img, detection)
                # print(idx, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), \
                    int(bbox_c.width * iw), int(bbox_c.height * ih)
                bboxs.append([idx, bbox, detection.score])
                if draw:
                    # cv.rectangle(img, bbox, (255, 0, 255), 2)
                    img = self.fancy_draw(img, bbox=bbox)
                    cv.putText(img, f'{int(detection.score[0] * 100)}%',
                               (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return img, bboxs

    @staticmethod
    def fancy_draw(img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left x,y
        cv.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv.line(img, (x, y), (x, y + l), (255, 0, 255), t)

        # Top Right x1,y
        cv.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        # Bottom Left x,y1
        cv.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        # Bottom Right x1,y1
        cv.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img


def main():
    cap = cv.VideoCapture(0)
    p_time = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.find_faces(img)
        print(bboxs)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
