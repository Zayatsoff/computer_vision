import cv2
from facenet_pytorch import MTCNN


class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):
        """
        Draw bounding box, probs, and landmarks
        """

        for box, prob, ld in zip(boxes, probs, landmarks):
            # Draw box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)
            # Draw probability
            cv2.putText(
                frame,
                str(prob),
                (box[2], box[3]),
                cv2.FONT_HERSHEY_DUPLEX,
                9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            # Draw landmarks
            for i in range(5):
                cv2.circle(frame, tuple(ld[i]), 5, (255, 0, 0))

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            try:
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                self._draw(frame, boxes, probs, landmarks)
            except:
                print("No faces found.")
                pass

            cv2.imshow("Face Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


mtcnn = MTCNN()
detector = FaceDetector(mtcnn)
detector.run()
