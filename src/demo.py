import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


class Demo:
    def __init__(self, emotions: dict, image_input_size: tuple, model_path: str):
        self.haar_classifier_path = '../data/haarcascade_frontalface_default.xml'
        self.emotions = emotions
        self.image_input_size = image_input_size
        self.model_path = model_path

    def run_demo(self):
        """
        Opens a new window which runs our emotion detection model on the live input video from a webcam
        """

        # use haar cascade classifier to detect faces
        face_cascade = cv2.CascadeClassifier(self.haar_classifier_path)

        # load our model
        classifier = load_model(self.model_path)

        # create a video capture object
        vid = cv2.VideoCapture(0)

        # number of frames elapsed
        frames = 0

        # classify emotions every X frames
        every_x_frames = 5

        # emotion's label and position of the label
        label = ""
        label_position = (0, 0)

        while True:

            frames = frames + 1

            # read the video per frame from the camera
            ret, frame = vid.read()

            # convert the captured frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the frame and store the pixel coordinates in faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:

                # mark the face with a rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (181, 215, 14), thickness=3)

                # rescale the frame to what our model accepts
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, self.image_input_size, interpolation=cv2.INTER_AREA)

                if frames % every_x_frames == 0:
                    frames = 0
                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype('float') / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        # make predictions on the face using our model
                        preds = classifier.predict(roi)[0]

                        # pick the prediction with the highest confidence
                        pred_index = preds.argmax()

                        # set the label of the predicted emotion with its prediction confidence percentage
                        label = self.emotions[pred_index] + " %.2f%%" % float(preds[pred_index] * 100)

                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # display the resulting frame
            cv2.imshow('Emotion Detection - (press q to quit)', frame)

            # if q is hit the window will close
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # after the loop release the cap object and destroy window objects
        vid.release()
        cv2.destroyAllWindows()
