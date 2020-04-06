# pip install numpy imutils CMake dlib opencv-contib-python
# All images used are available from https://unsplash.com/ using the name after emotion_
import os

import cv2
import dlib
import imutils
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle


def getLandmarks(input):
    # print input
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(input)
    image = imutils.resize(image, width=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    faces = detector(gray, 1)

    if len(faces) is not 1:
        print(f"Only supports 1 face per image faces found:{len(faces)} image {input}")
        return []

    for face in faces:
        landmarks = predictor(gray, face)
        list = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            list.append(x)
            list.append(y)

    return list


def get_data_frame():
    columns = []
    for i in range(68):
        columns.append("landmarkX" + str(i))
        columns.append("landmarkY" + str(i))
    columns.append('expression')

    arrayOfLandmarks = []
    for imageName in os.listdir("./data/"):
        print(f"started processing ./data/{imageName}")
        emotion = imageName.split("_")[0]
        landmarks = getLandmarks("./data/" + imageName)
        if len(landmarks) is not 0:
            landmarks.append(emotion)
            arrayOfLandmarks.append(landmarks)
        print(f"finished processing ./data/{imageName}")

    return pd.DataFrame(data=arrayOfLandmarks, columns=columns)


def run_classifier(x, y, test_size, random_state, feature_range, classifier):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    nb_clf = Pipeline(
        [("scaler", MinMaxScaler(feature_range=feature_range)),
         ('classifier', classifier)])

    nb_clf.fit(x_train, y_train)

    print("Accuracy", nb_clf.score(x_test, y_test))
    print("Confusion Matrix")
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, nb_clf.predict(x_test)))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(confusion_matrix_df)


# Running all functions in correct order
print("Getting data frame with landmarks as features")
data_frame = get_data_frame()
df = shuffle(data_frame, random_state=10)
data_frame_x = data_frame.drop(['expression'], axis=1)
data_frame_y = data_frame['expression']

print("Running NN analysis")
run_classifier(data_frame_x, data_frame_y, 0.3, 10, (-1, 1), MLPClassifier(max_iter=300, hidden_layer_sizes=50, shuffle=True, activation='relu', solver='lbfgs'))
print("Running SVC analysis")
run_classifier(data_frame_x, data_frame_y, 0.3, 10, (-1, 1), SVC(class_weight='balanced', kernel='rbf', C=1, degree=2, gamma=0.1))
print("Running Linear SVC analysis")
run_classifier(data_frame_x, data_frame_y, 0.3, 10, (-1, 1), LinearSVC(C=1, dual=False))
print("Running Naive bays analysis")
run_classifier(data_frame_x, data_frame_y, 0.3, 10, (0, 1), MultinomialNB(alpha=0.005))
