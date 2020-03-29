import imutils
import cv2
import dlib



def getLandmarks(input):
    #print input
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(input)
    image = imutils.resize(image, width=500)
    cv2.imshow("image", image)  # Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit program when the user presses 'q'
        print('finished')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    faces = detector(gray, 1)

    for face in faces:

        landmarks = predictor(gray, face)
        list = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            cv2.circle(image, (x, y), 1, (0, 0, 255), thickness=2)  # For each point, draw a red circle with thickness2 on the original frame

            list.append((x, y))

    while True:
        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit program when the user presses 'q'
            break





print(getLandmarks('sampleimage.jpeg'))