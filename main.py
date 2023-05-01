import torch
import torchvision
import cv2

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def face_detection():
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Set model to evaluation mode
    model.eval()

    # Load image
    img = cv2.imread("Photos\Alexandera.jpg")

    # Convert image to tensor
    img_tensor = torchvision.transforms.functional.to_tensor(img)

    # Add a batch dimension to the tensor
    img_tensor = img_tensor.unsqueeze(0)

    # Run image through the model
    output = model(img_tensor)

    # Get the bounding boxes and labels for detected faces
    boxes = output[0]["boxes"]
    labels = output[0]["labels"]

    # Loop through the detected faces and draw bounding boxes on the image
    for i in range(len(boxes)):
        if labels[i] == 1:  # 1 represents the label for faces
            box = boxes[i].detach().numpy()
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def object_detection(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print(torch.__version__)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread('Photos\Alexandera.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a rectangle around the detected objects:
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the image with the detected objects:
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# def face_detection_from_video():
#     # Get a reference to webcam
#     video_capture = cv2.VideoCapture(0)
#
#     # Initialize variables
#     face_locations = []
#
#     while True:
#         # Grab a single frame of video
#         ret, frame = video_capture.read()
#
#         # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#         rgb_frame = frame[:, :, ::-1]
#
#         # Find all the faces in the current frame of video
#         face_locations = face_recognition.face_locations(rgb_frame)
#
#         # Display the results
#         for top, right, bottom, left in face_locations:
#             # Draw a box around the face
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#
#         # Display the resulting image
#         cv2.imshow('Video', frame)
#
#         # Hit 'q' on the keyboard to quit!
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release handle to the webcam
#     video_capture.release()
#     cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    # object_detection('This is the object detection Progran')

    face_detection()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
