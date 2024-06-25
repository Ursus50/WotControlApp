import pyautogui
import cv2
import mediapipe as mp
import torch

from mlp_model import load_model_mlp
from cnn_model import load_model_cnn
from utils import load_dictionary_from_file

from actions import choose_action

import time

def get_model(type_net):
    if type_net == "mlp":
        # Załaduj wytrenowany model
        input_size = 63
        hidden_size1 = 128
        hidden_size2 = 32
        output_size = 9
        loaded_model = load_model_mlp("mlp_model_9_912.pth", input_size, hidden_size1, hidden_size2, output_size)
    elif type_net == "cnn":
        # Załaduj wytrenowany model
        input_channels = 1
        input_length = 63
        hidden_size1 = 128
        hidden_size2 = 32
        output_size = 9
        loaded_model = load_model_cnn("cnn_mlp_model_9_907.pth", input_channels, input_length, hidden_size1, hidden_size2,
                                      output_size)
    else:
        raise print("Wrong type of net")

    loaded_model.eval()
    return loaded_model


def predict_gesture(input_data):
    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)
        predicted = str(predicted.item())  # Extract the number from the tensor
        predicted_gesture = dictionary_gesture[predicted]
    return predicted_gesture

def get_name_gesture(number):
    pass

if __name__ == "__main__":
    # model = None
    model = get_model("cnn")

    pathDictionary = "class_names.json"
    dictionary_gesture = load_dictionary_from_file(pathDictionary)
    # for key in dictionary_gesture:
    #     print(key)
    #     print(type(key))
    # print(dictionary_gesture)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Ustawienia ekranu
    screen_width, screen_height = pyautogui.size()

    last_gesture = None
    # time.sleep(5)
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)

                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                    list_of_points = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        for land in hand_landmarks.landmark:
                            list_of_points.append(land.x)
                            list_of_points.append(land.y)
                            list_of_points.append(land.z)

                    # print(len(list_of_points))
                    if len(list_of_points) == 63:
                        gesture = predict_gesture(list_of_points)
                        print(gesture)
                        if gesture != last_gesture:
                            choose_action(gesture)
                            last_gesture = gesture






                # print(list_of_points)

                # # Landmark dla indeksu (palec wskazujący)
                # index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                #
                # # Konwersja współrzędnych do zakresu ekranu
                # x = int(index_finger_tip.x * screen_width)
                # y = int(index_finger_tip.y * screen_height)
                #
                # Sterowanie myszką
                # pyautogui.moveTo(x, y)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    choose_action("Fist")
    cap.release()
    cv2.destroyAllWindows()
