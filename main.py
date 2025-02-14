import pyautogui
import cv2
import mediapipe as mp
import torch
import threading
import queue

from mlp_model import load_model_mlp
from cnn_model import load_model_cnn
from utils import load_dictionary_from_file

from actions import choose_action, move_to, move, right, left, up, down

import time


def get_model(type_net):
    if type_net == "mlp":
        # Załaduj wytrenowany model
        input_size = 63
        hidden_size1 = 128
        hidden_size2 = 32
        output_size = 9
        loaded_model = load_model_mlp("mlp_model_9_954.pth", input_size, hidden_size1, hidden_size2, output_size)
    elif type_net == "cnn":
        # Załaduj wytrenowany model
        input_channels = 1
        input_length = 63
        hidden_size1 = 128
        hidden_size2 = 32
        output_size = 9
        loaded_model = load_model_cnn("cnn_mlp_model_9_907.pth", input_channels, input_length, hidden_size1,
                                      hidden_size2,
                                      output_size)
    else:
        raise ValueError("Wrong type of net")

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


def cursor_track(que):

    while True:
        try:
            cords = que.get_nowait()  # Próba pobrania elementu z kolejki bez blokowania
            if cords[0] is None:
                break  # Przerwij pętlę, jeśli otrzymano sygnał zakończenia

            x = cords[0]
            y = cords[1]

            if x > screen_width_margin_right or x < screen_width_margin_left or y > screen_height_margin_up or y < screen_height_margin_down:
                if x > screen_width_margin_right:
                    right()
                elif x < screen_width_margin_left:
                    left()
                if y > screen_height_margin_up:
                    down()
                elif y < screen_height_margin_down:
                    up()
            else:
                move_to(x, y)

            que.task_done()  # Zakończone przetwarzanie elementu
        except queue.Empty:
            # Kolejka jest pusta, możemy zrobić coś innego, na przykład:
            time.sleep(0.01)  # Zrób krótką przerwę, aby nie obciążać CPU

def make_action(que):

    while True:
        try:
            gesture = que.get_nowait()  # Próba pobrania elementu z kolejki bez blokowania
            if gesture is None:
                break  # Przerwij pętlę, jeśli otrzymano sygnał zakończenia

            choose_action(gesture)

            que.task_done()  # Zakończone przetwarzanie elementu
        except queue.Empty:
            # Kolejka jest pusta, możemy zrobić coś innego, na przykład:
            time.sleep(0.01)  # Zrób krótką przerwę, aby nie obciążać CPU



if __name__ == "__main__":

    model = get_model("mlp")

    pathDictionary = "class_names.json"
    dictionary_gesture = load_dictionary_from_file(pathDictionary)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    screen_width, screen_height = pyautogui.size()

    margin_width = 0.2
    margin_height = 0.25
    screen_width_margin_right = (1 - margin_width) * screen_width
    screen_width_margin_left = margin_width * screen_width
    screen_height_margin_up = (1 - margin_height) * screen_height
    screen_height_margin_down = margin_height * screen_height

    last_gesture = None
    last_last_gesture = None

    # Utworzenie kolejki
    q_cursor = queue.Queue()

    # Uruchomienie wątku odbierającego
    cursor_thread = threading.Thread(target=cursor_track, args=(q_cursor,))
    cursor_thread.start()

    # Utworzenie kolejkiwaaawa
    q_action = queue.Queue()

    # Uruchomienie wawwątku odbierającegowawaaawa
    action_thread = threading.Thread(target=make_action, args=(q_action,))
    action_thread.start()

    index_right = None
    time_stamp = 1

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        # FPS counter variables
        prev_time = time.time()
        fps = 0

        gesture = "None"

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time


            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            results = hands.process(image)

            image.flags.writeable = True
            # Display FPS on the image
            cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            if results.multi_hand_landmarks:

                if results.multi_handedness[0].classification[0].label == 'Right' or len(results.multi_hand_landmarks) == 1:
                    index_right = 0
                else:
                    index_right = 1


                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)

                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                    list_of_points = []


                    for land in results.multi_hand_landmarks[index_right].landmark:
                        list_of_points.append(land.x)
                        list_of_points.append(land.y)
                        list_of_points.append(land.z)

                    if len(list_of_points) == 63:
                        x = int(list_of_points[0] * screen_width)
                        y = int(list_of_points[1] * screen_height)


                        if time_stamp % 3 == 0:
                            cords = (x, y)
                            q_cursor.put(cords)

                        gesture = predict_gesture(list_of_points)
                        print(gesture)


                        if gesture == last_gesture and last_last_gesture != last_gesture:
                            q_action.put(gesture)

                        last_last_gesture = last_gesture
                        last_gesture = gesture


                        time_stamp += 1

            cv2.putText(image, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # Wysłanie sygnału zakończenia
    q_cursor.put((None, None))
    # Oczekiwanie na zakończenie wątku odbierającegos
    cursor_thread.join()

    # Wysłanie sygnału zakończenia
    q_action.put(None)
    # Oczekiwanie na zakończenie wątku odbierającego
    action_thread.join()

    choose_action("Fist")
    cap.release()
    cv2.destroyAllWindows()
