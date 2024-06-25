import pydirectinput
import time

def clear_keys():
    pydirectinput.keyUp('w')
    pydirectinput.keyUp('a')
    pydirectinput.keyUp('s')
    pydirectinput.keyUp('d')

def move_to(x, y):
    pydirectinput.moveTo(x, y)

def choose_action(gesture):
    clear_keys()
    if gesture == "Freedom":
        pydirectinput.click()
    elif gesture == "IndexFinger":
        pydirectinput.keyDown('w')

    elif gesture == "Thumb":
        pydirectinput.keyDown('a')
        # pydirectinput.keyUp('a')
    elif gesture == "Pinky":
        pydirectinput.keyDown('d')
        # pydirectinput.keyUp('d')
    elif gesture == "CallMe":
        pydirectinput.keyDown('s')
        # pydirectinput.keyUp('s')
    elif gesture == "Plat":
        pydirectinput.keyDown('shift')
        pydirectinput.keyUp('shift')
    elif gesture == "L":
        pydirectinput.keyDown('w')
        pydirectinput.keyDown('a')
        # pydirectinput.keyUp('w')
        # pydirectinput.keyUp('a')
    elif gesture == "Devil":
        pydirectinput.keyDown('w')
        pydirectinput.keyDown('d')
        # pydirectinput.keyUp('w')
        # pydirectinput.keyUp('d')
    elif gesture == "Fist":
        pass        

if __name__ == "__main__":
    # Przykładowe użycie funkcji choose_action
    choose_action("IndexFinger")

    while True:
        # Tutaj możesz dodać pętlę, która rejestruje gesty i wywołuje odpowiednie akcje
        # Na przykład:
        # gesture = funkcja_do_rejestrowania_gestów()
        # choose_action(gesture)
        pass
