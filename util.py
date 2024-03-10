import cv2
import numpy as np
import mediapipe as mp
import pickle

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels = {0: 'A', 1: 'B', 2: 'L'}

def detect_hand_gesture(image):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape

    results = hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)

        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        max_length = 84

        # Pad sequences with zeros to make them all the same length
        if len(data_aux) < max_length:
            data_aux += [0.0] * (max_length - len(data_aux))

        prediction = model.predict([np.asarray(data_aux, dtype=np.float64)])
        character = labels[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        return {
            "x1": x1,
            "y1": y1, 
            "x2": x2,
            "y2": y2,
        }

    # img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # img.save("image.jpg")

    # return x1, y1, x2, y2
