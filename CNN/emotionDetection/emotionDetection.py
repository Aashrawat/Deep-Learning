import os
import sys
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FACE_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
EMOTION_MODEL = os.path.join(MODELS_DIR, "emotion-ferplus.onnx")
WEIGHTS_PATH = os.path.join(BASE_DIR, "emotion_model.weights.h5")
JSON_PATH = os.path.join(BASE_DIR, "emotion_model.json")

# FER+ output order
EMOTIONS = [
    "Neutral",
    "Happy",
    "Surprised",
    "Sad",
    "Angry",
    "Disgusted",
    "Fearful",
    "Contempt",
]


def train_model():
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers.schedules import ExponentialDecay
    import matplotlib.pyplot as plt

    train_data_gen = ImageDataGenerator(rescale=1.0 / 255)
    validation_data_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_data_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode="categorical",
    )
    validation_generator = validation_data_gen.flow_from_directory(
        TEST_DIR,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode="categorical",
    )

    emotion_model = Sequential()
    emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
    emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))
    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation="relu"))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(7, activation="softmax"))
    emotion_model.summary()

    lr_schedule = ExponentialDecay(0.0001, decay_steps=100000, decay_rate=0.96)
    emotion_model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=lr_schedule),
        metrics=["accuracy"],
    )

    history = emotion_model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=30,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
    )
    emotion_model.evaluate(validation_generator)

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.title("Accuracy Graph")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Loss Graph")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    print(plt.savefig(os.path.join(BASE_DIR, "training_history.png")))
    plt.close()

    with open(JSON_PATH, "w") as json_file:
        json_file.write(emotion_model.to_json())
    emotion_model.save_weights(WEIGHTS_PATH)


# Neutral and contempt are over-predicted. Sad, angry, and fear need a lift
# so a real expression is not labeled Neutral.
CLASS_WEIGHTS = np.array(
    [0.62, 1.0, 0.85, 1.35, 1.25, 0.9, 1.7, 0.45], dtype=np.float32
)


def softmax(logits):
    shifted = logits - np.max(logits)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


def balance_scores(probabilities):
    adjusted = probabilities * CLASS_WEIGHTS
    total = np.sum(adjusted)
    if total <= 0:
        return probabilities
    return adjusted / total


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def run_camera():
    if not os.path.exists(FACE_MODEL) or not os.path.exists(EMOTION_MODEL):
        raise SystemExit("Emotion models are missing from the models folder.")

    face_detector = cv2.FaceDetectorYN.create(FACE_MODEL, "", (640, 480), 0.7, 0.3, 5000)
    emotion_net = cv2.dnn.readNetFromONNX(EMOTION_MODEL)
    emotion_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    emotion_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    cap = open_camera()
    smoothed_faces = []
    input_size = None
    last_time = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.perf_counter()
        elapsed = now - last_time
        last_time = now
        if elapsed > 0:
            fps = (0.9 * fps) + (0.1 * (1.0 / elapsed))

        height, width = frame.shape[:2]
        if input_size != (width, height):
            face_detector.setInputSize((width, height))
            input_size = (width, height)

        _, faces = face_detector.detect(frame)
        if faces is None:
            smoothed_faces = []
            faces = []
        if len(smoothed_faces) != len(faces):
            smoothed_faces = [None] * len(faces)

        for index_face, face in enumerate(faces):
            x, y, box_w, box_h = face[:4]
            # Extra forehead and chin so brows and the mouth are in the crop.
            # Angry, sad, and fear are read from those areas.
            x1 = max(int(x - 0.12 * box_w), 0)
            y1 = max(int(y - 0.28 * box_h), 0)
            x2 = min(int(x + box_w + 0.12 * box_w), width)
            y2 = min(int(y + box_h + 0.12 * box_h), height)
            if x2 <= x1 or y2 <= y1:
                continue

            gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            blob = resized.astype(np.float32).reshape(1, 1, 64, 64)
            emotion_net.setInput(blob)
            probabilities = balance_scores(softmax(emotion_net.forward().ravel()))
            previous = smoothed_faces[index_face]
            if previous is None:
                smoothed_faces[index_face] = probabilities
            else:
                smoothed_faces[index_face] = (0.25 * previous) + (0.75 * probabilities)

            scores = smoothed_faces[index_face]
            index = int(np.argmax(scores))
            label = f"{EMOTIONS[index]} {scores[index]:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"{fps:.0f} FPS",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--train" in sys.argv:
        train_model()
    run_camera()
