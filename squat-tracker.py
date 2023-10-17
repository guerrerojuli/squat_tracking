import cv2
import mediapipe as mp
import math
from dataclasses import dataclass


@dataclass
class Vector:
    x: float
    y: float
    angle: float
    length: float


def aligned(x1, x2):
    diff = x1-x2
    if -0.02 < diff < 0.02:
        return True
    return False


def get_angle_form_vector(v1, v2):
    angle = math.acos(
            (v1.x*v2.x + v1.y*v2.y) /
            (math.sqrt(pow(v1.x, 2) + pow(v1.y, 2)) *
             math.sqrt(pow(v2.x, 2) + pow(v2.y, 2)))
            )
    return angle


def create_vector(p1, p2):
    x = p1.x - p2.x
    y = p2.y - p2.y
    angle = math.atan(y/x)
    length = math.sqrt(pow(x, 2) + pow(y, 2))
    return Vector(x, y, angle, length)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extraer marcadores
        try:
            landmarks = results.pose_landmarks.landmark
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            r_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
            l_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
            points = [
                    r_shoulder, l_shoulder,
                    r_ankle, l_ankle,
                    r_hip, l_hip,
                    r_knee, l_knee,
                    r_heel, l_heel
            ]

            # Validacion de pose
            # Existencia de los puntos
            existence = True
            for point in points:
                if (point.visibility*100 <= 1):
                    existence = False

            if (not existence):
                print("Muestre su cuerpo a la camara")
                pass

            # Camara de perfil
            if (not aligned(r_shoulder.x, l_shoulder.x)):
                print("Coloque la camara de perfil")
                pass

            # Buena postura TODO
            # Barra alineada
            if (
                    not aligned(r_shoulder.y, r_knee.y) or
                    not aligned(l_shoulder.y, l_knee.y)
                    ):
                print("Mala postura, barra caida")
                pass

            # Maquina de estados TODO

        except:
            pass

        # Renderizar detecciones
        mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(85, 205, 195), thickness=10, circle_radius=2),
                mp_drawing.DrawingSpec(
                    color=(174, 50, 60), thickness=10, circle_radius=2)
                )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
