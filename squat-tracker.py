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


def aligned(x1, x2, tolerance):
    diff = x1-x2
    if -tolerance < diff < tolerance:
        return True
    return False


def between(x, x1, x2, tolerance):
    if (
            (x1-tolerance < x < x2+tolerance) or
            (x1+tolerance > x > x2-tolerance)
            ):
        return True
    return False


def get_angle_form_vector(v1, v2):
    angle = math.acos(
            (v1.x*v2.x + v1.y*v2.y) /
            (v1.length * v2.length)
            )
    return angle


def create_vector(p1, p2):
    x = p1.x - p2.x
    y = p1.y - p2.y
    angle = math.atan(y/x)
    length = math.sqrt(pow(x, 2) + pow(y, 2))
    return Vector(x, y, angle, length)


def reset_flags(flagsDict):
    for key in flagsDict:
        flagsDict[key] = False


msgFlags = {
        "landmarksExistance": False,
        "camera": False,
        "ready": False,
        }


errorFlags = {
        "misalignedBarbell": False,
        "feet": False,
        "notFull": False,
        }

statusFlags = {
        "onProcess": False,
        "down": False
        }

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolorear la imagen a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Realizar la detección
        results = pose.process(image)

        # Recolorear nuevamente a BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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

        cv2.imshow('Squat Tracker', image)

        # Cortar el programa al precionar Q
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # Extraer marcadores
        try:
            landmarks = results.pose_landmarks.landmark
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_foot_index = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            l_foot_index = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            r_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
            l_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
            points = [
                    r_shoulder, l_shoulder,
                    r_ankle, l_ankle,
                    r_hip, l_hip,
                    r_foot_index, l_foot_index,
                    r_knee, l_knee,
                    r_heel, l_heel
            ]
        except:
            if (not msgFlags["landmarksExistance"]):
                print("Muestre su cuerpo a la camara\n")
                reset_flags(errorFlags)
                msgFlags["landmarksExistance"] = True
            continue

        # Validacion de pose
        # Existencia de los puntos
        existence = True
        for point in points:
            if (not 0 < point.x < 1 or not 0 < point.y < 1):
                existence = False

        if (not existence):
            if (not msgFlags["landmarksExistance"]):
                print("Muestre su cuerpo a la camara\n")
                reset_flags(msgFlags)
                msgFlags["landmarksExistance"] = True
            continue
        msgFlags["landmarksExistance"] = False

        # Camara de perfil
        if (not aligned(r_shoulder.x, l_shoulder.x, 0.05)):
            if (not msgFlags["camera"]):
                print("Coloque la camara de perfil\n")
                reset_flags(msgFlags)
                msgFlags["camera"] = True
            continue
        msgFlags["camera"] = False

        if (not msgFlags["ready"]):
            print("Realice la sentadilla\n")
            reset_flags(msgFlags)
            msgFlags["ready"] = True
        


        # Maquina de estados
        v1 = create_vector(r_hip, r_knee)
        v2 = create_vector(r_heel, r_knee)
        angle = get_angle_form_vector(v1, v2)
        if (angle < math.pi/2):
            statusFlags["down"] = True
        elif (angle < (3/4)*math.pi):
            statusFlags["onProcess"] = True
        else:
            if (statusFlags["onProcess"]):
                print("\n")
                if (not statusFlags["down"]):
                    errorFlags["notFull"] = True
                print(f"""
                      - Pie apoyado completamente: {"Bien" if(not errorFlags["feet"]) else "Mal"}
                      - Barra alineada: {"Bien" if(not errorFlags["misalignedBarbell"]) else "Mal"}
                      - Recorrido completo: {"Bien" if(not errorFlags["notFull"]) else "Mal"}\n
                      """)
                reset_flags(errorFlags)
                reset_flags(statusFlags)

        # Buena postura
        # Barra alineada
        if (not between(r_shoulder.x, r_foot_index.x, r_heel.x, 0.02)):
            errorFlags["misalignedBarbell"] = True

        # Pie apoyado
        if (
                not aligned(r_foot_index.y, r_heel.y, 0.05) and
                not aligned(l_foot_index.y, l_heel.y, 0.05)
                ):
            errorFlags["feet"] = True

    cap.release()
    cv2.destroyAllWindows()
