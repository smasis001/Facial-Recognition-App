from pkg_resources import resource_filename
import time
import cv2
import dlib
import mediapipe as mp

CAMERA_ID = 2
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540

def detect_faces_with_mp(img_rgb, detector):
    faces = detector.process(img_rgb)
    img_h, img_w, _ = img_rgb.shape
    dlib_faces = []
    if faces.detections:
        for face in faces.detections:
            bbox = face.location_data.relative_bounding_box
            x1, y1 = int(bbox.xmin * img_w), int(bbox.ymin * img_h)
            w, h = int(bbox.width * img_w), int(bbox.height * img_h)
            dlib_faces.append(dlib.rectangle(x1, y1, x1+w, y1+h))
    return dlib_faces

def draw_bounding_boxes(frame, faces):
    for face in faces:
        x, y, w, h = face.left(), face.top(),\
                     face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h),\
                     (0, 255, 0), 3)

def get_landmarks_with_mp(img_rgb, faces, predictor, subset=None):
    subset_68_idxs = [127, 234, 93, 215, 172, 136, 150, 176, 152, 400, 379,\
                      365, 367, 433, 366, 447, 372, 70, 63, 105, 66, 107, 336,\
                      296, 334, 293, 276, 168, 197, 195, 4, 240, 97, 2, 326,\
                      290, 33, 160, 158, 133, 153, 144, 362, 385, 386, 249, 373,\
                      380, 61, 39, 37, 11, 267, 269, 291, 321, 314, 17, 85, 181,\
                      78, 82, 13, 402, 308, 402, 14, 87]
    subset_5_idxs = [249, 362, 33, 155, 2]
    if subset is None:
        subset = MP_LANDMARK_SUBSET
    if subset == 'large':
        subset_idxs = subset_68_idxs
    elif subset == 'small':
        subset_idxs = subset_5_idxs
    else:
        subset_idxs = [*range(0, 468)]

    raw_landmarks = predictor.process(img_rgb)
    h, w, _ = img_rgb.shape
    dlib_landmarks = []
    if raw_landmarks.multi_face_landmarks:
        for f, face_landmarks in enumerate(raw_landmarks.multi_face_landmarks):
            dlib_parts = []
            for p in subset_idxs:
                pt = face_landmarks.landmark[p]
                dlib_parts.append(dlib.point(round(pt.x*w), round(pt.y*h)))
            dlib_fod = dlib.full_object_detection(rect=faces[f], parts=dlib_parts)
            dlib_landmarks.append(dlib_fod)

    return dlib_landmarks

def draw_landmarks(frame, landmarks):
    for landmark in landmarks:
        for i in range(landmark.num_parts):
            x = landmark.part(i).x
            y = landmark.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

def show_webcam_feed():
    cap = cv2.VideoCapture(CAMERA_ID)
    if WINDOW_WIDTH is not None and WINDOW_HEIGHT is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.\
                            FaceDetection(min_detection_confidence=0.5)
    mp_face_mesh = mp.solutions.face_mesh
    landmark_predictor = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,\
                                            refine_landmarks=True, min_detection_confidence=0.5)
    current_time = time.time()

    while True:
        ret, frame = cap.read()

        if ret:
            delta = (time.time() - current_time)
            delta = delta if delta != 0 else 0.000001
            current_time = time.time()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detect_faces_with_mp(frame_rgb, face_detector)

            if faces:
                draw_bounding_boxes(frame, faces)
                landmarks = get_landmarks_with_mp(frame_rgb, faces,\
                                                    landmark_predictor)
                draw_landmarks(frame, landmarks)

            fps = 1 / delta
            cv2.putText(frame, f"FPS: {fps :02.1f}", (30, 30),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Detecting Facial Landmarks', frame)

            key = cv2.waitKey(1)
            if key == 27:  # Escape key
                break

    cap.release()
    cv2.destroyAllWindows()

# Main execution block
if __name__ == '__main__':
    show_webcam_feed()
