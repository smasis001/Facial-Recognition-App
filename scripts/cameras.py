"""Camera Testing Script"""
# pylint: disable=E1101
import cv2

def test_camera(index):
    """Attempt to capture from the camera at the given index."""
    cap = cv2.VideoCapture(index)
    if not cap.read()[0]:
        return None
    else:
        cap.release()
        return index

def test_all_cameras():
    """Find all available camera IDs and return their count."""
    index = 0
    while True:
        if test_camera(index) is not None:
            print(f"Found camera with ID: {index}")
            index += 1
        else:
            break
    return index

def release_all_cameras(max_cameras_to_check=10):
    """Attempt to release all cameras up to a maximum index."""
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        cap.release()

def main():
    """Loop through all cameras"""
    release_all_cameras(10)
    total_cameras = test_all_cameras()
    #release_all_cameras(total_cameras)
    if total_cameras == 0:
        print("No cameras found.")
        return

    current_camera_id = 0
    cap = cv2.VideoCapture(current_camera_id)

    while True:
        ret, frame = cap.read()
        if ret:
            # Calculate FPS
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Display Camera ID and FPS
            cv2.putText(frame, f"Camera ID: {current_camera_id} | FPS: {fps:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Camera Feed", frame)

            key = cv2.waitKey(1) & 0xFF

            # Check for keys to change camera
            if key in [ord("q"), 27]:  # 'q' or 'ESC' key
                break
            elif key in [ord("d"), ord("r"), 83, 82, 3, 0]:  # 'd' or 'r' or Right arrow or Up arrow
                cap.release()
                print("Next Camera: ",(current_camera_id + 1) % total_cameras)
                current_camera_id = (current_camera_id + 1) % total_cameras
                cap = cv2.VideoCapture(current_camera_id)
            elif key in [ord("a"), ord("l"), 81, 84, 2, 1]:
                # 'a' or 'l' or Left arrow or Down arrow
                cap.release()
                print("Previous Camera: ",(current_camera_id - 1) % total_cameras)
                current_camera_id = (current_camera_id - 1) % total_cameras
                cap = cv2.VideoCapture(current_camera_id)

    release_all_cameras(total_cameras)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
