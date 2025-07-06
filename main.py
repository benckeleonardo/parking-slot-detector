import cv2
import numpy as np


def main():
    empty_background = cv2.imread('empty_frame.png')
    cap = cv2.VideoCapture('assets/Estacionamento.mp4')
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    paused = False  # Initial state: not paused

    while True:
        if not paused:
            ret, frame = cap.read()

            if not ret:
                print("Could not read the frame. Exiting...")
                break

            gray_empty = cv2.cvtColor(empty_background, cv2.COLOR_BGR2GRAY)
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            difference = cv2.absdiff(gray_empty, gray_current)
            _, thresh = cv2.threshold(difference, 10, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                area = cv2.contourArea(contour)

                if area > 2000:  # Only draw if the area is greater than 2000 (adjust as needed)
                    cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

            # Draw parking slots (bottom left, bottom right, top right, top left)
            parking_slots = [
                [(-115, 470), (150, 470), (220, 170), (70, 180)],
                [(170, 470), (450, 470), (400, 160), (230, 170)],
                [(460, 460), (830, 460), (590, 150), (410, 160)],
                [(850, 460), (1100, 460), (740, 150), (600, 150)],
            ]

            for slot in parking_slots:
                pts = np.array([slot], dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow('Frame', frame)

        # Wait 30 ms for a key press and capture the key pressed
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):  # Key 'q' → exit
            break
        elif key == 32:  # Space key → toggle pause
            paused = not paused

    # Release the video and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()