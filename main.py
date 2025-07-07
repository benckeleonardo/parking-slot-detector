import cv2
import numpy as np


def load_resources():
    empty_frame = cv2.imread('empty_frame.png')
    cap = cv2.VideoCapture('assets/Estacionamento.mp4')
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    parking_slots = [
        [(-115, 460), (150, 460), (220, 160), (70, 160)],
        [(170, 460), (460, 460), (415, 160), (230, 160)],
        [(480, 460), (830, 460), (590, 160), (430, 160)],
        [(850, 460), (1100, 460), (740, 160), (600, 160)],
    ]

    parking_slots_np = [np.array(slot, dtype=np.int32) for slot in parking_slots]

    return empty_frame, cap, parking_slots_np


def preprocess_frame(frame_roi, empty_roi):
    gray_frame = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(empty_roi, gray_frame)
    _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    return thresh


def detect_contours(thresh, min_area=17000):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filtered = [c for c in contours if cv2.contourArea(c) > min_area]
    return filtered


def draw_car_rectangles(frame, contours, roi_offset):
    x_offset, y_offset = roi_offset
    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        x += x_offset
        y += y_offset
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2

        text = "Car"
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        text_x = x
        text_y = y - 10

        if text_y < text_size[1]:
            text_y = y + h + text_size[1] + 10

        cv2.putText(frame, text, (text_x, text_y), font, scale, (255, 0, 0), thickness, cv2.LINE_AA)


def calculate_occupied_area(contours, roi_offset, frame_shape):
    x_offset, y_offset = roi_offset
    area = np.zeros(frame_shape[:2], dtype=np.uint8)
    for contour in contours:
        contour_shifted = contour + [x_offset, y_offset]
        cv2.drawContours(area, [contour_shifted], -1, 255, -1)
    return area


def draw_parking_slots(frame, parking_slots, occupied_area):
    free_slots = 0
    for slot_pts in parking_slots:
        slot_area = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(slot_area, [slot_pts], 255)

        intersection = cv2.bitwise_and(occupied_area, slot_area)
        intersection_area = cv2.countNonZero(intersection)
        slot_area = cv2.countNonZero(slot_area)
        occupancy_ratio = intersection_area / slot_area

        if occupancy_ratio > 0.3:
            color = (0, 0, 255)  # Red: occupied
        else:
            color = (0, 255, 0)  # Green: free
            free_slots += 1

        cv2.polylines(frame, [slot_pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    return free_slots


def draw_free_slots_text(frame, free_slots):
    text = f"Parking slots available: {free_slots}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2

    cv2.putText(frame, text, (20, 40), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (20, 40), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    empty_frame, cap, parking_slots = load_resources()

    y1, y2 = 124, 460
    x1, x2 = 0, 848

    gray_empty_full = cv2.cvtColor(empty_frame, cv2.COLOR_BGR2GRAY)
    gray_empty_roi = gray_empty_full[y1:y2, x1:x2]

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Could not read the frame. Exiting...")
                break

            frame_roi = frame[y1:y2, x1:x2]
            thresh = preprocess_frame(frame_roi, gray_empty_roi)
            contours = detect_contours(thresh)
            draw_car_rectangles(frame, contours, (x1, y1))
            occupied_area = calculate_occupied_area(contours, (x1, y1), frame.shape)
            free_slots = draw_parking_slots(frame, parking_slots, occupied_area)
            draw_free_slots_text(frame, free_slots)

            cv2.imshow('Frame', frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()