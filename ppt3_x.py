import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

import time
import logging
logging.getLogger().setLevel(logging.WARNING)
from scipy.optimize import linear_sum_assignment


# Load the video file and model
video_path = './50fish_60fps_Qpc.avi'
output_path = './test_x.mp4'
# video_path = '/home/aopp/projects/Q/yuyuyu.mp4'
# output_path = '/home/aopp/projects/Q/yuyuyu_nms.mp4'
model = YOLO('./best.pt').to('cuda:1')  # Use raw string for path



# Initialize video writer
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (frame_width, frame_height))

# Parameters
max_y_distance = 120
momentum = 100

tracked_fish = []
fish_id = 0
counted_id = 1
counted_fish = 0

deleted_fish = []

x_changes , y_changes = [], []


def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def detect_fish_in_frame(frame):
    return_list = []
    results = model(frame, verbose=False)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        return_list.append((x1, y1, x2-x1, y2-y1))
    return return_list


# def dual_line(x, change_point):
#     if abs(x) < abs(change_point):
#         return x
#     else:
#         return change_point + 10 * (x - change_point)
    
def track_fish(detections, tracked_fish, max_y_distance):
    global fish_id
    global counted_fish
    global counted_id
    global x_changes, y_changes
    if len(detections) == 0:
        print(123)
        return tracked_fish
    new_tracked_fish = []

    num_detections = len(detections)
    num_tracked_fish = len(tracked_fish)

    # Initialize the distance matrix
    distance_matrix = np.full((num_detections, num_tracked_fish), 65535)

    for i, detection in enumerate(detections):
        detection_center = get_center(*detection)
        for j, fish in enumerate(tracked_fish):
            fish_center = fish['center']
            x_distance = np.abs(detection_center[0] - fish_center[0])
            y_distance = detection_center[1] - fish_center[1] - (fish['lost'] + 1) * momentum
            rank_distance = np.square(x_distance) + np.abs(y_distance)

            if x_distance < 100 and np.abs(y_distance) < max_y_distance:
                distance_matrix[i, j] = rank_distance

    # Make the matrix square by padding with np.inf
    max_dim = max(num_detections, num_tracked_fish)
    padded_matrix = np.full((max_dim, max_dim), np.inf)
    padded_matrix[:num_detections, :num_tracked_fish] = distance_matrix

    # Apply Hungarian algorithm
    print("Distance Matrix:")
    print(padded_matrix.shape)
    row_indices, col_indices = linear_sum_assignment(padded_matrix)

    matched_detections = set()
    matched_fish = set()

    # Process matches
    for row, col in zip(row_indices, col_indices):
        # Ignore padded values
        if row >= num_detections or col >= num_tracked_fish:
            continue

        if padded_matrix[row, col] == np.inf:  # Ignore invalid matches
            continue

        detection = detections[row]
        fish = tracked_fish[col]

        fish['center'] = get_center(*detection)
        fish['bbox'] = detection
        fish['lost'] = 0
        fish['show_times'] += 1

        if fish['show_times'] == 5:
            counted_fish += 1
            fish['name'] = counted_id
            counted_id += 1

        new_tracked_fish.append(fish)
        matched_detections.add(row)
        matched_fish.add(col)

    # Create new fish for unmatched detections
    for i, detection in enumerate(detections):
        if i not in matched_detections:
            center = get_center(*detection)
            new_fish = {
                'id': fish_id,
                'center': center,
                'bbox': detection,
                'lost': 0,
                'show_times': 0,
                'name': '_'
            }
            new_tracked_fish.append(new_fish)
            fish_id += 1

    # Handle unmatched tracked fish
    for j, fish in enumerate(tracked_fish):
        if j not in matched_fish:
            fish['lost'] += 1
            fish['show_times'] = 0
            if fish['lost'] < 2:
                new_tracked_fish.append(fish)

    return new_tracked_fish

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    detections = detect_fish_in_frame(frame)
    tracked_fish = track_fish(detections, tracked_fish, max_y_distance)


    for fish in tracked_fish:
        if fish['name'] != '_':
            x, y, w, h = fish['bbox']
            cv2.rectangle(frame, (x , y ), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'{fish["name"]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    out.write(frame)

print("TIME: {:.3f} s".format( time.time() - start_time))

cap.release()
out.release()
print(f"Total fish counted: {counted_fish}, {fish_id}")


