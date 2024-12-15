import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

import time
        
# Load the video file and model
video_path = r'G:\CVBasedFishRecognition\Q\50fish_60fps_Qpc.avi'
output_path = r'G:\CVBasedFishRecognition\Q\50fish_60fps_Qpc_out.mp4'
model = YOLO(r'G:\CVBasedFishRecognition\Q\bestfake.pt').to('cuda')  # Use raw string for path

model.conf = 0.5
model.iou = 0.4

# Initialize video writer
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 3, (frame_width, frame_height))

# Parameters
max_y_distance = 100
momentum = 85

tracked_fish = []
fish_id = 0
counted_id = 0
counted_fish = 0

deleted_fish = []

x_changes , y_changes, temp_score_list = [], [], []

 
def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def detect_fish_in_frame(frame):
    return_list = []
    results = model(frame)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        return_list.append((x1, y1, x2-x1, y2-y1))
    return return_list


def track_fish(detections, tracked_fish, max_y_distance):
    global fish_id
    global counted_fish
    global counted_id
    global x_changes , y_changes, temp_score_list
    new_tracked_fish = []

    distance_matrix = []
    
    # Calculate distance matrix
    for detection in detections:
        detection_center = get_center(*detection)
        distances = []
        for fish in tracked_fish:
            fish_center = fish['center']

            x_distance = detection_center[0] - fish_center[0]
            y_distance = detection_center[1] - fish_center[1]

            if abs(x_distance) < 100 and abs(y_distance - (fish['lost'] + 1) * momentum) < 300 and y_distance > 0:
                rank_distance =  3 * np.square(x_distance)  + np.square(y_distance - (fish['lost']+1) * momentum) 
            else:
                rank_distance = np.inf

            temp_score = None

            distances.append((rank_distance, x_distance, y_distance, temp_score))
        distance_matrix.append(distances)

    matched_detections = set()
    matched_fish = set()

    # Rank by  rank_distance
    flat_distances = [
        (i, j, distance_matrix[i][j][0], distance_matrix[i][j][1], distance_matrix[i][j][2], distance_matrix[i][j][3])
        for i in range(len(detections))
        for j in range(len(tracked_fish))
    ]
    flat_distances.sort(key=lambda item: (item[2]))  # Sort by rank_distance
    # flat_distances.sort(key=lambda item: (item[5]))  # Sort by temp_score

    # Matching detection and fish

    ji_time = 0
    for i, j, rank_distance, x_distance, y_distance, temp_score in flat_distances:
        ji_time += 1
        if i in matched_detections or j in matched_fish:
            continue
        if rank_distance != np.inf:
        # if abs(x_distance) < 80 and np.abs(y_distance)<140:

            if fish['lost'] == 0:
                x_changes.append(x_distance)
                y_changes.append(y_distance)
                if rank_distance:
                    temp_score_list.append(rank_distance)

            detection = detections[i]
            fish = tracked_fish[j]

            fish['center'] = get_center(*detection)
            fish['bbox'] = detection
            fish['lost'] = 0
            fish['show_times'] = fish['show_times'] + 1
            if fish['show_times'] == 6:
                counted_fish += 1
                fish['name'] = counted_id
                counted_id += 1
            new_tracked_fish.append(fish)
            matched_detections.add(i)
            matched_fish.add(j)
        else:
            print("JI LOG::: Break at {} over {}".format(ji_time, i*j))
            break


    # if not matched, creat new Fish
    for i, detection in enumerate(detections):
        if i not in matched_detections:
            center = get_center(*detection)

            new_fish = {'id': fish_id, 'center': center, 'bbox': detection, 'lost': 0, 'show_times': 0, 'name': '_'}
            new_tracked_fish.append(new_fish)

            fish_id += 1

    # Delete Unmatched fishes
    for j, fish in enumerate(tracked_fish):
        if j not in matched_fish:
            fish['lost'] += 1
            fish['show_times'] = 0
            if fish['lost'] < 3:                    # JI Log
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
            cv2.putText(frame, f'{x},{y},{w},{h}, name:{fish["name"]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            x, y, w, h = fish['bbox']
            cv2.rectangle(frame, (x , y ), (x + w, y + h), (0, 0, 80), 2)
            cv2.putText(frame, f'{x},{y},{w},{h}', (x, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    out.write(frame)
print("TIME: ", time.time() - start_time)
x_changes.sort()
y_changes.sort()
temp_score_list.sort()
plt.figure(figsize=(10, 6))
plt.plot(range(len(x_changes)), x_changes, label='X Change')
plt.plot(range(len(y_changes)), y_changes, label='Y Change')
# plt.plot(range(len(temp_score_list)), temp_score_list, label='TempScore')

plt.ylabel('Average Change')
plt.title('Average X and Y Changes for Each Fish')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
# 设置纵坐标每 10 为一个刻度
y_min = min(min(x_changes), min(y_changes))
y_max = max(max(x_changes), max(y_changes))
plt.yticks(range(int(y_min), int(y_max) + 10, 10))

plt.savefig('./fig.png')
print("MEAN of X Change: ", np.mean(x_changes))
print("MEAN of Y Change: ", np.mean(y_changes))
# 打印 X Y Change 数据的中位数
print("MEDIAN of x Change: ", np.median(x_changes))
print("MEDIAN of Y Change: ", np.median(y_changes))

cap.release()
out.release()
print(f"Total fish counted: {counted_fish}, {fish_id}")


