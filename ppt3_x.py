import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import time

# Load the video file and model

# Set one: momentum = 100
video_path = './test.mp4'
output_path = './output.mp4'
model = YOLO('./r.pt').to('cuda')


# Set one: momentum = 50

# video_path = './yyy.mp4'
# output_path = './output_yyy.mp4'
# model = YOLO('./r.pt').to('cuda')


# Initialize video writer
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 3, (frame_width, frame_height)) # 3 is the fps

# Parameters
momentum = 100

tracked_fish = []
fish_id = 0
counted_id = 0
counted_fish = 0

deleted_fish = []

# Logs
x_changes , y_changes, temp_score_list = [], [], []

 
def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def detect_fish_in_frame(frame):
    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

    return_list = []
    results = model(frame, verbose=False)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        return_list.append((x1, y1, x2 - x1, y2 - y1))

    # 去重
    filtered_list = []
    for i, box1 in enumerate(return_list):
        keep = True
        for j, box2 in enumerate(return_list):
            if i != j and compute_iou(box1, box2) > 0.5:  # IoU 阈值为 0.5
                keep = False
                break
        if keep:
            filtered_list.append(box1)

    return filtered_list

def track_fish(detections, tracked_fish):
    global fish_id
    global counted_fish
    global counted_id
    global x_changes , y_changes, temp_score_list
    c_show = 4
    c_lost = 3
    new_tracked_fish = []

    distance_matrix = []
    
    if len(detections) < 2:
        for detection in detections:
            center = get_center(*detection)
            distances = []
            for fish in tracked_fish:
                fish_center = fish['center']
                x_distance = center[0] - fish_center[0]
                y_distance = center[1] - fish_center[1]

                if y_distance < 20:
                    rank_distance = np.inf
                elif abs(x_distance) < 500 and abs(y_distance - (fish['lost'] + 1) * momentum) < 800:
                    rank_distance = np.square(x_distance) + np.square(y_distance - (fish['lost']+1) * momentum)
                else:
                    rank_distance = np.inf
                distances.append(rank_distance)

            if distances:
                min_distance = min(distances)
                min_index = distances.index(min_distance)
                if min_distance != np.inf:
                    fish = tracked_fish[min_index]
                    x_changes.append(center[0] - fish['center'][0])
                    y_changes.append(center[1] - fish['center'][1])
                    temp_score_list.append(min_distance)
                    fish['center'] = center
                    fish['bbox'] = detection
                    fish['lost'] = 0
                    fish['show_times'] += 1
                    if fish['show_times'] == c_show and fish['name'] == '_':
                        counted_fish += 1
                        fish['name'] = counted_id
                        counted_id += 1
                    new_tracked_fish.append(fish)
                else:
                    new_fish = {'id': str(fish_id)+ 'e', 'center': center, 'bbox': detection, 'lost': 0, 'show_times': 0, 'name': '_'}
                    new_tracked_fish.append(new_fish)
                    fish_id += 1
            else:
                new_fish = {'id': str(fish_id)+ 'f', 'center': center, 'bbox': detection, 'lost': 0, 'show_times': 0, 'name': '_'}
                new_tracked_fish.append(new_fish)
                fish_id += 1

        for fish in tracked_fish:
            if fish not in new_tracked_fish:
                fish['lost'] += 1
                if fish['lost'] < c_lost:
                    new_tracked_fish.append(fish)

        return new_tracked_fish

    elif len(tracked_fish) < 2:
        for fish in tracked_fish:
            fish_center = fish['center']
            distances = []
            for detection in detections:
                center = get_center(*detection)
                x_distance = center[0] - fish_center[0]
                y_distance = center[1] - fish_center[1]

                if y_distance < 20:
                    rank_distance = np.inf
                elif abs(x_distance) < 500 and abs(y_distance - (fish['lost'] + 1) * momentum) < 800:
                    rank_distance = np.square(x_distance) + np.square(y_distance - (fish['lost']+1) * momentum)
                else:
                    rank_distance = np.inf
                distances.append(rank_distance)

            if distances:
                min_distance = min(distances)
                min_index = distances.index(min_distance)
                if min_distance != np.inf:
                    detection = detections[min_index]
                    center = get_center(*detection)
                    x_changes.append(center[0] - fish['center'][0])
                    y_changes.append(center[1] - fish['center'][1])
                    temp_score_list.append(min_distance)
                    fish['center'] = center
                    fish['bbox'] = detection
                    fish['lost'] = 0
                    fish['show_times'] += 1
                    if fish['show_times'] == c_show and fish['name'] == '_':
                        counted_fish += 1
                        fish['name'] = counted_id
                        counted_id += 1
                    new_tracked_fish.append(fish)
                else:
                    new_fish = {'id': (str(fish_id)+ 'a'), 'center': fish_center, 'bbox': None, 'lost': fish['lost'] + 1, 'show_times': fish['show_times'], 'name': fish['name']}
                    new_tracked_fish.append(new_fish)
            else:
                new_fish = {'id': (str(fish_id)+ 'b'), 'center': fish_center, 'bbox': None, 'lost': fish['lost'] + 1, 'show_times': fish['show_times'], 'name': fish['name']}
                new_tracked_fish.append(new_fish)

        for detection in detections:
            if detection not in [fish['bbox'] for fish in new_tracked_fish]:
                center = get_center(*detection)
                new_fish = {'id': (str(fish_id)+ 'c'), 'center': center, 'bbox': detection, 'lost': 0, 'show_times': 0, 'name': '_'}
                new_tracked_fish.append(new_fish)
                fish_id += 1

        return new_tracked_fish

    else:
        for detection in detections:
            detection_center = get_center(*detection)
            distances = []
            for fish in tracked_fish:
                fish_center = fish['center']

                x_distance = detection_center[0] - fish_center[0]
                y_distance = detection_center[1] - fish_center[1]

                if y_distance < 0:
                    rank_distance = 1e10
                elif abs(x_distance) < 300 and abs(y_distance - (fish['lost'] + 1) * momentum) < 300:
                    rank_distance =  np.square(x_distance)  + np.square(y_distance - (fish['lost']+1) * momentum) 
                else:
                    rank_distance = 1e7

                distances.append(rank_distance)
            distance_matrix.append(distances)



        distance_matrix = np.array(distance_matrix)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        matched_detections = set()
        matched_fish = set()

        for i, j in zip(row_ind, col_ind):
            if distance_matrix[i, j] <= 1e6:
                detection = detections[i]
                fish = tracked_fish[j]

                center = get_center(*detection)
                x_changes.append(center[0] - fish['center'][0])
                y_changes.append(center[1] - fish['center'][1])
                temp_score_list.append(distance_matrix[i, j])

                fish['center'] = get_center(*detection)
                fish['bbox'] = detection
                fish['lost'] = 0
                fish['show_times'] = fish['show_times'] + 1
                if fish['show_times'] == c_show and fish['name'] == '_':
                    counted_fish += 1
                    fish['name'] = counted_id
                    counted_id += 1
                new_tracked_fish.append(fish)
                matched_detections.add(i)
                matched_fish.add(j)

        # if not matched, create new Fish
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                center = get_center(*detection)

                new_fish = {'id': str(fish_id)+ 'd', 'center': center, 'bbox': detection, 'lost': 0, 'show_times': 0, 'name': '_'}
                new_tracked_fish.append(new_fish)

                fish_id += 1
        # Delete Unmatched fishes
        for j, fish in enumerate(tracked_fish):
            if j not in matched_fish:
                fish['lost'] += 1
                fish['show_times'] = 0
                if fish['lost'] < c_lost:                    # JI Log
                    new_tracked_fish.append(fish)
        
        return new_tracked_fish
start_time = time.time()


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    detections = detect_fish_in_frame(frame)
    
    detection_count = len(detections)
    tracked_fish_count = len(tracked_fish)
    cv2.putText(frame, f'Detections: {detection_count}', (frame_width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f'Tracked Fish: {tracked_fish_count}', (frame_width - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    tracked_fish = track_fish(detections, tracked_fish)
    for detection in detections:
        x, y, w, h = detection
        cv2.rectangle(frame, (x-5 , y ), (x-5 + w, y + h), (0, 255, 0),2)
        cv2.putText(frame, 'D', (x-5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    for fish in tracked_fish:
        # if fish['name'] != '_':
        #     x, y, w, h = fish['bbox']
        #     cv2.rectangle(frame, (x , y ), (x + w, y + h), (255, 0, 0), 2)
        #     cv2.putText(frame, f'{x},{y},{w},{h}, name:{fish["name"]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # else:
        #     x, y, w, h = fish['bbox']
        #     cv2.rectangle(frame, (x , y ), (x + w, y + h), (0, 0, 80), 2)
        #     cv2.putText(frame, f'{x},{y},{w},{h}', (x, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if fish['name'] != '_':
            x, y, w, h = fish['bbox']
            cv2.rectangle(frame, (x , y ), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'{fish["name"]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
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

plt.savefig('./fig.png')
print("MEAN of X Change: ", np.mean(x_changes))
print("MEAN of Y Change: ", np.mean(y_changes))
# 打印 X Y Change 数据的中位数
print("MEDIAN of x Change: ", np.median(x_changes))
print("MEDIAN of Y Change: ", np.median(y_changes))

cap.release()
out.release()
print(f"Total fish counted: {counted_fish}, {fish_id}")


