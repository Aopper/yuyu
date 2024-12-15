import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time

        
# Load the video file and model
video_path = '/home/aopp/projects/Q/test.mp4'
output_path = '/home/aopp/projects/Q/test_origin.mp4'
model = YOLO('/home/aopp/Q/yy/runs/detect/train3/weights/best.pt').to('cuda')  # Use raw string for path



# Initialize video writer
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 3, (frame_width, frame_height))

# Parameters
max_x_distance = 100
max_y_distance = 100
momentum = 100

entry_line_position = 100
exit_line_position = 300

tracked_fish = []
fish_id = 0
counted_id = 0
counted_fish = 0

deleted_fish = []

def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def detect_fish_in_frame(frame):
    return_list = []
    results = model(frame)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        return_list.append((x1, y1, x2-x1, y2-y1))
    return return_list


def track_fish(detections, tracked_fish):
    global fish_id
    global counted_fish
    global counted_id
    new_tracked_fish = []
    for detection in detections:
        center = get_center(*detection)
        matched = False
        for fish in tracked_fish:
            # previous_center = fish['center']
            f_center = fish['center']
            f_center = (f_center[0], f_center[1] + momentum)
            x_distance, y_distance = np.linalg.norm(center[0] - f_center[0]), np.linalg.norm(center[1] - f_center[1])
            if (x_distance < max_x_distance) and (y_distance < max_y_distance):
                fish['center'] = center
                fish['bbox'] = detection
                fish['lost'] = 0
                fish['show_times'] = fish['show_times'] + 1
                if fish['show_times'] == 2:
                    counted_fish+=1
                    fish['name'] = counted_id
                    counted_id += 1
                x_center_change = center[0] - f_center[0]
                y_center_change = center[1] - f_center[1]

                if 'center_changes' not in fish:
                    fish['center_changes'] = []
                    fish['center_changes'].append((x_center_change, y_center_change))
                else:
                    fish['center_changes'].append((x_center_change, y_center_change))
                new_tracked_fish.append(fish)
                matched = True
                break
        if not matched:
            new_fish = {'id': fish_id, 'center': center, 'bbox': detection, 'lost': 0, 'center_changes': [], 'show_times': 0, 'name': '_'}
            new_tracked_fish.append(new_fish)
            fish_id += 1
    for fish in tracked_fish:
        if fish not in new_tracked_fish:
            fish['lost'] += 1

            # Drop time
            if fish['lost'] < 2:
                new_tracked_fish.append(fish)
            else:
                deleted_fish.append(fish)
    return new_tracked_fish


def count_fish(tracked_fish, line_position):
    global counted_fish
    for fish in tracked_fish:
        x, y, w, h = fish['bbox']
        center_y = y + h // 2
        if center_y > line_position and not fish.get('counted', False):
            counted_fish += 1
            fish['counted'] = True

start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    detections = detect_fish_in_frame(frame)
    tracked_fish = track_fish(detections, tracked_fish)
    # tracked_fish = track_fish(detections, tracked_fish, max_x_distance, max_y_distance)
    # count_fish(tracked_fish, entry_line_position)
    # for fish in tracked_fish:
    #     x, y, w, h = fish['bbox']
    #     cv2.rectangle(frame, (x , y ), (x + w, y + h), (255, 0, 0), 2)
    #     cv2.putText(frame, f'{fish["name"]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # for detection in detections:
    #     x, y, w, h = detection
    #     cv2.rectangle(frame, (x , y ), (x + w, y + h), (0, 0, 255), 2)
    #     cv2.putText(frame, '_', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    out.write(frame)

print("Time: ", time.time()-start_time)
# print("JI HNAG", len(deleted_fish))
# avg_y_cahnges = []
# for fish in deleted_fish:
#     x_changes = [change[0] for change in fish['center_changes']]
#     y_changes = [change[1] for change in fish['center_changes']]
#     avg_y_cahnges.append(np.mean(y_changes))


# plt.figure(figsize=(10, 6))
# plt.plot(range(len(avg_y_cahnges)), avg_y_cahnges, label='Average Y Change')

# plt.xlabel('Fish ID')
# plt.ylabel('Average Change')
# plt.title('Average X and Y Changes for Each Fish')
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

cap.release()
out.release()
print(f"Total fish counted: {counted_fish}, {fish_id}")


