import cv2
import numpy as np
import torch
from torchvision import models, transforms
from collections import deque
import imageio
import time

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pre-trained object detection model and move it to the GPU
try:
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
except Exception as e:
    print(f"Error loading model: {e}")
model.eval()

# Define a transformation to preprocess the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# Function to detect people in a frame
def detect_people(frame):
    image = transform(frame).to(device)  # Move the image tensor to the GPU
    image = image.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)

    scores = outputs[0]['scores'].cpu().numpy()  # Move the outputs back to the CPU
    boxes = outputs[0]['boxes'].cpu().numpy()  # Move the outputs back to the CPU
    labels = outputs[0]['labels'].cpu().numpy()  # Move the outputs back to the CPU

    people_boxes = []
    for i, label in enumerate(labels):
        if label == 1 and scores[i] > 0.5:  # Label 1 is for 'person' class
            people_boxes.append(boxes[i])
    return people_boxes

# Function to draw bounding boxes and counts
def draw_boxes_and_counts(frame, boxes, counts):
    for box in boxes:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

    cv2.putText(frame, f"Total: {counts['total']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"IN: {counts['in']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {counts['out']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Function to process the GIF and store the result in a video
def process_gif(gif_path, output_path, max_fps=10):
    # Initialize counts
    counts = {"total": 0, "in": 0, "out": 0}

    # Read the GIF
    try:
        gif = imageio.get_reader(gif_path)
    except Exception as e:
        print(f"Error reading GIF: {e}")
        return

    # Get GIF properties
    meta_data = gif.get_meta_data()
    fps = meta_data.get('fps', 10)  # Use 10 FPS as default if 'fps' key is not found
    delay = 1 / min(fps, max_fps)
    first_frame = gif.get_data(0)
    height, width, _ = first_frame.shape
    line_y = height // 2

    # Initialize video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(output_path, fourcc, min(fps, max_fps), (width, height))
    except Exception as e:
        print(f"Error initializing video writer: {e}")
        return

    # Trackers for people
    trackers = []
    person_id = 0
    person_ids = []
    person_states = {}

    for frame in gif:
        start_time = time.time()

        boxes = detect_people(frame)
        current_trackers = []
        current_ids = []

        for box in boxes:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2

            # Check if the person is already being tracked
            is_new_person = True
            for tracker, pid in zip(trackers, person_ids):
                if np.linalg.norm(np.array([x_center, y_center]) - np.array(tracker[-1])) < 50:
                    tracker.append((x_center, y_center))
                    current_trackers.append(tracker)
                    current_ids.append(pid)
                    is_new_person = False
                    break

            # If it's a new person, create a new tracker
            if is_new_person:
                current_trackers.append(deque([(x_center, y_center)], maxlen=50))
                current_ids.append(person_id)
                person_states[person_id] = "unknown"
                person_id += 1

        # Update trackers
        trackers = current_trackers
        person_ids = current_ids

        # Check for entries and exits
        for tracker, pid in zip(trackers, person_ids):
            if len(tracker) > 1:
                # Check if the person has crossed the line
                if tracker[0][1] < line_y and tracker[-1][1] > line_y and person_states[pid] != "in":
                    counts['in'] += 1
                    counts['total'] += 1
                    person_states[pid] = "in"
                    print(f"Person {pid} entered")
                elif tracker[0][1] > line_y and tracker[-1][1] < line_y and person_states[pid] != "out":
                    counts['out'] += 1
                    counts['total'] -= 1
                    person_states[pid] = "out"
                    print(f"Person {pid} exited")

        # Draw the line
        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)

        # Draw boxes and counts on the frame
        draw_boxes_and_counts(frame, boxes, counts)

        # Write the frame to the output video
        out.write(frame)

        # Ensure processing respects the FPS limit
        elapsed_time = time.time() - start_time
        if elapsed_time < delay:
            time.sleep(delay - elapsed_time)

    out.release()

# Process the uploaded GIF and store the result in a video
process_gif('/content/Untitled video - Made with Clipchamp.mp4', '/content/output_video.mp4', max_fps=10)

