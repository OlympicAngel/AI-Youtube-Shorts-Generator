from ctypes import Array
import cv2
from ultralytics import YOLO

# Load YOLO model (runs on GPU if available)
yolo_model = YOLO("yolov5nu.pt")  # or yolov5n.pt for faster
yolo_model.to('cuda') 

temp_audio_path = "temp_audio.wav"

def extract_audio_from_video(video_path, audio_path):
    from pydub import AudioSegment
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(audio_path, format="wav")

def process_audio_frame(audio_data, sample_rate=16000, frame_duration_ms=30):
    n = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
    offset = 0
    while offset + n <= len(audio_data):
        frame = audio_data[offset:offset + n]
        offset += n
        yield frame

global Frames
Frames = [] # [x,y,w,h]
Frames: list[Array[int,int,int,int]]

def detect_faces(input_video_path,batch_size=32):
    import os
    import tempfile
    import wave
    import contextlib

    
    # Return Frams:
    global Frames
    Frames = []

    temp_audio_path = tempfile.mktemp(suffix=".wav")

    # Extract audio
    extract_audio_from_video(input_video_path, temp_audio_path)

    # Load audio data
    with contextlib.closing(wave.open(temp_audio_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    # Prepare video
    cap = cv2.VideoCapture(input_video_path)

    audio_generator = process_audio_frame(audio_data, sample_rate, 30)
    
    frame_buffer = []
    
    index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        audio_frame = next(audio_generator, None)
        if audio_frame is None:
            break
        
        frame_buffer.append(frame)
        index += 1

        if len(frame_buffer) == batch_size:
            process_batch(frame_buffer)
            frame_buffer.clear()
            
    if frame_buffer:
        process_batch(frame_buffer)
        
    cap.release()
    cv2.destroyAllWindows()
    os.remove(temp_audio_path)
    
    Frames = smooth_boxes(Frames)

def process_batch(frames):
    global Frames
    resized_batch = [cv2.resize(f, None, fx=0.25, fy=0.25)[..., ::-1] for f in frames]  # BGR → RGB
    results = yolo_model(resized_batch, verbose=False)

    for frame, result in zip(frames, results):
        boxes = []
        for box in result.boxes:
            if int(box.cls.item()) == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w = x2 - x1
                h = y2 - y1
                boxes.append((x1, y1, w, h))

        # Face selection logic
        candidates = []
        max_lip_distance = 25
        for (x, y, w_face, h_face) in boxes:
            x1, y1 = x + w_face, y + h_face
            lip_distance = abs((y + 2 * h_face // 3) - y1)
            candidates.append(((x, y, x1, y1), lip_distance))
            max_lip_distance = max(max_lip_distance, lip_distance)

        for (x, y, x1, y1), lip_distance in candidates:
            if lip_distance >= max_lip_distance:
                Frames.append([x, y, x1, y1])
                break
        else:
            Frames.append(None)

def smooth_boxes(frames, alpha=0.4):
    smoothed = []
    prev_box = None

    for box in frames:
        if box is None:
            smoothed.append(None)
        else:
            if prev_box is None:
                smoothed_box = box
            else:
                smoothed_box = [
                    alpha * b + (1 - alpha) * pb
                    for b, pb in zip(box, prev_box)
                ]
            smoothed.append(smoothed_box)
            prev_box = smoothed_box

    return smoothed

def detect_faces_yolo(frame):
    resized = cv2.resize(frame, None, fx=0.25, fy=0.25)
    results  = yolo_model(resized[..., ::-1], verbose=False)[0]
    face_boxes = []
    for box in results.boxes:
        cls = int(box.cls.item())
        if cls == 0:  # person class
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w = x2 - x1
            h = y2 - y1
            face_boxes.append((x1, y1, w, h))
    return face_boxes