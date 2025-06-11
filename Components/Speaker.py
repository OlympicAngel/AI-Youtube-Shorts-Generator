from ctypes import Array
import cv2

temp_audio_path = "temp_audio.wav"

def voice_activity_detection(audio_frame, sample_rate=16000):
    import webrtcvad

    # Initialize VAD
    vad = webrtcvad.Vad(2)  # Aggressiveness mode from 0 to 3

    return vad.is_speech(audio_frame, sample_rate)

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

def detect_faces_and_speakers(input_video_path, output_video_path):
    import os
    import tempfile
    import wave
    import contextlib

    
    # Return Frams:
    global Frames
    Frames = []

    temp_audio_path = tempfile.mktemp(suffix=".wav")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Extract audio
    extract_audio_from_video(input_video_path, temp_audio_path)

    # Load audio data
    with contextlib.closing(wave.open(temp_audio_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    # Prepare video
    cap = cv2.VideoCapture(input_video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))

    audio_generator = process_audio_frame(audio_data, sample_rate, 30)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(60, 60))

        audio_frame = next(audio_generator, None)
        if audio_frame is None:
            break
        is_speaking = voice_activity_detection(audio_frame, sample_rate)

        candidates = []
        max_lip_distance = 25

        for (x, y, w_face, h_face) in faces:
            x1, y1 = x + w_face, y + h_face
            lip_distance = abs((y + 2 * h_face // 3) - y1)
            candidates.append(((x, y, x1, y1), lip_distance))
            max_lip_distance = max(max_lip_distance, lip_distance)

        for (x, y, x1, y1), lip_distance in candidates:
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            if lip_distance >= max_lip_distance:
                if is_speaking:
                    cv2.putText(frame, "Active Speaker", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                Frames.append([x, y, x1, y1])
                break
            else:
                Frames.append(None)

        out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    os.remove(temp_audio_path)
