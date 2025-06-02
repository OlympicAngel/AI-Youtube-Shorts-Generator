from typing import List
import cv2
import numpy as np
from moviepy import *
from moviepy.video.io.VideoFileClip import VideoFileClip  # ✅ precise import
from Components.LanguageTasks import ClipSegment
from Components.Speaker import detect_faces_and_speakers, Frames
global Fps


def crop_to_vertical_debug(input_video_path, output_video_path, debugView=False, fallback_crop_center=True):
    detect_faces_and_speakers(input_video_path, "DecOut.mp4")

    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    global Fps
    Fps = fps

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vertical_width = int(original_height * 9 / 16)

    out = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (vertical_width, original_height)
        )

    def get_focus_center(prev_frame,frame, face_box=None):
        if face_box:
            x, y, x1, y1 = face_box
            return [0,(x + x1) // 2]
        if fallback_crop_center:
            return [1,detect_salient_motion_region(prev_frame,frame)]
        return frame.shape[1] // 2

    def detect_salient_motion_region(prev_frame, curr_frame):
        height, width = curr_frame.shape[:2]
        cx, cy = width // 2, height // 2

        # Define allowed bounds (±30% of frame size around center)
        x_min, x_max = int(cx - width * 0.3), int(cx + width * 0.3)
        y_min, y_max = int(cy - height * 0.3), int(cy + height * 0.3)

        if prev_frame is None:
            return cx, cy

        # Crop UI
        y_start, y_end = int(height * 0.15), int(height * 0.85)
        prev_crop = prev_frame[y_start:y_end, :]
        curr_crop = curr_frame[y_start:y_end, :]

        prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)

        prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        diff = cv2.absdiff(prev_blur, curr_blur)
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        moments = cv2.moments(motion_mask)
        if moments["m00"] != 0:
            x_center = int(moments["m10"] / moments["m00"])
            y_center = int(moments["m01"] / moments["m00"]) + y_start

            if x_min <= x_center <= x_max and y_min <= y_center <= y_max:
                return x_center, y_center

        return cx, cy

    last_centerX = original_width // 2
    alpha_center = 0.01  # smoothing factor center
    alpha_face = 0.3  # smoothing factor for face detection
    prev_frame = None

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        face_box = Frames[i] if i < len(Frames) and Frames[i] else None
        detectType, result = get_focus_center(prev_frame, frame, face_box)
        temporalAlpha = alpha_face if detectType == 0 else alpha_center
        centerX = result[0] if isinstance(result, (list, tuple)) else result
        centerX = int(temporalAlpha * centerX + (1 - temporalAlpha) * last_centerX)
        last_centerX = centerX

        prev_frame = frame

        # Debug draw
        if debugView:
            debug_frame = frame.copy()
            cv2.line(debug_frame, (centerX, 0), (centerX, original_height), (0, 255, 0), 2)
            if face_box:
                x, y, x1, y1 = face_box
                cv2.rectangle(debug_frame, (x, y), (x1, y1), (255, 0, 0), 2)

        x_start = max(0, centerX - vertical_width // 2)
        x_end = x_start + vertical_width
        if x_end > original_width:
            x_end = original_width
            x_start = x_end - vertical_width

        cropping_base = debug_frame if debugView else frame
        cropped = cropping_base[:, x_start:x_end]

        # Fallback to center crop if the current crop is invalid
        if cropped is None or cropped.shape[1] != vertical_width or cropped.shape[0] != original_height:
            print(f"⚠️ Invalid crop at frame {i}, falling back to center crop.")
            centerX = original_width // 2
            x_start = max(0, centerX - vertical_width // 2)
            x_end = x_start + vertical_width
            if x_end > original_width:
                x_end = original_width
                x_start = x_end - vertical_width
            cropped = cropping_base[:, x_start:x_end]

        if cropped is None or not isinstance(cropped, np.ndarray):
            print(f"❌ Skipped frame {i}: Cropped is None or not ndarray")
            continue

        if cropped.shape[0] != original_height or cropped.shape[1] != vertical_width:
            print(f"❌ Skipped frame {i}: Unexpected shape {cropped.shape}, expected ({original_height}, {vertical_width})")
            continue

        if cropped.dtype != np.uint8:
            print(f"❌ Skipped frame {i}: Invalid dtype {cropped.dtype}, expected uint8")
            continue

        try:
            out.write(cropped)
        except Exception as e:
            print(f"❌ Error writing frame {i}: {e}")
            continue

    cap.release()
    out.release()

def combine_videos(video_with_audio:str, video_without_audio:str, output_filename:str):
    try:
        # Load video clips
        clip_with_audio = VideoFileClip(video_with_audio)
        clip_without_audio = VideoFileClip(video_without_audio)

        audio = clip_with_audio.audio
        combined_clip:VideoFileClip = clip_without_audio.with_audio(audio)

        global Fps
        combined_clip.write_videofile(output_filename, codec='h264_nvenc', audio_codec='aac', fps=Fps, preset='medium', bitrate='3000k')
        print(f"Combined video saved successfully as {output_filename}")
    
    except Exception as e:
        print(f"Error combining video and audio: {str(e)}")



if __name__ == "__main__":
    input_video_path = r'Out.mp4'
    output_video_path = 'Croped_output_video.mp4'
    final_video_path = 'final_video_with_audio.mp4'
    detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    crop_to_vertical_debug(input_video_path, output_video_path)
    combine_videos(input_video_path, output_video_path, final_video_path)



