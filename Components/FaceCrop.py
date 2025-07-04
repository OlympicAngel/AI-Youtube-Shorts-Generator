import ffmpeg
global Fps


def crop_to_vertical_debug(input_video_path, output_video_path, debugView=False, fallback_crop_center=True):
    from Components.FaceDetection import detect_faces, Frames
    import cv2
    import numpy as np
    
    print("Detecting faces..")
    
    detect_faces(input_video_path)

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

    import subprocess
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{vertical_width}x{original_height}',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-vcodec', 'h264_nvenc',
        '-preset', 'fast',
        '-rc', 'vbr',
        '-cq', '24',
        '-b:v', '0',
        "-hide_banner",
        output_video_path
    ]
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

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
    alpha_center = 0.005  # smoothing factor center
    alpha_face = 0.55  # smoothing factor for face detection
    prev_frame = None

    print("Cropping vertical region of interest (face / moving objects)...")

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        face_box = Frames[i] if i < len(Frames) and Frames[i] else None
        detectType, result = get_focus_center(prev_frame, frame, face_box)
        temporalAlpha = alpha_face if detectType == 0 else alpha_center
        centerX = result[0] if isinstance(result, (list, tuple)) else result
        centerX = round(temporalAlpha * centerX + (1 - temporalAlpha) * last_centerX)
        last_centerX = centerX

        prev_frame = frame

        # Debug draw
        if debugView:
            debug_frame = frame.copy()
            cv2.line(debug_frame, (centerX, 0), (centerX, original_height), (0, 255, 0), 2)
            if face_box:
                x, y, x1, y1 = face_box
                cv2.rectangle(debug_frame, (x, y), (x1, y1), (255, 0, 0), 2)
            cv2.imshow("crop preview", debug_frame) #make sure it showing

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
            ffmpeg_process.stdin.write(cropped.tobytes())
            # out.write(cropped)
        except Exception as e:
            print(f"❌ Error writing frame {i}: {e}")
            continue

    cap.release()
    cv2.destroyAllWindows()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

def combine_videos(video_with_audio: str, video_without_audio: str, output_filename: str):
    print("Generating final video...")
    try:
        (
            ffmpeg
            .input(video_without_audio)
            .video
            .output(
                ffmpeg.input(video_with_audio).audio,
                output_filename,
                vcodec='copy',  # try to copy video
                acodec='copy',  # try to copy audio
                loglevel='error'
            )
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print("Copy failed, trying with GPU encode...")
        try:
            (
                ffmpeg
                .input(video_without_audio)
                .video
                .output(
                    ffmpeg.input(video_with_audio).audio,
                    output_filename,
                    vcodec='h264_nvenc',  # GPU encoding
                    acodec='aac',         # re-encode audio if needed
                    preset='medium',
                    loglevel='error'
                )
                .run(overwrite_output=True)
            )
        except ffmpeg.Error as e2:
            print(f"Error combining video and audio:\n{e2.stderr.decode()}")
