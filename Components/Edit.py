from typing import List
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy import VideoFileClip, concatenate_videoclips
from Components.LanguageTasks import ClipSegment
import random
from typing import List, TypedDict, Optional, Callable, Tuple
from moviepy import (
    VideoFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
)
from moviepy.video.fx.HeadBlur import HeadBlur

class ClipSegment(TypedDict):
    start_time: float
    end_time: float
    content: str
    speakers: Optional[List[str]]


def crop_video(
    input_file: str,
    output_file: str,
    segments: List[ClipSegment],
    *,
    gap_threshold: float = 3.0,           # seconds: gap > 3s triggers transition
    transition_duration: float = 0.6,      # length of pan transition (seconds)
) -> None:
    """
    Cuts `input_file` into sub‐clips defined by `segments` (each with start_time, end_time).
    Whenever two consecutive segments have a gap > gap_threshold, and both segments are at least
    transition_duration long, inserts a 1s pan transition (random direction: top/bottom/left/right)
    between them. Uses MoviePy 2.2.1 “new” API (no .set_* calls, only .with_* and .subclipped).

    The pan transition moves Clip A out of frame while simultaneously moving Clip B into frame,
    both over transition_duration seconds, with cosine ease‐in‐out. During the transition,
    applies a constant HeadBlur (blur_x, blur_y) for visual flair.

    Ensures no frames are duplicated or skipped:
      – If prev segment length = LA, curr segment length = LB, and transition_duration = Td,
        then:
          • prev_main  = prev_full.subclipped(0, LA – Td)
          • prev_trans = prev_full.subclipped(LA – Td, LA)
          • curr_trans = curr_full.subclipped(0, Td)
          • curr_main  = curr_full.subclipped(Td, LB)

    Finally concatenates all pieces with method="compose" and writes to output_file.
    """

    def _ease(u: float) -> float:
        # Cosine ease-in-out on [0,1]
        return 2 * u * u if u < 0.5 else -1 + (4 - 2 * u) * u

    # Load full video once
    video: VideoFileClip = VideoFileClip(input_file)

    # Sort segments by start_time just in case
    segments_sorted: List[ClipSegment] = sorted(segments, key=lambda s: s["start_time"])

    final_clips: List[CompositeVideoClip] = []
    prev_segment: Optional[ClipSegment] = None
    prev_full_clip: Optional[VideoFileClip] = None

    for idx, seg in enumerate(segments_sorted):
        start_time: float = seg["start_time"]
        end_time: float = seg["end_time"]
        seg_length: float = end_time - start_time

        # Extract the full clip for this segment via new .subclipped(...)
        current_full: VideoFileClip = video.subclipped(start_time, end_time)

        if idx == 0:
            # First segment – hold until next iteration
            prev_segment = seg
            prev_full_clip = current_full
            continue

        # From here on, prev_segment and prev_full_clip are not None
        assert prev_segment is not None
        assert prev_full_clip is not None

        gap: float = seg["start_time"] - prev_segment["end_time"]

        # If gap <= threshold, no transition: append prev_full_clip now
        if gap <= gap_threshold:
            final_clips.append(prev_full_clip)
            prev_segment = seg
            prev_full_clip = current_full
            continue

        # Otherwise, gap > threshold.  Check if both clips >= transition_duration
        prev_len: float = prev_segment["end_time"] - prev_segment["start_time"]
        curr_len: float = seg["end_time"] - seg["start_time"]

        if prev_len >= transition_duration and curr_len >= transition_duration:
            # Split previous clip into main + trans
            tA_main_end: float = prev_len - transition_duration
            prev_main: VideoFileClip = prev_full_clip.subclipped(0.0, tA_main_end)
            prev_trans: VideoFileClip = prev_full_clip.subclipped(tA_main_end, prev_len)

            # Split current clip into trans + main
            curr_trans: VideoFileClip = current_full.subclipped(0.0, transition_duration)
            curr_main: VideoFileClip = current_full.subclipped(transition_duration, curr_len)

            # Choose random pan direction
            direction: str = random.choice(["left", "right", "top", "bottom"])
            W: int
            H: int
            W, H = video.size  # width, height of the original video

            # Define position functions for A (outgoing) and B (incoming)
            def make_positions(dir_str: str) -> Tuple[Callable[[float], Tuple[float, float]],
                                                     Callable[[float], Tuple[float, float]]]:
                if dir_str == "right":
                    def posA(t: float) -> Tuple[float, float]:
                        u: float = t / transition_duration
                        x: float = _ease(u) * W
                        return (x, 0.0)

                    def posB(t: float) -> Tuple[float, float]:
                        u: float = t / transition_duration
                        x: float = -W * (1.0 - _ease(u))
                        return (x, 0.0)

                elif dir_str == "left":
                    def posA(t: float) -> Tuple[float, float]:
                        u: float = t / transition_duration
                        x: float = -_ease(u) * W
                        return (x, 0.0)

                    def posB(t: float) -> Tuple[float, float]:
                        u: float = t / transition_duration
                        x: float = W * (1.0 - _ease(u))
                        return (x, 0.0)

                elif dir_str == "top":
                    def posA(t: float) -> Tuple[float, float]:
                        u: float = t / transition_duration
                        y: float = -_ease(u) * H
                        return (0.0, y)

                    def posB(t: float) -> Tuple[float, float]:
                        u: float = t / transition_duration
                        y: float = H * (1.0 - _ease(u))
                        return (0.0, y)

                else:  # "bottom"
                    def posA(t: float) -> Tuple[float, float]:
                        u: float = t / transition_duration
                        y: float = _ease(u) * H
                        return (0.0, y)

                    def posB(t: float) -> Tuple[float, float]:
                        u: float = t / transition_duration
                        y: float = -H * (1.0 - _ease(u))
                        return (0.0, y)

                return posA, posB

            posA_func, posB_func = make_positions(direction)

            # Ensure transition pieces have correct duration & fps using new .with_*
            prev_trans: VideoFileClip = (
                prev_trans
                .with_duration(transition_duration)
                .with_fps(video.fps)
            )
            curr_trans: VideoFileClip = (
                curr_trans
                .with_duration(transition_duration)
                .with_fps(video.fps)
            )

            # Apply position (new .with_position)
            movingA: VideoFileClip = prev_trans.with_position(posA_func)
            movingB: VideoFileClip = curr_trans.with_position(posB_func)

            def Blur(t: float) -> float:
                        u: float = t / transition_duration
                        y: float = _ease(u) * movingA.w
                        return y

            def UnBlur(t: float) -> float:
                u: float = 1 - (t / transition_duration)
                y: float = _ease(u) * movingA.w
                return y
            def NoBlur(t: float) -> float:
                return 0.0
            
            isHorizontal = direction in ["left", "right"]

            # TODO set direction for HeadBlur based on pan direction

            # Apply a constant HeadBlur for motion‐blur feel
            # movingA = movingA.with_effects([HeadBlur( Blur if isHorizontal else NoBlur,
            #                                          NoBlur if isHorizontal else Blur
            #                                          ,999,intensity=50)])
            # movingB = movingB.with_effects([HeadBlur(  NoBlur if isHorizontal else UnBlur, 
            #                                          UnBlur if isHorizontal else NoBlur
            #                                          ,999,intensity=50)])

            # Composite both together for the transition
            transition_clip: CompositeVideoClip = CompositeVideoClip(
                [movingA, movingB], size=(W, H)
            ).with_duration(transition_duration)

            # Append prev_main, then transition; carry curr_main forward
            final_clips.append(prev_main)
            final_clips.append(transition_clip)

            # Prepare for next iteration
            prev_segment = seg
            prev_full_clip = curr_main

        else:
            # One of the clips is too short; skip pan. Just append prev_full_clip
            final_clips.append(prev_full_clip)
            prev_segment = seg
            prev_full_clip = current_full

    # At loop end, append the final clip (whatever remains)
    if prev_full_clip is not None:
        final_clips.append(prev_full_clip)

    # Concatenate all pieces (compose to preserve sizes/fps)
    final_video = concatenate_videoclips(final_clips, method="compose")

    # Write out result
    final_video.write_videofile(output_file, codec="libx264",threads=16, bitrate='4000k')

    # Close everything
    for clip in final_clips:
        try:
            clip.close()
        except Exception:
            pass

    try:
        final_video.close()
    except Exception:
        pass

    video.close()



def extractAudio(video_path):
    try:
        video_clip = VideoFileClip(video_path)
        audio_path = "audio.wav"
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        print(f"Extracted audio to: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"An error occurred while extracting audio: {e}")
        return None

