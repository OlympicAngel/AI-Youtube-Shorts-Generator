from typing import List
from moviepy.video.io.VideoFileClip import VideoFileClip

from moviepy.editor import VideoFileClip, concatenate_videoclips

from Components.LanguageTasks import ClipSegment

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



def crop_video(input_file: str, output_file: str, segments: List[ClipSegment]) -> None:
    subclips = []

    with VideoFileClip(input_file) as video:
        for segment in segments:
            subclip = video.subclip(segment["start_time"], segment["end_time"])
            subclips.append(subclip)

        final_clip = concatenate_videoclips(subclips, method="compose")
        final_clip.write_videofile(output_file, codec='libx264')

        for clip in subclips:
            clip.close()
        final_clip.close()

# Example usage:
if __name__ == "__main__":
    input_file = r"Example.mp4" ## Test
    print(input_file)
    output_file = "Short.mp4"
    start_time = 31.92 
    end_time = 49.2   

    crop_video(input_file, output_file, start_time, end_time)

