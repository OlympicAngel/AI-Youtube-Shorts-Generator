import os
import re
import unicodedata
from pytubefix import YouTube
import ffmpeg

def get_video_size(stream):

    return stream.filesize / (1024 * 1024)

def download_youtube_video(url):
    if not os.path.exists('videos'):
        os.makedirs('videos')
        
    try:
        yt = YouTube(url)

        output_file = os.path.join('videos', f"{sanitize_filename(yt.video_id)}.mp4")
        if os.path.exists(output_file):
            print(f"File already exists: {output_file}")
            return output_file

        selected_stream = yt.streams.get_highest_resolution()
        audio_stream = yt.streams.get_audio_only()

        print(f"Downloading video: {yt.title}")
        video_file = selected_stream.download(output_path='videos', filename_prefix="video_")

        if not selected_stream.is_progressive:
            print("Downloading audio...")
            audio_file = audio_stream.download(output_path='videos', filename_prefix="audio_")

            print("Merging video and audio...")
            input_kwargs = {'hwaccel': 'cuda'}
            stream = ffmpeg.input(video_file,**input_kwargs)
            audio = ffmpeg.input(audio_file,**input_kwargs)
            stream = ffmpeg.output(stream, audio, output_file, vcodec='h264_nvenc', acodec='aac')
            ffmpeg.run(stream, overwrite_output=True,quiet=True)

            os.remove(video_file)
            os.remove(audio_file)
        else:
            output_file = video_file

        print(f"Downloaded: {yt.title} to 'videos' folder")
        print(f"File path: {output_file}")
        return output_file

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure you have the latest version of pytube and ffmpeg-python installed.")
        print("You can update them by running:")
        print("pip install --upgrade pytube ffmpeg-python")
        print("Also, ensure that ffmpeg is installed on your system and available in your PATH.")

def sanitize_filename(title):
    # Normalize Unicode, remove problematic chars
    safe_title = unicodedata.normalize('NFKD', title)
    safe_title = re.sub(r'[\\/*?:"<>|]', "", safe_title)
    return safe_title