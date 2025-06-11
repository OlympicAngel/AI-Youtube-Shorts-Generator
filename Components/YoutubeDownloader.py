import os

def get_video_size(stream):
    return stream.filesize / (1024 * 1024)

def download_youtube_video(url):
    from pytubefix import YouTube
    import ffmpeg
    
    if not os.path.exists('videos'):
        os.makedirs('videos')
        
    try:
        yt = YouTube(url)

        output_file = os.path.join('videos', f"{sanitize_filename(yt.video_id)}.mp4")
        if os.path.exists(output_file):
            print(f"File already exists: {output_file}")
            return output_file

        #selected_stream = yt.streams.get_highest_resolution(progressive=False)
        selected_stream = yt.streams.filter(type="video").order_by("filesize").desc()[0]
        if(selected_stream == None):
            raise ("No suitable video stream found.")
        
        print(f"Resolution: {selected_stream.resolution}, Size: {get_video_size(selected_stream):.2f} MB.")

        
        audio_stream = yt.streams.get_audio_only()

        print(f"Downloading video: {yt.title}")
        video_file = selected_stream.download(output_path='videos', filename_prefix="video_")

        print("Downloading audio...")
        audio_file = audio_stream.download(output_path='videos', filename_prefix="audio_")

        print("Merging video and audio...")
        input_kwargs = {'hwaccel': 'cuda'}
        stream = ffmpeg.input(video_file,**input_kwargs)
        audio = ffmpeg.input(audio_file,**input_kwargs)
        stream = ffmpeg.output(stream, audio, output_file, vcodec='h264_nvenc', acodec='aac')
        ffmpeg.run(stream, overwrite_output=True,quiet=False)

        os.remove(video_file)
        os.remove(audio_file)

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
    import re
    import unicodedata

    # Normalize Unicode, remove problematic chars
    safe_title = unicodedata.normalize('NFKD', title)
    safe_title = re.sub(r'[\\/*?:"<>|]', "", safe_title)
    return safe_title