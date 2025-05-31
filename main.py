import json
import os
import faulthandler
faulthandler.enable()
from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extractAudio, crop_video
from Components.Transcription import transcribeAudio
from Components.LanguageTasks import GetHighlight
from Components.FaceCrop import  combine_videos, crop_to_vertical_debug

def cache_path(video_path: str) -> str:
    return video_path + ".transcription.json"

def save_transcription(path: str, data):
    try:
        print(f"Saving transcription to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure dir exists
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print("✅ Transcription successfully saved.")
    except Exception as e:
        print(f"❌ Failed to save transcription: {e}")

def load_transcription(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def get_video_source():
    while True:
        print("Select video source:")
        print("1 - Local")
        print("2 - Online")
        choice = input("Enter 1 or 2: ").strip()

        if choice == '1':
            return True   # Local
        elif choice == '2':
            return False  # Online
        else:
            print("Invalid input. Please enter 1 or 2.\n")

def get_short_theme():
    while True:
        # underdog, dark-humor, intense, , weird, shocking, or thought-provoking
        print("Select short theme:")
        print("1 - Funny")
        print("2 - Emotional")
        print("3 - Intense")
        print("4 - Informational")
        print("5 - Any")
        choice = input("Enter 1, 2, 3, 4 or 5: ").strip()

        if choice == '1':
            return "funny, light, trolls, pranks, comedy, humor, hilarious, funny moments, dark-humor"
        elif choice == '2':
            return "emotional, motivational, inspiring, uplifting, positive, heartwarming, sad, heartbreak, betryal, love, romance"
        elif choice == '3':
            return "intense, action, thriller, suspense, dramatic, high-energy, adrenaline-pumping, shocking"
        elif choice == '4':
            return "informational, educational, knowledge, facts, learning, science, culture, guide, how-to, tricks, life-hacks"
        elif choice == '5':
            return "funny, light, trolls. pranks, underdog, dark-humor, intense, emotional, weird, shocking, or thought-provoking"
        else:
            print("Invalid input. Please enter 1, 2, 3, 4 or 5.\n")
            get_short_theme()
            

# Main logic
isLocal = get_video_source()

shortTheme = get_short_theme()
print("Selected short theme:", shortTheme)

if isLocal:
    Vid = input("Enter local video path: ").strip()
else:
    url = input("Enter YouTube video URL: ")
    Vid = download_youtube_video(url)
    if Vid:
        Vid = Vid.replace(".webm", ".mp4")
        print(f"Downloaded video and audio files successfully! at {Vid}")
    else:
        print("Unable to Download the video")
        exit()

Audio = extractAudio(Vid)
if Audio:
        cache_file = cache_path(Vid)

        if os.path.exists(cache_file):
            transcriptions = load_transcription(cache_file)
        else:
            transcriptions = transcribeAudio(Audio)
            if len(transcriptions) == 0:
                exit()
            save_transcription(cache_file, transcriptions)
            print("Transcription saved to cache at '"+cache_file+"'.")
            
        if len(transcriptions) > 0:
            TransText = "" 

            for text, start, end in transcriptions:
                TransText += (f"{start} - {end}: {text}.\n")

            clipSegments = GetHighlight(TransText,shortTheme)
            if len(clipSegments) > 0:
                print(f"AI picked {len(clipSegments)} segments")

                Output = "Out.mp4"

                crop_video(Vid, Output, clipSegments)
                croped = "croped.mp4"

                crop_to_vertical_debug("Out.mp4", croped,False)
                combine_videos("Out.mp4", croped, "Final.mp4")
            else:
                print("Error in getting highlight")
        else:
            print("No transcriptions found")
else:
        print("No audio file found")