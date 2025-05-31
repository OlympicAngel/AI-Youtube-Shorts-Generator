import json
import os
import faulthandler
from typing import List
from Components.EmotionMetadata import TranscribeSegmentType_withSpeakersAndSentiment, add_hebrew_sentiment
from Components.SpeakersMetadata import TranscribeSegmentType_withSpeakers, assign_speaker, get_speakers_metadata
from Components.TranscriptionTimingRefine import refine_transcript
from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extractAudio, crop_video
from Components.Transcription import save_transcription, transcribeAudio, transcription_cache_path
from Components.LanguageTasks import GetHighlight
from Components.FaceCrop import  combine_videos, crop_to_vertical_debug
import uuid

test = True

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
            return "funny, light, trolls, pranks, comedy, humor, hilarious, funny moments, dark-humor, funny-dialogs(using speakers data), mostly positive sentiment"
        elif choice == '2':
            return "emotional, motivational, inspiring, uplifting, positive, heartwarming, sad, heartbreak, betrayal, love, romance, emotional-dialogs(using speakers data), contrasting sentiment"
        elif choice == '3':
            return "intense, action, thriller, suspense, dramatic, high-energy, adrenaline-pumping, shocking, dramatic-dialogs(using speakers data), mostly negative sentiment"
        elif choice == '4':
            return "informational, educational, knowledge, facts, learning, science, culture, guide, how-to, tricks, life-hacks"
        elif choice == '5':
            return "funny, light, trolls. pranks, underdog, dark-humor, intense, emotional, weird, shocking, or thought-provoking, interesting-dialogs(using speakers data)"
        else:
            print("Invalid input. Please enter 1, 2, 3, 4 or 5.\n")
            get_short_theme()

def remove_consecutive_duplicates(s: str) -> str:
    words = s.split()
    if not words:
        return ""
    result = [words[0]]
    for w in words[1:]:
        if w != result[-1]:
            result.append(w)
    return " ".join(result)

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
# get audio from video
Audio = extractAudio(Vid)
if not Audio:
     print("No audio file found")

#get transcription (generate / load from cache)
transcriptions = transcribeAudio(Vid,Audio)

# check of alrady has speakers metadata
firstSegment = transcriptions[0]
if(len(firstSegment) <= 3): # if no speakers metadata (len should be 4+)
    print("No speakers metadata found in transcription, generating...")
    # get speakers metadata from audio
    speakers = get_speakers_metadata(Audio)
    # assign speakers to transcription segments
    transcriptions = assign_speaker(transcriptions, speakers)
    
    # resave transcription with speakers metadata
    cache_file = transcription_cache_path(Vid)
    save_transcription(cache_file,transcriptions)
# redeclare type
transcriptions: List[TranscribeSegmentType_withSpeakers]

transcriptions = add_hebrew_sentiment(transcriptions) if not test else [(a,b,c,d,"") for a,b,c,d in transcriptions]
transcriptions: List[TranscribeSegmentType_withSpeakersAndSentiment]

     
# convert transcriptions to text format for GPT
TransText = ""
for text, start, end, speakers, sentiment in transcriptions:
    clean_speakers = [("#"+s.replace("SPEAKER_", "")) for s in speakers]
    TransText += json.dumps({'speakers': ", ".join(clean_speakers),'in':start,'out':end,'content':remove_consecutive_duplicates(text),'sentiment':sentiment},
                            ensure_ascii=False,separators=(',', ':')) + ","
TransText = TransText[:-1]  # remove last comma

# get highlights from transcriptions using GPT
clipSegments = GetHighlight(TransText,shortTheme,test)
if len(clipSegments) == 0: # Error no highlights found
    print("Error in getting highlight")

# refine the clip segments using VAD
refined_clipSegments = refine_transcript(Audio, clipSegments) if not test else clipSegments

# trim video based on clip segments
trimmedVideoPath = "trimmed.mp4"
crop_video(Vid, trimmedVideoPath, clipSegments)

# crop trimmed video to vertical format
croppedVideoPath = "verticalCropped.mp4"
crop_to_vertical_debug(trimmedVideoPath, croppedVideoPath,False)

# combine trimmed and cropped videos into a single short
uuid = uuid.uuid4()
combine_videos(trimmedVideoPath, croppedVideoPath, f"Generated Shorts/{shortTheme.split(",")[0]}_{Vid.split("\\").pop().split('.')[0]}_{str(uuid)}.mp4")
    