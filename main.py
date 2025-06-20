import time
start_time = time.time()
from Components.UserInput import input_get_theme, input_get_videoSource, test
from typing import List
from Components.Editor import edit_video_ffmpeg_py, extractAudio
from Components.EmotionMetadata import TranscribeSegmentType_withSpeakersAndSentiment, add_hebrew_sentiment
from Components.SpeakersMetadata import TranscribeSegmentType_withSpeakers, assign_speaker, get_speakers_metadata
from Components.TranscriptionTimingRefine import refine_transcript
from Components.Transcription import save_transcription, transcribeAudio, transcription_cache_path
from Components.LanguageTasks import GetHighlight
from Components.FaceCrop import  combine_videos, crop_to_vertical_debug
import uuid

print("is test run:", test)

def remove_consecutive_duplicates(s: str) -> str:
    words = s.split()
    if not words:
        return ""
    result = [words[0]]
    for w in words[1:]:
        if w != result[-1]:
            result.append(w)
    return " ".join(result)

Vid = input_get_videoSource()
shortTheme = input_get_theme()
print("Selected short theme:", shortTheme)

# extract audio from video into local directory
Audio = extractAudio(Vid,"audio.wav")
if not Audio:
     raise FileNotFoundError("No audio file found after exacting audio from video. Please check the video file.")

#get transcription (generate / load from cache)
transcriptions = transcribeAudio(Vid,Audio)

# check of already has speakers metadata
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
for text, start, end, speakers_raw, sentiment in transcriptions:
    speakers = ", ".join([("#"+s.replace("SPEAKER_", "")) for s in speakers_raw])
    TransText += f"[{speakers}] {start}->{end}: '{remove_consecutive_duplicates(text)}' ({sentiment})\n"
TransText = TransText[:-1]  # remove last comma or \n

# get highlights from transcriptions using GPT
clipSegments = GetHighlight(TransText,shortTheme,test)
if len(clipSegments) == 0: # Error no highlights found
    raise TypeError("Error in getting highlight")

# refine the clip segments using VAD
refined_clipSegments = refine_transcript(Audio, clipSegments) if not test else clipSegments

# trim video based on clip segments
trimmedVideoPath = "trimmed.mp4"
edit_video_ffmpeg_py(Vid,trimmedVideoPath,refined_clipSegments,gap_threshold=15,motionBlurType=None)


# crop trimmed video to vertical format
croppedVideoPath = "verticalCropped.mp4"
crop_to_vertical_debug(trimmedVideoPath, croppedVideoPath,False)

# combine trimmed and cropped videos into a single short
uuid = uuid.uuid4()
combine_videos(trimmedVideoPath, croppedVideoPath, f"Generated Shorts/{shortTheme.split("/")[0]}_{Vid.split("\\").pop().split('.')[0]}_{str(uuid)}.mp4")
print("--- DONE in %s seconds ---" % (time.time() - start_time))
