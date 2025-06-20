from typing import List
import os
import torch
from typing_extensions import Unpack
from Components.Transcription import TranscribeSegmentType

prefix = "[SpeakersMetadata]: "

speakerSegmentType = tuple[float, float, str]
TranscribeSegmentType_withSpeakers = tuple[Unpack[TranscribeSegmentType], List[str]]

def get_speakers_metadata(audio_path):
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token (HF_TOKEN) is not set in environment variables.")
    
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    pipeline.to(torch.device("cuda"))

    with Spinner(prefix + "Running diarization (speaker detection)"):
        diarization:Annotation  = pipeline(audio_path)

    speaker_segments:list[speakerSegmentType] = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    return speaker_segments


# Spinner thread
class Spinner:
    def __init__(self, message="Processing"):
        import threading
        self.message = message
        self.spinner = ['|', '/', '-', '\\']
        self.running = False
        self.thread = threading.Thread(target=self.spin)

    def spin(self):
        import sys
        import time

        i = 0
        while self.running:
            sys.stdout.write(f"\r{self.message}... {self.spinner[i % len(self.spinner)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        sys.stdout.write("\rDone.                                 \n")

    def __enter__(self):
        self.running = True
        self.thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread.join()
        
def assign_speaker(transcript:List[TranscribeSegmentType], speaker_segments:list[speakerSegmentType])->List[TranscribeSegmentType_withSpeakers]:
    results = []
    for segment in transcript:
        content,t_start, t_end = segment

        # Collect unique overlapping speakers using set
        speakers = list({
            spk.replace("SPEAKER_","") for s_start, s_end, spk in speaker_segments
            if t_start < s_end and t_end > s_start
        })

        results.append((content,t_start, t_end,speakers))
    return results