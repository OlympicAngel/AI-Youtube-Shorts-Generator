import os
import subprocess
import json
from pathlib import Path
from typing import Tuple

TranscribeSegmentType = Tuple[str, float, float]


def transcribeAudio(video_path: str,audio_path:str) -> list[TranscribeSegmentType]:
    # check if transcription cache exists
    cache_file = transcription_cache_path(video_path)
    if os.path.exists(cache_file):
        transcriptions = load_transcription(cache_file) #load it
    else: # else transcribe it
        transcriptions = transcribeAudio_logic(audio_path)
        save_transcription(cache_file, transcriptions) # save it 
        if len(transcriptions) == 0:
            print("No transcriptions found")
            exit()
        print("Transcription saved to cache at '"+cache_file+"'.")
        
    return transcriptions

def transcribeAudio_logic(audio_path: str): 
    try:
        print(f"Transcribing audio (subprocess)...{audio_path}")

        venv_python = Path("clean_env/Scripts/python.exe")
        env = os.environ.copy()

        process = subprocess.Popen(
            [venv_python, "Components/transcribe_worker.py", audio_path],
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            encoding="utf-8"
        )

        output_lines = []
        for line in process.stdout:
            try:
                parsed = json.loads(line)
                output_lines = parsed
            except json.JSONDecodeError as e:
                raise RuntimeError(f"\nJSON parsing error: {e}\nRaw Line Output: {repr(line)}")
            
        process.wait()

        if process.returncode != 0 and not output_lines:
            raise RuntimeError("Subprocess failed with no output")
        
        print("Transcription subprocess completed")

        if isinstance(output_lines, dict) and "error" in output_lines:
            raise RuntimeError(output_lines["error"])

        print("Transcription completed")
        print("Total segments:", len(output_lines))
                
        return output_lines

    except Exception as e:
        print("Transcription Error:", json.dumps({"error": str(e)}))
        exit()


# utils
def transcription_cache_path(video_path: str) -> str:
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

def load_transcription(path: str)->list[TranscribeSegmentType]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)