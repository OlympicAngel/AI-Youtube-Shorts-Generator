import os
import subprocess
import json
from pathlib import Path
import sys

def transcribeAudio(path):
    try:
        print("Transcribing audio (subprocess)...")

        venv_python = Path(".venv/Scripts/python.exe")
        env = os.environ.copy()

        process = subprocess.Popen(
            [venv_python, "Components/transcribe_worker.py", path],
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
        
        print(len(output_lines))


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
        return []
