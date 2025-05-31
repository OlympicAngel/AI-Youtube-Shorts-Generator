from math import floor
import sys
import json
from faster_whisper import WhisperModel, BatchedInferencePipeline
import torch
import re
if torch.cuda.is_available():
    torch.cuda.init()

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
    
import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.ERROR)
    
eprint("[CUDA] is_available:", torch.cuda.is_available())
eprint("[CUDA] device_count:", torch.cuda.device_count())

try:
    audio_path = sys.argv[1]
    model_path = "D:/coding/TranscribeService/whisper-large-v3-turbo-ct2"

    if torch.backends.cuda.is_built() and torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    eprint(f"[Whisper] Using device: {device.upper()}")

    model = WhisperModel(model_path, device=device)
    segments, _ = model.transcribe(
        audio_path,
        beam_size=3,
        language="he",
        condition_on_previous_text=False,
        vad_filter=True,
    )

    result = []

    for segment in segments:
        clean_text = segment.text.replace('\n', ' ').replace('\r', ' ').strip()
        clean_text = re.sub(r'[^א-תa-zA-Z0-9 ]+', '', clean_text).strip()
        data = [clean_text, round(segment.start, 4), round(segment.end, 4)]
        result.append(data)
        
        if(len(result) % 10 == 0):
            eprint(f"[Whisper] {len(result)}#")
            
    result[-1][2] = floor(result[-1][2])
    
    print(json.dumps(result, ensure_ascii=False).replace('\n', ' ').replace('\r', ' ').strip(), flush=True)
    eprint(f"[Whisper] DONE.. {len(result)}")

    torch.cuda.empty_cache()
    sys.exit(0)

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
