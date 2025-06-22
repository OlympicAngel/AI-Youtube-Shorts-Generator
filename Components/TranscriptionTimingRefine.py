from typing import List
from Components.LanguageTasks import ClipSegment

# Max adjustment allowed when refining (in seconds)
max_adjust = 0.45  

def refine_transcript(audio_path:str, transcript: List[ClipSegment]):
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    import concurrent.futures
    
    print("Refining transcript timing...")
        
    # Load VAD model on CUDA or CPU
    model = load_silero_vad()

    # Read audio (mono, 16kHz expected)
    wav = read_audio(audio_path)

    # Run VAD on the audio
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,
        threshold=0.65,
        min_speech_duration_ms=200,
        min_silence_duration_ms=150,
        max_speech_duration_s=15,
        speech_pad_ms=60,
        progress_tracking_callback=lambda progress : print(f"[VAD]: Progress - {round(progress)}%") if round(progress,3) % 5 == 0 else None
    )
    
    # --- Segment refinement logic ---
    def refine_segment(seg:ClipSegment) -> ClipSegment:
        start_sec, end_sec = seg.start_time, seg.end_time
        refined_start, refined_end = start_sec, end_sec

        # Find closest start within ±max_adjust range
        start_candidates = [
            chunk['start'] for chunk in speech_timestamps
            if abs(chunk['start'] - start_sec) <= max_adjust
        ]
        if start_candidates:
            refined_start = min(start_candidates, key=lambda x: abs(x - start_sec))

        # Find closest end within ±max_adjust range
        end_candidates = [
            chunk['end'] for chunk in speech_timestamps
            if abs(chunk['end'] - end_sec) <= max_adjust
        ]
        if end_candidates:
            refined_end = min(end_candidates, key=lambda x: abs(x - end_sec))

        return ClipSegment(
        start_time=refined_start,
        end_time=refined_end,
        content=seg.content,
    )

    # --- Parallel processing of transcript segments ---
    with concurrent.futures.ThreadPoolExecutor() as executor:
        refined_transcript = list(executor.map(refine_segment, transcript))

    return refined_transcript
