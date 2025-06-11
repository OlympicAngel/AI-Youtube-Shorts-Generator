from typing import List, Optional
import ffmpeg

from Components.LanguageTasks import ClipSegment

def extractAudio(video_path, audio_path="audio.wav"):
    """
    Extract audio from a video file using ffmpeg-python.
    """
    try:
        ffmpeg.input(video_path).output(
            audio_path,
            acodec='pcm_s16le',  # WAV codec
            vn=None,             # Disable video
            loglevel='error'
        ).run(overwrite_output=True, quiet=True)
        print(f"Extracted audio to: {audio_path}")
        return audio_path
    except ffmpeg.Error as e:
        print(f"An error occurred while extracting audio: {e.stderr.decode()}")
        return None

def cut_segment(input_file: str, start: float, end: float, output_file: str, use_gpu: bool = True,codec: str = 'h264'):
    """
    Cut a segment [start, end] from input_file using ffmpeg-python.
    """
    input_kwargs = {'hwaccel': 'cuda'} if use_gpu else {}
    stream = ffmpeg.input(input_file, ss=start, t=end - start, **input_kwargs)
    
    output_args = {
        'c:a': 'aac',
        'b:a': '192k',
        'preset': 'fast'
    }
    if codec == 'h264':
        output_args['c:v'] = 'h264_nvenc' if use_gpu else 'libx264'
        output_args.update({'rc': 'vbr', 'cq': '24'})
    else:
        output_args['c:v'] = 'hevc_nvenc' if use_gpu else 'libx265'
        output_args['x265-params'] = 'crf=26'
    
    stream = ffmpeg.output(stream, output_file, **output_args)
    ffmpeg.run(stream, overwrite_output=True, quiet=True)
    
def get_video_info(input_file: str):
    """
    Return width, height, duration, fps using ffmpeg.probe.
    """
    probe = ffmpeg.probe(input_file)
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    if not video_streams:
        raise ValueError('No video stream found')

    v = video_streams[0]
    width = int(v['width'])
    height = int(v['height'])
    duration = float(v.get('duration', probe['format']['duration']))
    avg_frame_rate = v['avg_frame_rate']  # e.g., '30000/1001'
    num, den = map(float, avg_frame_rate.split('/'))
    fps = num / den if den else 0.0

    return width, height, duration, fps

def apply_transition(A_file: str, B_file: str, output_file: str,
                               pad: float,
                               motionBlurType: Optional[str],
                               width: int, height: int,
                               use_gpu: bool = True, codec: str = 'h264',fps=30.0):
    import random
    
    direction = random.choice(['left', 'right', 'up', 'down'])


    # Easing expression normalized to [0,1] over duration `pad`
    ease_expr = (
    f"if(lt(t,{pad}/2), "
    f"pow(2*t/{pad},5)/2, "
    f"(pow(2*t/{pad}-2,5)+2)/2)"
)

    if direction == 'left':
        exprA = {'x': f"if(lt(t,{pad}), -({ease_expr})*W, NAN)", 'y': '0'}
        exprB = {'x': f"if(lt(t,{pad}), W - ({ease_expr})*W, NAN)", 'y': '0'}
    elif direction == 'right':
        exprA = {'x': f"if(lt(t,{pad}), ({ease_expr})*W, NAN)", 'y': '0'}
        exprB = {'x': f"if(lt(t,{pad}), -w + ({ease_expr})*W, NAN)", 'y': '0'}
    elif direction == 'up':
        exprA = {'x': '0', 'y': f"if(lt(t,{pad}), -({ease_expr})*H, NAN)"}
        exprB = {'x': '0', 'y': f"if(lt(t,{pad}), H - ({ease_expr})*H, NAN)"}
    else:  # down
        exprA = {'x': '0', 'y': f"if(lt(t,{pad}), ({ease_expr})*H, NAN)"}
        exprB = {'x': '0', 'y': f"if(lt(t,{pad}), -h + ({ease_expr})*H, NAN)"}

    # Inputs
    base = ffmpeg.input(f'color=black:size={width}x{height}:duration={pad}:rate={fps}', f='lavfi',r=fps,)
    A = ffmpeg.input(A_file,hwaccel='cuda' if use_gpu else "none")
    B = ffmpeg.input(B_file,hwaccel='cuda' if use_gpu else "none")

    # First overlay (A on base)
    over1 = ffmpeg.overlay(base, A, shortest=1, **exprA)
    # Second overlay (B on top)
    over2 = ffmpeg.overlay(over1, B, shortest=1, **exprB)
 
 
    # Audio crossfade
    audio = ffmpeg.filter([A.audio, B.audio], 'acrossfade', d=pad)
    
    # Output
    output_args = {
        'c:a': 'aac',
        'b:a': '192k',
        'preset': 'fast'
    }
    if codec == 'h264':
        output_args['c:v'] = 'h264_nvenc' if use_gpu else 'libx264'
        output_args.update({'rc': 'vbr', 'cq': '24'})
    else:
        output_args['c:v'] = 'hevc_nvenc' if use_gpu else 'libx265'
        output_args['x265-params'] = 'crf=26'
        
 
    # Add optional motion blur
    if motionBlurType == 'optical':
        over2 = ffmpeg.filter(over2,'minterpolate',fps=round(fps*2.75), mi_mode='mci', mc_mode='obmc', search_param=100,scd="none") # aobmc slower
        over2 = ffmpeg.filter(over2, 'tmix', frames=6, weights="0.5 1 1 1 0.8 0.4")
        
    output = ffmpeg.output(over2, audio, output_file, **output_args,r=fps)
    ffmpeg.run(output, overwrite_output=True, quiet=True)
      
def edit_video_ffmpeg_py(input_file: str, output_file: str,
                      segments: List[ClipSegment],
                      transitionPad: float = 0.4,
                      motionBlurType: Optional[str] = "optical",
                      gap_threshold: float = 15.0,
                      use_gpu: bool = True,
                      codec: str = 'h264'):
    import os
    
    # TODO: threading for cutting segments
    
    print("extracting selected clips...")

    w, h, duration, fps = get_video_info(input_file)
    os.makedirs("temp_clips", exist_ok=True)

    n = len(segments)
    trans_needed = [(segments[i+1]['start_time'] - segments[i]['end_time'] > gap_threshold)
                    for i in range(n-1)]

    # Step 1: cut each segment, extending by pad if needed
    for i, clipSeg in enumerate(segments):
        start = clipSeg['start_time'] - (transitionPad if i>0 and trans_needed[i-1] else 0) # if previous segment needs transition add START padding
        end = clipSeg['end_time'] + (transitionPad if i<n-1 and trans_needed[i] else 0) # if current segment needs transition add END padding
        cut_segment(input_file, max(0, start), min(duration, end), f"temp_clips/seg{i}.mp4", use_gpu)

    print(f"segments extracted, applying transitions({trans_needed.count(True)})...")

    parts = []
    i = 0
    while i < n:
        seg_file = f"temp_clips/seg{i}.mp4"
        seg_duration = segments[i]['end_time'] - segments[i]['start_time'] # get duration of current segment

        # Check if transition is needed before / current segment
        if (i>0 and trans_needed[i-1]) or (i<n-1 and trans_needed[i]):
            # Handle base segment cutting (without transition/padding)
            trim_start = transitionPad if i>0 and trans_needed[i-1] else 0 # set padding for start if previous segment needs transition
            # get base's stream
            seg_stream = ffmpeg.input(seg_file, ss=trim_start, t=seg_duration - trim_start, hwaccel='cuda' if use_gpu else "none")
            # save base segment
            base_file = f"temp_clips/seg{i}base.mp4"
            ffmpeg.output(seg_stream, base_file, c='copy').run(overwrite_output=True, quiet=True)
            parts.append(base_file)
        else:
            parts.append(seg_file) # not transition needed, just append the segment

        # Insert transition if needed
        if i<n-1 and trans_needed[i]:
            A_overlap = f"temp_clips/overlap_{i}.mp4"
            B_pre = f"temp_clips/pre_{i+1}.mp4"
            trans_file = f"temp_clips/trans_{i}_{i+1}.mp4"
            
             # Overlap last part of segment i
            (
                ffmpeg
                .input(seg_file, ss=seg_duration, t=transitionPad, hwaccel='cuda' if use_gpu else "none")
                .output(A_overlap, c='copy')
                .run(overwrite_output=True, quiet=True)
            )

            # Start of next segment
            (
                ffmpeg
                .input(f"temp_clips/seg{i+1}.mp4", t=transitionPad, hwaccel='cuda' if use_gpu else "none")
                .output(B_pre, c='copy')
                .run(overwrite_output=True, quiet=True)
            )
            
            # Apply transition
            apply_transition(A_overlap, B_pre, trans_file,
                            transitionPad, motionBlurType,
                            w, h, use_gpu, codec,fps=fps)
            parts.append(trans_file)
            
            # Also include rest of segment i+1 after transition pad
            base_file = f"temp_clips/seg{i+1}base.mp4"
            orig_len_next = segments[i+1]['end_time'] - segments[i+1]['start_time']
            (
                ffmpeg
                .input(f"temp_clips/seg{i+1}.mp4", ss=transitionPad, t=orig_len_next, hwaccel='cuda' if use_gpu else "none")
                .output(base_file, c='copy')
                .run(overwrite_output=True, quiet=True)
            )
            parts.append(base_file)
            
            i += 1  # next segment(mind the next i++)
        i += 1 # increment to next segment

    # Final concat
    
    print("concatenating selected segments & transitions...")
    
    out_codec = {
            'h264': ('h264_nvenc' if use_gpu else 'libx264'),
            'hevc': ('hevc_nvenc' if use_gpu else 'libx265'),
    }[codec]

    out_args = {
            'vcodec': out_codec,
            'acodec': 'aac',
            'b:a': '192k',
    }
    if codec == 'h264':
            out_args.update({'preset': 'fast', 'rc': 'vbr', 'cq': '24'})
    else:
            out_args.update({'preset': 'fast', 'x265-params': 'crf=26'})

    # Create input streams
    streams = [ffmpeg.input(p, hwaccel='cuda' if use_gpu else "none") for p in parts]
    
    # Split inputs into video/audio streams
    video_streams = [input.video for input in streams]
    audio_streams = [input.audio for input in streams]
    
    # Flatten all streams: [v1, a1, v2, a2, ..., vn, an]
    all_streams = []
    for v, a in zip(video_streams, audio_streams):
        all_streams.extend([v, a])
        
    # Concatenate all streams
    concat_filter = ffmpeg.concat(*all_streams, v=1, a=1, n=len(parts))
    out = ffmpeg.output(concat_filter,
                        output_file,**out_args)
    ffmpeg.run(out, overwrite_output=True,quiet=True)


    # Clean temp
    for f in os.listdir("temp_clips"):
        os.remove(os.path.join("temp_clips", f))
