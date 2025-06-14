import ffmpeg

input_path = 'input.mp4'
output_path = 'output.mp4'
text = 'Your TikTok Title'
font_path = '/path/to/font.ttf'  # Use bold font like Montserrat-Bold.ttf

# Create a text overlay video with fade-in, scale-in, and blur shadow using a complex filter
drawtext = (
    f"drawtext=fontfile='{font_path}':"
    f"text='{text}':"
    "x=(w-text_w)/2:"
    "y=h*0.15:"
    "fontsize=48:"
    "fontcolor=white@1.0:"
    "shadowcolor=black@0.7:"
    "shadowx=0:shadowy=0:"
    "alpha='if(lt(t,0.5), 0, if(lt(t,1.5), (t-0.5)/1.0, 1))':"
    "enable='lt(t,5)'"
)

# Use scale filter before drawtext to simulate slight zoom-in effect
filter_complex = (
    "[0:v]scale=iw*if(lt(t,1.5), 0.98+(t*0.02/1.5), 1):ih*if(lt(t,1.5), 0.98+(t*0.02/1.5), 1),"
    + drawtext
)

ffmpeg.input(input_path).output(
    output_path,
    vf=filter_complex,
    **{"c:a": "copy"}
).run(overwrite_output=True)
