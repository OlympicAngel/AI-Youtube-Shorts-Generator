This is a fork of https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator for this a have changed / added:
- Whisper runs on Worker - prevent crush on windows.
- Optional load video from local rather than youtuber directly.
- Transcription & video download caching.
- GPT prompt improvements.
- Hardcoded selection of GPT module selection.
- Supports for GPT's "O" (reasoning) model with optimized prompt and universal API workflow.
- Dialog to select theme - what should GPT look for (Funny / Emotional / Intense / Informational).
- Modified FaceCrop.py for: smoother transition, on failed detection attempt to focus on the attraction point around the center of the frame (fast moving objects around the center).
- Timing refinement using silero-vad preventing "hard cuts" at a start / end of a sub-clip (audio).
- Speakers detection for each segment - for better gpt results
- Sentimental detection for each segment (using context for surrounding segments) - for better gpt results.
- Complete replacement of Moviepy - with Python ffmpeg / raw ffmpeg(cli).
- Replaced face detection to use YOLO5 - a GPU acc with detection of both human & general objects (attempt to focus frame central object as fall back).
- cli support.
- General fixed / optimization for my needs.


# AI Youtube Shorts Generator

AI Youtube Shorts Generator is a Python tool designed to generate engaging YouTube shorts from long-form videos. By leveraging the power of GPT-4, Whisper, it extracts the most interesting highlights, detects speakers, and crops the content vertically for shorts while focusing potential interest points.


## Features

- **Video Download**: Given a YouTube URL, the tool downloads the video.
- **Transcription**: Uses Whisper to transcribe the video.
- **Highlight Extraction**: Utilizes OpenAI's GPT-4 to identify the most engaging parts of the video.
- **Speaker Detection**: Detects speakers in the video.
- **Vertical Cropping**: Crops the highlighted sections vertically, making them perfect for shorts.

## Installation

### Prerequisites

- Python 3.7 or higher
- FFmpeg
- OpenCV

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/OlympicAngel/AI-Youtube-Shorts-Generator.git
   cd AI-Youtube-Shorts-Generator
   ```

2. Create a virtual environment

```bash
python3.10 -m venv venv
```

3. Activate a virtual environment:

```bash
source venv/bin/activate # On Windows: venv\Scripts\activate
```

4. Install the python dependencies:

```bash
pip install -r requirements.txt
```

---

1. Set up the environment variables.

Create a `.env` file in the project root directory and add your OpenAI API key:

```bash
OPENAI_API=your_openai_api_key_here
HF_TOKEN=hugging_face_token_here
```

## Usage

1. Ensure your `.env` file is correctly set up.
2. Run the main script and enter the desired YouTube URL when prompted:
   run time:
   ```bash
   python main.py
   ```
   and follow internal instructions.
   ---
   cli:
    ```bash
   python main.py "pathToYourVideo_OR_youtubeUrlWithHttp" "selectedTheme(1-5)" "testMode(True/False)"
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
