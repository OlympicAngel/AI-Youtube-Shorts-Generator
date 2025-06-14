import sys
from  os import path
from Components.LanguageTasks import readPromptFile
from Components.YoutubeDownloader import download_youtube_video

test = (sys.argv[3] if len(sys.argv) >= 4 else input("Is this a test run? (true/false): ").strip().lower()) == "true"


def input_get_theme():
    """
    Get the theme for the short video from user input or command line argument.
    """
    themeCount = 5
    def get_theme_prompt(choice):
        if choice == '1':
            return readPromptFile("themes/funny")
        elif choice == '2':
            return readPromptFile("themes/emotional")
        elif choice == '3':
            return readPromptFile("themes/intense")
        elif choice == '4':
            return readPromptFile("themes/info")
        elif choice == '5':
            return readPromptFile("themes/any")
        raise ValueError(f"Invalid input. Please enter a number between 1 and {themeCount} representing a theme.")

    
    user_input_selectedTheme = sys.argv[2] if len(sys.argv) >= 3 else None
    # ifuse provided theme from command line argument
    if user_input_selectedTheme and len(user_input_selectedTheme) == 1:
        return get_theme_prompt(user_input_selectedTheme)


    while True:
        print("Select short theme:")
        print("1 - Funny")
        print("2 - Emotional")
        print("3 - Intense")
        print("4 - Informational")
        print("5 - Any")
        choice = input("Enter 1, 2, 3, 4 or 5: ").strip()

        try:
            return get_theme_prompt(choice)
        except:
            print("Invalid input. Please enter 1, 2, 3, 4 or 5.\n")           

def input_get_videoSource():
    user_input_videoSource = sys.argv[1] if len(sys.argv) >= 2 else input("Enter YouTube URL / local path: ")
    if user_input_videoSource.startswith("http"):
        # Online source
        url = user_input_videoSource
        video_path = download_youtube_video(url) # download the video at max resolution
        if isinstance(video_path,str):
            print(f"Downloaded video and audio files successfully! at {video_path}")
            return video_path
        else:
            error = video_path
            raise error
    else:
        # Local source
        if(not path.exists(user_input_videoSource)):
            raise FileNotFoundError(f"Local video file not found: {user_input_videoSource}")
        return user_input_videoSource  