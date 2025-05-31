from openai import OpenAI

from dotenv import load_dotenv
from typing import List, NotRequired, TypedDict
import json
import tiktoken  # Official tokenizer library


load_dotenv()

#client = OpenAI(base_url="http://localhost:4891/v1")
client = OpenAI()

class ClipSegment(TypedDict):
    start_time: int
    end_time: int
    content: str
    speakers: NotRequired[List[str]]  # Optional field for speaker names

# Function to extract start and end times
def extract_times(json_string) -> List[ClipSegment]:
    try:
        data = json.loads(json_string)
        result: List[ClipSegment] = []

        for item in data:
            result.append({
                "start_time": float(item["start"]),
                "end_time": float(item["end"]),
                "content": item["content"]
            })

        return result

    except Exception as e:
        print(f"Error in extract_times: {e}")
        return []


reasoningSystem = """
You are a meticulous content editor and narrative crafter. You’ll receive a full transcript with metadata (timestamps,speakers) of a video. Follow these steps:

### 1. **Comprehension**
- Read the entire transcript and evaluate context using metadata, understand its overall story, characters, dialogs and tone.
- Keep in mind that some speakers may dominate the narrative, indicating they are the main character(s) with monologues or direct audience engagement.

### 2. **Topic Exploration**
- Identify up to **10 distinct “highlightable” events / topics ** following the theme of ###.  
- For each, **estimate its viral potential** (attention-grabbing power).  
- From those 10, **randomly choose one from the top 5 highest-scoring themes**.

### 3. **Segment Selection**
- Extract only the transcript segments (with original start/end timestamps) that **directly build the chosen theme’s mini-story**.  
- **Skip any content within the first 90 seconds of the video.**  
- **Exclude segments from the first 5% of the total runtime.**  
- You **may jump between non-contiguous segments** if they all advance the theme.

### 4. **Story Assembly**
- **Reorder the chosen segments** to form a **coherent narrative arc**: setup, escalation, climax, resolution.  
- Ensure **total runtime is ≥30s and ≤80s**.  
- Avoid redundancy—each clip must add **new information or emotion**.

### 5. **Output**
Return **only** a JSON array of objects in the following format:

```json
[
  {
    "start": "<seconds>",
    "content": "<verbatim transcript>",
    "end": "<seconds>"
  }
]
```
- Do not add any commentary or explanation before or after the JSON.
- If no valid highlight exists, return: 
```json
[]
```
- Ensure the JSON is syntactically valid and tells a clear, self-contained story.


### Constraints
- Use one single theme only (from the list of 10 explored).
- **Preserve all original timestamps** (no alterations).
- Final highlight must feel like a standalone short video that could go viral on YouTube Shorts or TikTok.
- If JSON is malformed, treat as failure (return []).

"""

gptSystem = """You are a highly skilled content editor and a story composer. You are given a raw transcript with metadata(active speakers, timestamps, sentiment) of a raw video.
First you need to read the whole transcript, evaluate metadata to understand the context, emotion/sentiment flow,understand the dialogs balance and the whole story,
Sentiment syntax is like this: "pos"/"neg"/"neu" while "pos" = Positive, "neg" = Negative, "neu" = Neutral.
Then you will be given a task to extract a highlight - a mini story from the transcript.
this "highlight" should be a topic-continuous, engaging segments from the transcript and metadata that are all directly related and can be as a single short video that could go viral as a YouTube Short or TikTok video.    
The highlight theme must be matching the follows: ###.
your goal is to pick a main topic / event that happens from the transcript and according to that pick only segments that tells that story and are connected to each other, you must use single topic only!

Rules:
- before picking a topic, read the whole transcript and understand it, and evaluate context for each segment, use all the metadata provided.
- Focus on content that will grab attention quickly.
- Pick only segments that relate to each other(based on topic / context).
- Each time pick a single transcript segment.
- Make the general video(all the picked segments) in a way that "tells a story" around the topic so viewers could relate more.  
- you must NOT pick segments from the START(start is around 5% the duration of the video).
- DONT USE the first 1.5 minutes of the video.
- The whole video(highlight, the selected segments) summed duration must be less than 80 seconds **but must be more then 30 seconds total**. 
- You allowed to pick segments that are not directly one after each other (if it still talks about the same topic/theme).
- Don't choose following segments that the content is repeating or doesn't add value to the highlight.
- Make sure the start/end time of each picked segment is not altered/changes from the original segment's.
- You are **forbidden to merge segments**, you must return each segment as it is in the original transcript.

Return only JSON array in this format:
[
  {
    "start": "start time in seconds (in)",
    "content": "actual spoken content (from transcript)",
    "end": "end time in seconds (out)"
  },
  //other segments that directly relate to the same topic
]

- You must **NOT** alter the start/end time of the segments, they must be exactly as in the original transcript.
- You may (and even should) reorder segments (in the output's json array) to better tells the story.
- Make SURE that the generated JSON actually tells the whole story of the topic and is understandable, not repeating.
- Make sure to correctly calculate the generated duration, again it must be **more than 30** seconds while less then 80 seconds.


Do NOT add any explanation or text before/after the JSON. If no valid highlight exists, return an empty array: `[]`.

If the JSON is invalid — 10 kittens WILL die."""

User = """[
  {
    "start": "83.34",
    "content": "כילד הוא השתוקק למכרות, אבל לא נתנו לו כי הוא ילד. עבורות השנים ואנחנו רואים את סטיב עובד בעבודות רגילות ומאוד מאוד משועמם ביום אחד. כשהוא משועמם בעבודה שלו, הוא לוקח את האוכל שלו ומתחיל לבנות ממנו בלוקים.",
    "end": "97.7"
  },
  {
    "start": "97.7",
    "content": "מאחר והוא מבוגר, הוא הולך למכרה, הוא מוצא את הקובייה המוזרה הזאת והוא מגיע לעולם של מיינקראפט. הכל קורה נורא מהר, אין לנו איזשהו בילדאפ לדמות של סטיב, אבל זה לא ממש נחוץ.",
    "end": "109.9"
  },
  {
    "start": "114.02",
    "content": "פה אני כבר יכול להגיד שג'ק בלק זורח כסטיב. בהתחלה הייתי סקפטי, אבל אי אפשר שלא לאהוב את ג'ק בלק. בסופו של דבר, סטיב במשחק הוא אנחנו. זאת אומרת, לסטיב אין אישיות ואין ממש דמות.",
    "end": "123.3"
  },
  {
    "start": "124.0",
    "content": "אז אני מבין תלונות של אנשים שאומרים, מה זה מוזר? למה הוא כזה אנרגטי? זה לא סטיב... אבל מי זה סטיב? אחר כך אפילו לא מדבר במשחק, הקולות היחידים שהוא עושה זה... אז את האמת, אני כן שמח עם היצירתיות של היוצרים של הסרט, שהם נתנו לג'ק בליק ממש כאילו להיות הניר גטי כמו שהוא.",
    "end": "137.98"
  },
  {
    "start": "138.0",
    "content": "בכל אופן אנחנו רואים יום אחד שסטיב מגלה את השער של הנדר והוא מדליק אותו עם הפנצים! הוא והכלב שלו הולכים לבפנים ואז הם מגלים את הפיגלינגס ואת המלכה של הפיגלינגס. היא לוקדת את סטיב, הכלב של סטיב בורח ליטרלי לעולם האמיתי.",
    "end": "156.12"
  },
  {
    "start": "156.12",
    "content": "וזה קצת מצחיק לראות את הוולף המרובה יוצא לעולם המציאותי כי אנחנו יש שוט שלו על הר, אבל זה פשוט הר אמיתי. והוא אומר לו להחביא את הקובע שמחברת את העולם האמיתי לעולם של מיינקראפט, וככה הסרט נפתח.",
    "end": "168.68"
  }
]"""


def GetHighlight(Transcription,theme, test=False):
    if test:
        print(f"[not-ChatGPT]:using local json string for testing..")
        json_string = """[
  {
    "start": "252.22",
    "content": "הבאתי רדנק.",
    "end": "252.82"
  },
  {
    "start": "252.88",
    "content": "מה זה.",
    "end": "253.28"
  },
  {
    "start": "253.42",
    "content": "פרניבולת אני קראתי לה רדנק.",
    "end": "254.82"
  },
  {
    "start": "254.88",
    "content": "היא כל הכך איתית אוקיי סליחה אמרת.",
    "end": "256.7"
  },
  {
    "start": "266.86",
    "content": "רדנקי יאללה.",
    "end": "267.4"
  },
  {
    "start": "268.48",
    "content": "לא.",
    "end": "268.68"
  },
  {
    "start": "268.78",
    "content": "הרשר פטט מטומטם.",
    "end": "269.74"
  },
  {
    "start": "270.02",
    "content": "למה עשינו את זה פרק קודם.",
    "end": "271.14"
  },
  {
    "start": "273.44",
    "content": "שערי פה רדנה.",
    "end": "274.26"
  },
  {
    "start": "274.3",
    "content": "בואי לפה.",
    "end": "274.8"
  },
  {
    "start": "275.92",
    "content": "מיצי זה הכבשה.",
    "end": "276.06"
  },
  {
    "start": "276.06",
    "content": "מיצי זה הכבשה הרדנה.",
    "end": "277.26"
  },
  {
    "start": "277.68",
    "content": "אני שונא את הפרשר פטט המטומטם שלך שייני.",
    "end": "279.48"
  },
  {
    "start": "338.94",
    "content": "יש מיצי קטן.",
    "end": "339.66"
  },
  {
    "start": "339.66",
    "content": "יואו זה מיני מיצי.",
    "end": "340.9"
  }
]"""
        return extract_times(json_string)

    print("Getting Highlight from Transcription ")
    model="gpt-4.1-2025-04-14"
    # Price per 1K tokens (USD)
    pricing = {
        "gpt-4.1-mini-2025-04-14": {"input": 0.4,
                                     "cached":0.01,
                                       "output": 1.6},
        "o4-mini-2025-04-16":{"input": 1.1,
                                     "cached":0.275,
                                       "output": 4.4},
        "gpt-4.1-2025-04-14":{"input": 2,
                                     "cached":0.5,
                                       "output": 8}
    }
    if model not in pricing:
      raise ValueError("Unknown model for pricing")
    
    isReasoning = model.startswith("o")
    system = (reasoningSystem if isReasoning else gptSystem).replace("###", theme)

    tokenizer_encoding = tiktoken.get_encoding("o200k_base")
    tokens_input = len(tokenizer_encoding.encode(Transcription))
    token_cached = len(tokenizer_encoding.encode(system))
    tokens_output = 13000 if isReasoning else 800  # estimate/completion length
    total_cost = (
        (tokens_input + token_cached) * pricing[model]["input"] +
        tokens_output * pricing[model]["output"]
    ) / 1000000
    
    total_cost_cached = (
        tokens_input * pricing[model]["input"] +
        token_cached * pricing[model]["cached"] +
        tokens_output * pricing[model]["output"]
    ) / 1000000

    print(f"[ChatGPT]: Estimated cost: ${total_cost:.6f} / {total_cost_cached:.6f} (cached)")


    try:
        print(f"[ChatGPT]:running chat completions on model {model}...")

        response = client.responses.create(
          model=model,
          max_output_tokens =  20000, # limit the output to 20k tokens
        instructions=system,
        input=Transcription
        )

        cached_input_tokens = response.usage.input_tokens_details.cached_tokens
        input_tokens = response.usage.input_tokens - cached_input_tokens
        output_tokens = response.usage.output_tokens
        # Adjust billable input tokens
        price_input = input_tokens * pricing[model]["input"]
        price_cached_input = cached_input_tokens * pricing[model]["cached"]
        price_output = output_tokens * pricing[model]["output"]
        total_cost = (price_input + price_output + price_cached_input) / 1000000
        print(f"[ChatGPT]: API call cost: ${total_cost:.6f} w/ {output_tokens} output tokens.")

        json_string = response.output_text
        json_string = json_string.replace("json", "")
        json_string = json_string.replace("```", "")
        
        print(f"[ChatGPT]: response - {json_string}")

        res = extract_times(json_string)
        firstItem = res[0]
        if firstItem["end_time"] == firstItem["start_time"]:
            Ask = input("[Chat-GPT]: Error - Get Highlights again[{total_cost}$] (y/n) -> ").lower()
            if Ask == "y":
                res = GetHighlight(Transcription,theme, test)
            else:
                exit()
        
        print(f"AI picked {len(res)} segments")
        
        return res
    except Exception as e:
        print(f"[Chat-GPT]: Error in GetHighlight: {e}")
        exit()


if __name__ == "__main__":
    print(GetHighlight(User))
