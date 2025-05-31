from openai import OpenAI

from dotenv import load_dotenv
from typing import List, TypedDict
import json
import tiktoken  # Official tokenizer library


load_dotenv()

#client = OpenAI(base_url="http://localhost:4891/v1")
client = OpenAI()

class ClipSegment(TypedDict):
    start_time: int
    end_time: int
    content: str

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
You are a meticulous content editor and narrative crafter. You’ll receive a full transcript (with timestamps) of a video. Follow these steps:

### 1. **Comprehension**
- Read the entire transcript to understand its overall story, characters, and tone.  
- **Do not select any content until you’ve fully absorbed the context.**

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

gptSystem = """You are a highly skilled content editor and a story composer. You are given a raw transcript with timestamps of a raw video.
First you need to read the whole transcript and understand the story of it, then you will be given a task to extract a highlight - a mini story from the transcript.
this "highlight" should be a topic-continuous, engaging segments from the transcript that are all directly related and can be as a single short video that could go viral as a YouTube Short or TikTok video.    
The highlight theme must be 1 of the follows: ###.
the goal is to pick a topic / event that happens from the transcript and according to that pick only segments that tells that story and are connected to each other, you must use single topic only!

Rules:
- before picking a topic, read the whole transcript and understand it, and evaluate context for each segment.
- Focus on content that will grab attention quickly - pick ~10 possible topics and evaluate their possible attention rate, from that top 10 pick randomly between the top 5 topics.
- Pick only segments that relate to each other(based on topic / context).
- Each time pick a single transcript segment.
- Make the general video(all the picked segments) in a way that "tells a story" around the topic so viewers could relate more.  
- you must NOT pick segments from the START(start is around 5% the duration of the video).
- DONT USE the first 1.5 minutes of the video.
- The whole video(highlight) duration must be less than 80 seconds **but must be more then 30 seconds total**. 
- You may pick segments that are not directly one after each other if it still talks about the same topic.
- Try not to choose segments that the content is repeating or doesn't add value to the highlight.
- Make sure the start/end time of each picked segment is not altered/changes from the original segment's.

Return only JSON in this format:
[
  {
    "start": "start time in seconds",
    "content": "actual spoken content (from transcript)",
    "end": "end time in seconds"
  },
  //other segments that directly relate to the same topic
]

- You may reorder segments (in the json's array) if it better tells the story.
- Make SURE that the generated JOSN actually tells the whole story of the topic and is understandable, not repeating.
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


def GetHighlight(Transcription,theme):
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
    total_cost_cached = (
        (tokens_input + token_cached) * pricing[model]["input"] +
        tokens_output * pricing[model]["output"]
    ) / 1000000
    
    total_cost = (
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
        print(f"[ChatGPT]: API call cost: ${total_cost:.6f}")

        json_string = response.output_text
        json_string = json_string.replace("json", "")
        json_string = json_string.replace("```", "")
        
        print(f"[ChatGPT]: response - {json_string}")

#         print(f"[not-ChatGPT]:using local json string for testing..")
#         json_string = """[
#    {
#       "start": "956.736",
#       "content": "רגע לא אמרנו את זה אבל ברגע שמישהו מגיע לחיים האדמים הוא יכול להרוג אנשים אחרים כמו אבוללה אין לו סיבה לעשות את זה אבל הוא יכול אסור להצבן אותם אסור להצבן אותם",
#       "end": "976.96"
#    },
#    {
#       "start": "976.96",
#       "content": "אני אני כן עושים בולקר עושים בונקר אי אפשר שכרגע אין כיוון הוא כל אחד סתם חוצף בלוקים זה טוב שיין יצא אחראי על הדלת אני אביא לך טיפה רדסטון כן סטיקי פיסטונים ככה יש לי פשוט דברים של רדסטון קח זה אמור להספיק",
#       "end": "1017.248"
#    },
#    {
#       "start": "1017.248",
#       "content": "אני רוצה שזה יהיה פרשיר פלייט אני רוצה שזה פרשיר פלייט שנעבור את זה פתח לבד אבל ככה פרשיר פלייט אובר אנגינרד לא לצורך אחי איזה יותר טוב אנחנו מתנסים זה לא טוב",
#       "end": "1037.152"
#    },
#    {
#       "start": "1037.152",
#       "content": "אבל אתה צריך שהוא יפתח לך בשביל לצאת החוצה אתה לא יכול לצאת החוצה אם לא לא חכה חכה חכה חכה אני חשב יש לי רצפה יותר טובה סים סים ככה ככה שתי שורות כזה ככה רגע אני בונה ואני משתמשתי עם כל הבלוקים שהסחקתיים של לבנות איזה יופי נעם אני בונה עם כל הקראפסינטבל שלקחתי להם אחי זה כל כך מצחיק",
#       "end": "1059.44"
#    },
#    {
#       "start": "1129.488",
#       "content": "מה קורה שם מה קרה אני רוצה סגברים למה למה מה עשיתי לכם נשמעת יניב זה עובד אה זה עובד תעזרו לי תעזרו לי אני באה לכם הביתה תתמדדו שאני חסרי לבית אני רוצה הסבר מה קרה זה חשוב מה קרה",
#       "end": "1155.648"
#    },
#    {
#       "start": "1155.648",
#       "content": "ועכשיו היא אדומה רגע אתה עכשיו יכולה להרוג אנשים לא אני לא הורגת אף אחד אין לי אין לי כלום אולימפיק אני ליטרלי רגע מה קרה אני רק תחשבו על זה ככה תחשבו על זה ככה אפשר פשוט להרוג את נוגה אם היא יצאה החוצה אתם יכולים אבל אין לכם למה יש למה יהיה פחות סיכון עליהם",
#       "end": "1168.112"
#    },
#    {
#       "start": "1188.58",
#       "content": "אני בשוק זה כל כך אם אמרתי שזה לא אחד מאיתנו אני כן אפתח שזה יהיה אבל לא שקיע לכם שמח לכם לשנייה רק אומר שם הרבה אנשים מאחוריי אנחנו כנראה נצליח להיפרד מהקוצריות כי אני לא יכול לסמוך על כולם",
#       "end": "1200"
#    }
# ]"""

        res = extract_times(json_string)
        firstItem = res[0]
        if firstItem["end_time"] == firstItem["start_time"]:
            Ask = input("[Chat-GPT]: Error - Get Highlights again[{total_cost}$] (y/n) -> ").lower()
            if Ask == "y":
                res = GetHighlight(Transcription)
        return res
    except Exception as e:
        print(f"[Chat-GPT]: Error in GetHighlight: {e}")
        return 0, 0


if __name__ == "__main__":
    print(GetHighlight(User))
