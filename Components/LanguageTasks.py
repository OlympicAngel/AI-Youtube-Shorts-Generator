import os
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
    "start": "195.32",
    "content": "אחי חיות זה נדיר יצח אני פשוט שאשכרה מעצמנו",
    "end": "197.5"
  },
  {
    "start": "246.06",
    "content": "קדימה רדנק אני סומך עליך שלא תמותי בשניה שנגיע לשם איכשהו",
    "end": "249.26"
  },
  {
    "start": "253.38",
    "content": "קרנבולת אני קראתי לה רדנק היא כל הכח איתית אוקיי סליחה אמרתי",
    "end": "256.78"
  },
  {
    "start": "263.98",
    "content": "קדימה בנות קדימה",
    "end": "266.78"
  },
  {
    "start": "266.78",
    "content": "רדנקי יאללה קדימה",
    "end": "268.42"
  },
  {
    "start": "275.94",
    "content": "מיצי זה הכבשה",
    "end": "277.3"
  },
  {
    "start": "337.46",
    "content": "יש מיצי קטן",
    "end": "338.9"
  },
  {
    "start": "339.62",
    "content": "יואו זה מיני מיצי",
    "end": "340.82"
  },
  {
    "start": "342.78",
    "content": "זה אופר מיילי אקספרס",
    "end": "343.78"
  },
  {
    "start": "343.78",
    "content": "מיצי בואי",
    "end": "344.46"
  },
  {
    "start": "354.42",
    "content": "שלב שני שייני את הייתי",
    "end": "355.1"
  },
  {
    "start": "355.1",
    "content": "אה כן",
    "end": "356.5"
  },
  {
    "start": "356.5",
    "content": "אנחנו צריכים להשיג פאמקינג",
    "end": "356.94"
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
    system = readPromptFile("instruction").replace("<<<THEME PROMPT>>>", theme)

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
        json_string = json_string.replace("```", "").split("]")[0] # sometimes gpt will output after the json
        
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


def readPromptFile(filename):
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # <root>
        filepath = os.path.join(base_dir, 'prompts', f'{filename}.txt')

        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except UnicodeDecodeError:
        return "Error: Could not decode file with UTF-8 encoding."