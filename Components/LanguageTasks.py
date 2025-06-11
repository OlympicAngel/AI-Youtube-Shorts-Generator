from typing import List, NotRequired, TypedDict

class ClipSegment(TypedDict):
    start_time: int
    end_time: int
    content: str
    speakers: NotRequired[List[str]]  # Optional field for speaker names

# Function to extract start and end times
def extract_times(json_string) -> List[ClipSegment]:
    import json


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

def GetHighlight(Transcription,theme, test=False):
    from openai import OpenAI
    import tiktoken  # Official tokenizer library
    from dotenv import load_dotenv

    load_dotenv()
    client = OpenAI()

    if test:
        print(f"[not-ChatGPT]:using local json string for testing..")
        json_string = """[
  {
    "start": "254.75",
    "content": "כי זה בגדול ריק ומורטי זה סנדבוקס ענק שאפשר לעשות בו מלא שטויות וליהנות",
    "end": "259.83"
  },
  {
    "start": "268.99",
    "content": "כי זה הערך העליון בריק ומורטי לקחת כל מיני דברים מגניבים",
    "end": "272.27"
  },
  {
    "start": "272.27",
    "content": "מפופ קולצר מסייפיי כל הדברים הגיקים המגניבים האלה",
    "end": "276.19"
  },
  {
    "start": "276.19",
    "content": "ולחבר אותם לכיף",
    "end": "277.95"
  },
  {
    "start": "286.59",
    "content": "כמו מורטי מיינדבלוורס קייבל טבעי 12 טוטל ריקול",
    "end": "290.31"
  },
  {
    "start": "290.47",
    "content": "מה זה הפרקים האלה תכלס",
    "end": "291.55"
  },
  {
    "start": "291.55",
    "content": "מוצאים איזה תירוץ שנותן להם פשוט להראות לנו מלאמלא רילזים בגדול",
    "end": "296.47"
  },
  {
    "start": "296.47",
    "content": "פרקים של דקה",
    "end": "297.67"
  },
  {
    "start": "297.67",
    "content": "המוח שלהם עובד בתדרים האלה לא התכנון הכבד הספונטני",
    "end": "301.71"
  },
  {
    "start": "484.77",
    "content": "שזה יותר זרם תודעה רפרנסים וצחוקים",
    "end": "488.21"
  },
  {
    "start": "511.19",
    "content": "סדרה ענקית כיף גדול מושלמת לשכתא פשוט",
    "end": "514.35"
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
        json_string = json_string.replace("```", "").split("]")[0] + "]" # sometimes gpt will output after the json
        
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
    import os

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