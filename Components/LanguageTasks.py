
from typing import List, Optional
from pydantic import BaseModel, TypeAdapter

class ClipSegment(BaseModel):
    start_time: float
    end_time: float
    content: str
    
class GPT_SelectedSegments(BaseModel):
    highlightTitle:str
    selectedSegments: list[ClipSegment]

def GetHighlight(Transcription,theme, test=False):
    from openai import OpenAI
    import tiktoken  # Official tokenizer library
    from dotenv import load_dotenv

    load_dotenv()
    client = OpenAI()

    if test:
        print(f"[not-ChatGPT]:using local json string for testing..")
        adapter = TypeAdapter(List[ClipSegment])

        return adapter.validate_python([ { "start_time": 195, "end_time": 197.52, "content": "אחי חיות זה נדיר יצא אני פשוט שאשכרה מעצמנו", "speaker_ids": [ "#06" ] }, { "start_time": 208.28, "end_time": 208.84, "content": "בואי מיצי", "speaker_ids": [ "#06" ] }, { "start_time": 220.76, "end_time": 222.92, "content": "אנחנו צריכים לתת לשם אנחנו נקרא לה רדנק", "speaker_ids": [ "#06" ] }, { "start_time": 223.88, "end_time": 226.56, "content": "יש שמה הדבר סירה", "speaker_ids": [ "#06" ] }, { "start_time": 246, "end_time": 249.16, "content": "קדימה רדניק אני סומך עליך שלא תמותי בשנייה שנגיע לשם איכשהו", "speaker_ids": [ "#06" ] }, { "start_time": 250.16, "end_time": 252.12, "content": "רגע למה לא הבאת איתך כבשה", "speaker_ids": [ "#04" ] }, { "start_time": 252.12, "end_time": 252.8, "content": "הבאתי רדניק", "speaker_ids": [ "#04" ] }, { "start_time": 253.36, "end_time": 254.76, "content": "פרניבולט קראתי לה רדניק", "speaker_ids": [ "#04", "#06" ] }, { "start_time": 264.96, "end_time": 268.4, "content": "קדימה בנות קדימה רדנקי יאללה קדימה", "speaker_ids": [ "#06" ] }, { "start_time": 268.4, "end_time": 271.08, "content": "לא רשר פלט מטומטם למה עשינו את זה פרק קודם", "speaker_ids": [ "#06" ] }, { "start_time": 273.32, "end_time": 274.64, "content": "תשארי פה רדנק בואי לפה", "speaker_ids": [ "#06", "#04" ] }, { "start_time": 274.64, "end_time": 276, "content": "מיצי זה הכבשה", "speaker_ids": [ "#06", "#04" ] }, { "start_time": 277.28, "end_time": 279.6, "content": "אני שונא את הפרשר פלט המטומטם שלך שיינים", "speaker_ids": [ "#06" ] }, { "start_time": 281.24, "end_time": 283.04, "content": "איפה מיצי תקרא לה", "speaker_ids": [ "#06" ] }, { "start_time": 283.44, "end_time": 284.36, "content": "על הגג", "speaker_ids": [ "#06", "#04" ] }, { "start_time": 284.36, "end_time": 286.48, "content": "מיצי על הגג לא", "speaker_ids": [ "#06" ] } ]
        )

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
    tokens_output = 14000 if isReasoning else 800  # estimate/completion length
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

        response = client.responses.parse(
          model=model,
          max_output_tokens =  1500, # limit the output to 20k tokens
          instructions=system,
          input=Transcription,
          text_format= GPT_SelectedSegments
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

        res = response.output_parsed.selectedSegments
        
        print(f"[ChatGPT]: response - {len(res)}")

        if res == None:
            raise RuntimeError("Failed to generate short transcript.")
        
        print(f"AI picked {len(res)} segments")
        
        return res
    except Exception as e:
        raise RuntimeError(f"[Chat-GPT]: Error in GetHighlight: {e}")

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