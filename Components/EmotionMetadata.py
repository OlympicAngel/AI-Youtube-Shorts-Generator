from typing import List, TypedDict, Unpack
from Components.SpeakersMetadata import TranscribeSegmentType_withSpeakers

TranscribeSegmentType_withSpeakersAndSentiment = tuple[Unpack[TranscribeSegmentType_withSpeakers], str]

prefix = "[EmotionsMetadata]: "

class SentimentResult(TypedDict):
    label: str
    score: float

def add_hebrew_sentiment(transcriptions: List[TranscribeSegmentType_withSpeakers]) -> List[TranscribeSegmentType_withSpeakersAndSentiment]:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import tqdm

    print(prefix + "Adding Hebrew sentiment analysis to transcriptions...")

    # Load Hebrew sentiment model
    model_name = "dicta-il/dictabert-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)

    full_texts = [seg[0] for seg in transcriptions]
    max_chars = 512
    min_chars_needed = 60

    contexts = []
    meta:List[tuple[str,float,float,List[str]]] = []

    for i, segment in tqdm.tqdm(enumerate(transcriptions), total=len(transcriptions)):
        content, t_start, t_end, speakers = segment

        context_parts = [content]
        context_len = len(content)

        max_offset = 4  # 2 previous + 2 next
        offset = 1
        directions_used = {"prev": 0, "next": 0}

        while context_len < min_chars_needed and offset <= max_offset:
            if offset % 2 == 1:  # prev
                idx = i - (offset + 1) // 2
                if directions_used["prev"] < 2 and idx >= 0:
                    text = full_texts[idx]
                    context_parts.insert(0, text)
                    context_len += len(text)
                    directions_used["prev"] += 1
            else:  # next
                idx = i + offset // 2
                if directions_used["next"] < 2 and idx < len(full_texts):
                    text = full_texts[idx]
                    context_parts.append(text)
                    context_len += len(text)
                    directions_used["next"] += 1
            offset += 1

        context = " ".join(context_parts)[:max_chars]
        contexts.append(context)
        meta.append((content, t_start, t_end, speakers))

    # Batch sentiment analysis
    predictions = sentiment_analysis(contexts, batch_size=32)

    # Map results to sentiment tags
    label_map = {'Neutral': 'neu', 'Negative': 'neg', 'Positive': 'pos'}
    results = [
        (content, t_start, t_end, speakers, label_map[pred['label']])
        for (content, t_start, t_end, speakers), pred in zip(meta, predictions)
    ]

    print(prefix + "done.")
    return results
