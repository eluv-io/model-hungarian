import requests
from typing import List

import ollama
from huggingsound import SpeechRecognitionModel

from common_ml.model import VideoModel
from common_ml.tag_formatting import VideoTag

from config import config

class HungarianSTT(VideoModel):
    def __init__(self):
        """
        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")
        self.model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")
        self.model.eval()
        """
        self.model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian", device=config["device"])
        self.translator = ollama.Client(config["llama"])
        
    def tag(self, fpath: str) -> List[VideoTag]:

        transcription = self.model.transcribe([fpath])[0]

        # Get words
        words = transcription['transcription']

        prompt = f"Translate the following Hungarian text to English, considering it's from a soccer game context:\n\n\"{words}\""

        response = self.translator.generate(
                model="llama:70b",
                stream=False,
                prompt=prompt,
                options={'seed': 1, "temperature": 0.0})

        print(response)

        return []

        # Use ollama to translate to English
        # words = self.translate_to_english(words)

        audio_duration = transcription['end_timestamps'][-1]

        # Uniformly spread timestamps
        n_words = len(words)
        seconds_per_word = audio_duration / n_words

        # Assign timestamps
        timestamps = []
        for idx, word in enumerate(words):
            start_time = idx * seconds_per_word
            end_time = (idx + 1) * seconds_per_word
            timestamps.append((word, start_time, end_time))

        tags = []
        for word, st, et in timestamps:
            tags.append(VideoTag(
                start_time=st,
                end_time=et,
                text=word,
            ))

        return tags
    