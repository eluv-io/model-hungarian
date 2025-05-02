from typing import List
import json

import ollama
from huggingsound import SpeechRecognitionModel
from loguru import logger

from common_ml.model import VideoModel
from common_ml.tag_formatting import VideoTag

from config import config

class HungarianSTT(VideoModel):
    def __init__(self, llama_model: str, prompt: str):
        self.model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian", device=config["device"])
        self.translator = ollama.Client(config["llama"])
        self.llama_model = llama_model
        self.prompt = prompt
        
    def tag(self, fpath: str) -> List[VideoTag]:

        transcription = self.model.transcribe([fpath])[0]

        # Get words
        words = transcription['transcription']

        prompt = f"{self.prompt}\n" + words + "\nOutput your response in the following format: {\"translation\": translated_text}. Do not output anything else."
        
        response = self.translator.generate(
                model=self.llama_model,
                stream=False,
                prompt=prompt,
                options={'seed': 1, "temperature": 0.0})["response"]

        try:
            response = response[response.index("{"):response.index("}") + 1]
            response = json.loads(response)
            words = response['translation']
        except Exception as e:
            logger.error(f"Error parsing translation response: {e}")
            return []
        
        if not words:
            logger.debug("No words found in transcription.")
            return []
        
        words = words.split(' ')

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
    