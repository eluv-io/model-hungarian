#from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
#import torch
#import torchaudio
from huggingsound import SpeechRecognitionModel

from typing import List

from common_ml.model import VideoModel
from common_ml.tag_formatting import VideoTag

class HungarianSTT(VideoModel):
    def __init__(self):
        """
        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")
        self.model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")
        self.model.eval()
        """
        self.model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")
        
    def tag(self, fpath: str) -> List[VideoTag]:

        transcription = self.model.transcribe([fpath])[0]

        # Get words
        words = transcription['transcription'].split()

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