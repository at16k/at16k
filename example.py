"""
Example usage
"""
from at16k.api.speech_to_text import SpeechToText

# One-time initialization
STT = SpeechToText('en_16k')

# Run STT on an audio file
print(STT('./samples/test.wav'))
