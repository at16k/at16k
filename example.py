"""
Example usage
"""
from at16k import SpeechToText

# One-time initialization
STT = SpeechToText('en_16k')

# Run STT on an audio file
print(STT('./samples/test.wav'))
