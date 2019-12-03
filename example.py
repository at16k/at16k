"""
Example usage
"""
from at16k.api import SpeechToText

# One-time initialization
STT = SpeechToText('en_16k') # or en_8k

# Run STT on an audio file, returns a dict
print(STT('./samples/test.wav'))
