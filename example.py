"""
Example usage
"""
from at16k.api import SpeechToText


def main():
    """
    Main
    """
    # One-time initialization
    convert = SpeechToText('en_16k')  # or en_8k

    # Run STT on an audio file, returns a dict
    results = convert('./samples/test_16k.wav')
    print('Text:', results['text'])


if __name__ == '__main__':
    main()
