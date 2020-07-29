"""
Example usage
"""
from at16k.api import LiveSpeechToText


def main():
    """
    Main
    """
    # One-time initialization
    convert = LiveSpeechToText('en_16k_rnnt', faster=True)

    text = None
    for result in convert.from_file('./samples/test_16k.wav'):
        text = result['text']
        print(text, end="\r", flush=True)
    print('Final result:', text)


if __name__ == '__main__':
    main()
