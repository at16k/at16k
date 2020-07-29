"""
Example usage
"""
from at16k.api import LiveSpeechToText
from at16k.core.microphone import MicrophoneStream


def main():
    """
    Main
    """
    # One-time initialization
    convert = LiveSpeechToText('en_16k_rnnt', faster=False)

    # Initialize context and microphone stream
    context = None
    wav_data = b''
    with MicrophoneStream() as stream:
        for chunk in stream.generator():
            if chunk:
                text, context = convert.from_buffer(chunk, context, dtype='<i2')
                print(text, end="\r", flush=True)

    print('\n\nFinal Result:', text)


if __name__ == '__main__':
    main()
