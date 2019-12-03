"""
Speech to text converter (command-line)
"""
import argparse
from at16k.api import SpeechToText

PARSER = argparse.ArgumentParser('at16k Speech-to-Text')
PARSER.add_argument('-i', '--input', type=str,
                    required=True, help='Input file (wav)')
PARSER.add_argument('-m', '--model', type=str,
                    required=True, choices=['en_8k', 'en_16k'])
FLAGS = PARSER.parse_args()

def main():
    """
    Main
    """
    stt = SpeechToText(FLAGS.model)
    result = stt(FLAGS.input)
    print('-'*100)
    print('Output:', result['text'])
    print('-'*100)


if __name__ == '__main__':
    main()
