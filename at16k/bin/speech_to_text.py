"""
Speech to text converter (command-line)
"""
import argparse
from at16k.api import SpeechToText, LiveSpeechToText

PARSER = argparse.ArgumentParser('at16k Speech-to-Text')
PARSER.add_argument('-m', '--model', type=str,
                    required=True, choices=['en_8k', 'en_16k', 'en_16k_rnnt'])
PARSER.add_argument('-i', '--input', type=str,
                    help='Input WAV file. Optional, if using en_16k_rnnt model, else mandatory.')
PARSER.add_argument('-d', '--decode', type=str, choices=['beam', 'greedy'], default='beam',
                    help='Applies only when using en_16k_rnnt model. Beam will be slower but more accurate.')
FLAGS = PARSER.parse_args()


def convert_live_from_file(model, args):
    text = None
    faster = False if args.decode == 'beam' else True
    stt = LiveSpeechToText(model_name=model, faster=faster)
    for result in stt.from_file(args.input):
        text = result['text']
        print('Intermediate results:', text, end="\r", flush=True)
    return text


def convert_live_from_microphone(model, args):
    from at16k.core.microphone import MicrophoneStream
    faster = False if args.decode == 'beam' else True
    stt = LiveSpeechToText(model_name=model, faster=faster)
    text = None
    context = None
    with MicrophoneStream() as stream:
        for chunk in stream.generator():
            if chunk:
                text, context = stt.from_buffer(chunk, context, dtype='<i2')
                print('Intermediate results:', text, end="\r", flush=True)
    return text


def convert_offline_from_file(model, args):
    assert args.input, 'Please specify input file (-i). See help for more details'
    stt = SpeechToText(model)
    result = stt(args.input)
    text = result['text']
    return text


def main():
    """
    Main
    """
    model = FLAGS.model
    if model in ['en_16k_rnnt']:
        if FLAGS.input:
            text = convert_live_from_file(model, FLAGS)
        else:
            text = convert_live_from_microphone(model, FLAGS)
    else:
        text = convert_offline_from_file(model, FLAGS)
    print('-' * 100)
    print('Result:', text)
    print('-' * 100)


if __name__ == '__main__':
    main()
