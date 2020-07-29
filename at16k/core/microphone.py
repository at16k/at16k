"""
Microphone handler
"""
import sys
import signal

try:
    import pyaudio
except ImportError as error:
    print('*' * 100)
    print(
        'Error: Please install pyaudio to use the microphone\nTry installing via pip. For example, pip install pyaudio')
    print('*' * 100)
    sys.exit(1)

from six.moves import queue


class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate=16000, chunk=4096):
        self._rate = rate
        self._chunk = chunk
        self._audio_interface = None
        self._audio_stream = None
        self._buff = queue.Queue()
        self.closed = True
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )

        self.closed = False
        print('-' * 100)
        print("You're on speaker! Start talking...\n(Press Ctrl-C to stop recording.)")
        print('-' * 100)
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            chunk_data = b''.join(data)
            yield chunk_data
