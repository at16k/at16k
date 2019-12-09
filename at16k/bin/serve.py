"""
REST API server
"""
import os
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from at16k.api import SpeechToText

APP = Flask(__name__)
CORS(APP)


def _get_args():
    parser = argparse.ArgumentParser(description='Example API')
    parser.add_argument('-p', '--port', type=int,
                        default=9999, help='Port number')
    parser.add_argument('-m', '--model', type=str,
                        default='en_16k', help='Model name')
    args = parser.parse_args()
    return args


@APP.route('/', methods=['GET'])
def index():
    """
    Heartbeat
    """
    return 'OK'


@APP.route('/speechtotext', methods=['POST'])
def file_upload():
    """
    Handle file upload
    """
    try:
        file = request.files.get('file', None)
        if file is None:
            raise Exception('file missing')
        file_name = secure_filename(file.filename)
        file_path = os.path.abspath(os.path.join(UPLOAD_DIR, file_name))
        file.save(file_path)
        results = STT_MODEL(file_path)
        return jsonify(results)
    except Exception as error:
        message = str(error)
        response = jsonify({"message": message})
        response.status_code = 500
        return response


ARGS = _get_args()

UPLOAD_DIR = './uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

STT_MODEL = SpeechToText(ARGS.model)

print('Ready to serve.')

HTTP_SERVER = WSGIServer(('', ARGS.port), APP)
HTTP_SERVER.serve_forever()


def main():
    """
    Main
    """
    # Do nothing


if __name__ == '__main__':
    main()
