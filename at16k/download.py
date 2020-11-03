import os
import shutil
import urllib.request
from urllib.parse import urljoin
from pathlib import Path
import sys
import progressbar

BASE_URL = 'https://storage.googleapis.com/at16k-ce/models'
AVAILABLE_MODELS = ['en_8k', 'en_16k', 'en_16k_rnnt']

PROGRESS_BAR = None


def show_progress(block_num, block_size, total_size):
    global PROGRESS_BAR
    if PROGRESS_BAR is None:
        PROGRESS_BAR = progressbar.ProgressBar(maxval=total_size)
        PROGRESS_BAR.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        PROGRESS_BAR.update(downloaded)
    else:
        PROGRESS_BAR.finish()
        PROGRESS_BAR = None


def setup_home():
    """
    Init home directory to store assets and models
    """
    if 'AT16K_RESOURCES_DIR' in os.environ:
        at16k_model_dir = os.environ['AT16K_RESOURCES_DIR']
    else:
        home_dir = str(Path.home())
        at16k_model_dir = os.path.join(home_dir, '.at16k')
    if not os.path.exists(at16k_model_dir):
        os.mkdir(at16k_model_dir)
    return at16k_model_dir


def download_model(remote_path, local_path):
    """
    Download file
    """
    if not os.path.exists(local_path):
        print('Downloading from %s' % remote_path)
        urllib.request.urlretrieve(remote_path, local_path, show_progress)


def unarchive(local_path, base_dir):
    """
    Unarchive zipped file
    """
    shutil.unpack_archive(local_path, base_dir)


def main():
    """
    Main
    """
    assert len(
        sys.argv) > 1, ('Please specify model name: one of en_8k, en_16k, en_16k_rnnt, all')
    name = sys.argv[1]
    if name == 'all':
        name = AVAILABLE_MODELS
    else:
        assert name in AVAILABLE_MODELS, (
            'Please specify a valid model name: one of en_8k, en_16k, en16k_rnnt, all')
        name = [name]
    base_dir = setup_home()
    for item in name:
        file_name = '%s.tar.gz' % item
        remote_path = BASE_URL + '/' + file_name
        local_path = os.path.join(base_dir, file_name)
        download_model(remote_path, local_path)
        unarchive(local_path, base_dir)
        os.remove(local_path)
        print('Downloaded model: %s' % item)


if __name__ == '__main__':
    main()
