import os
from typing import List, Tuple
import urllib.request
import tarfile
from glob import glob
import argparse
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO)

datasets = {
    'dev-clean': {
        'url': 'http://www.openslr.org/resources/12/dev-clean.tar.gz',
    },
    'dev-other': {
        'url': 'http://www.openslr.org/resources/12/dev-other.tar.gz',
    },
    'test-clean': {
        'url': 'http://www.openslr.org/resources/12/test-clean.tar.gz',
    },
    'test-other': {
        'url': 'http://www.openslr.org/resources/12/test-other.tar.gz',
    },
    'train-clean-100': {
        'url': 'http://www.openslr.org/resources/12/train-clean-100.tar.gz',
    },
    'train-clean-360': {
        'url': 'http://www.openslr.org/resources/12/train-clean-360.tar.gz',
    },
    'train-other-500': {
        'url': 'http://www.openslr.org/resources/12/train-other-500.tar.gz',
    }
}


def extract_tar(tarfile, dest='.', strip_level=0):
    """
    Extracts a tar file to dest and optionally removes the prefix of files
    :param tarfile: tar file
    :param dest: destination folder
    :param strip_level: remove this number of levels from the compressed filename 
    :return: 
    """
    for member in tarfile.getmembers():
        if member.isreg():
            name_split = member.name.split(os.sep)[strip_level:]
            if not name_split:
                raise ValueError(f'Can not remove {strip_level}'
                                 f' levels from filename: {member.name}')
            member.name = os.path.join(*name_split)
            print(f'Extracting: {member.name}')
            tarfile.extract(member, dest)


@delayed
def convert_flac_to_wav(source: str, dest: str, keep_original=False):
    import librosa
    if os.path.isfile(dest):
        logging.info(
            f"Tried transfer {source} but file already exists {dest}")
   #audio = AudioSegment.from_file(source, 'flac')
   #audio.export(dest, 'wav')

    y, sr = librosa.load(source, sr=16000)
    librosa.output.write_wav(dest, y, sr)
    if not keep_original:
        os.remove(source)


def transcode_flac_wav_recursive(path, keep_original=False, force_overwrite=False, n_jobs: int=4):
    """
    For any MP3 file found in path creates a corresponding WAV file
    :param path: top dir of the dataset
    :param keep_original: if original MP3 file should be kept
    :param n_jobs: number of parallel Joblib tasks
    :return: None if successfull
    """
    out = Parallel(n_jobs=n_jobs, verbose=1)(
        convert_flac_to_wav(
            filename, filename.replace('.flac', '.wav'), keep_original=keep_original)
            for filename in glob(path + '/**/*.flac', recursive=True)
            if (force_overwrite or
                not os.path.isfile(filename.replace('.flac', '.wav'))
                ) # avoid extra work
    )


def read_transcript(filename: str) -> List[Tuple[str, str]]:
    """
    Reads rows and splits them by first space
    :param filename:
    :return: tuples of (name, transcript)
    """
    with open(filename, 'r') as file:
        result = [(line.strip().split(' ', 1)[0],
                  line.strip().split(' ', 1)[1].lower())
                  for line in file.readlines()]
    return result


def create_index_data(ds_name: str, index_dir: str, data_dir: str, n_jobs: int=4) -> pd.DataFrame:
    """
    Creates the index file which is suitable for the pipeline.
    The file contains paths to audiofiles and the transcripts

    :param index_dir: path where the index will be used.
                All paths will be relative to this directory.
    :param data_dir: path of the dataset. May be either absolute or relative
    :param n_jobs: number of parallel Joblib tasks
    :return: pandas DataFrame with path, transcript and file size
    """
    dataset_path = os.path.join(data_dir, ds_name)
    
    @delayed
    def gen_entry(filename, index_dir):
        dirname = os.path.dirname(filename)
        entry = pd.DataFrame(list(read_transcript(filename)), columns=['path', 'transcript'])
        entry.path = entry.path.apply(
            lambda x: os.path.relpath(
                os.path.join(dirname, x + '.wav'), start=index_dir
            )
        )
        return entry
    
    entries = Parallel(n_jobs=n_jobs, verbose=1)(
        gen_entry(filename, index_dir)
        for filename in glob(dataset_path + '/**/*.trans.txt', recursive=True)
    )
    index_data = pd.concat(entries)
    index_data['filesize'] = index_data.path.apply(os.path.getsize)
    return index_data


@delayed
def change_sound_speed(source: str, dest: str, speed: float):
    """
    :param in_filename: filename of the input file
    :param out_filename: filename of the output file
    :param speed: speed of the output file
    """
    import librosa
    import pyrubberband

    y, sr = librosa.load(source, sr=16000)
    yy = pyrubberband.pyrb.time_stretch(
        y, sr, speed, rbargs=None)
    librosa.output.write_wav(dest, yy, sr)


def create_augmentation_data(
        dataset_path: str, speed_augment_by: float = 0.1, force_overwrite=False, n_jobs: int=4):
    """
    For each audio file in the dataset creates its slower and faster version.

    :param dataset_path: path of the dataset. May be either absolute or
                relative
    :param speed_augment_by: speed augment by this number (parts of 1)
    :param force_overwrite: force overwriting of exiting augmentation files
    :param n_jobs: number of parallel Joblib tasks
    """
    out = Parallel(n_jobs=n_jobs, verbose=1)(
        change_sound_speed(
            filename, filename.replace('.wav', '-FAST.wav'), speed=1 + speed_augment_by)
            for filename in glob(dataset_path + '/**/*.wav', recursive=True)
            if (not (filename.endswith('FAST.wav') or filename.endswith('SLOW.wav')) # avoid double augmentation
                and (force_overwrite or
                     not os.path.isfile(filename.replace('.wav', '-FAST.wav'))) # avoid extra work
                )
    )
    out = Parallel(n_jobs=n_jobs, verbose=1)(
        change_sound_speed(
            filename, filename.replace('.wav', '-SLOW.wav'), speed=1 - speed_augment_by)
            for filename in glob(dataset_path + '/**/*.wav', recursive=True)
            if (not (filename.endswith('FAST.wav') or filename.endswith('SLOW.wav'))
                and (force_overwrite or 
                     not os.path.isfile(filename.replace('.wav', '-SLOW.wav')))  # avoid extra work
                )
    )


def extend_index_for_augmentation(index_data) -> pd.DataFrame:
    """
    Takes the index DataFrame and extends it with augmentation data
    :param index: initial index
    :return: new_index
    """
    new_index_data = pd.concat(
        [pd.DataFrame(
            zip(
                index_data.path.apply(lambda x: x.replace('.wav', '-' + aug + '.wav')),
                index_data.transcript
            ),
            columns=['path', 'transcript']
        ) for aug in ('FAST', 'SLOW')],
        ignore_index=True
    )
    new_index_data['filesize'] = new_index_data.path.apply(os.path.getsize)
    return pd.concat([index_data, new_index_data], ignore_index=True)


def main(ds_name: str, index_dir: str, data_dir: str,
         augment: bool = False, force_overwrite:bool = False,
         n_jobs: int=4):
    """
    :param ds_name: name of the dataset to use
    :param index_dir: current work dir, where index will be placed
    :param data_dir: location where the dataset will be placed
    :param augment: if creation of the augmented sound files is requested
    :param n_jobs: nuber of parallel Joblib tasks
    """
    url = datasets[ds_name]['url']

    tar_file_name = url.split('/')[-1]
    audio_data_dir = os.path.join(
        data_dir, tar_file_name.split('.', 1)[0])

    # Prepare main dataset
    if not os.path.isdir(audio_data_dir):
        logging.info(f'Dataset dir: {audio_data_dir} not found')
        if not os.path.isfile(tar_file_name):
            logging.info(f'Not found tar file {tar_file_name}')
            logging.info('Downloading tar file')
            urllib.request.urlretrieve(url, tar_file_name)
            logging.info(f'Successfully downloaded tar file.')

        logging.info(f'Extracting into {data_dir}')
        tar = tarfile.open(tar_file_name, "r:gz")
        extract_tar(tar, dest=data_dir, strip_level=1) # remove the enclosing LibriSpeech folder
        tar.close()

    # Transcode FLAC to WAV and create an index
    logging.info('Transcoding FLAC files')
    transcode_flac_wav_recursive(audio_data_dir, n_jobs=n_jobs)

    if augment:
        logging.info('Creating augmented sound files')
        create_augmentation_data(audio_data_dir, speed_augment_by=0.1, n_jobs=n_jobs)

    logging.info('Generating index data')
    index_data = create_index_data(
        ds_name, index_dir, data_dir, n_jobs=n_jobs)
    if augment:
        index_data = extend_index_for_augmentation(index_data)
    index_data.to_csv(os.path.join(index_dir, f'libri-{ds_name}-index.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare LibriSpeech data')
    parser.add_argument('--type', type=str,
                        help='which dataset to download',
                        default='dev-clean',
                        choices=datasets.keys())
    parser.add_argument('--data_dir', type=str,
                        help='where to place final dataset',
                        default='.')
    parser.add_argument('--index_dir', type=str,
                        help='path relative to which all'
                             'audio paths will be indexed',
                        default='.')
    parser.add_argument('--augment', type=bool,
                        help='if generation of augmented'
                        ' files is requested',
                        default=False)
    parser.add_argument('--force_overwrite', type=bool,
                        help='force overwriting audio files during'
                        ' dataset preparation (slower)',
                        default=False)
    parser.add_argument('--n_jobs', type=int,
                        help='number of parallel Joblib processes',
                        default=4)

    args = parser.parse_args()
    main(args.type, args.index_dir, args.data_dir,
         args.augment, args.force_overwrite, args.n_jobs)

