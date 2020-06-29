import os
from typing import List, Tuple
import urllib.request
import tarfile
import argparse
import logging
import pandas as pd
from tqdm import tqdm

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


def write_transcript(filename: str,
                     items: List[Tuple[str, str]]):
    """
    Writes the database file
    :param filename: filename of the database
    :param items: tuples of the local filename and transcripts
    """
    with open(filename, 'w') as file:
        result = '\n'.join(
            '{} {}'.format(
                name.strip(),
                transcript.strip().upper())
            for name, transcript in items)
        file.write(result)


def convert_flac_to_wav(source: str, dest: str):
    import librosa
    if os.path.isfile(dest):
        logging.info(
            f"Tried transfer {source} but file already exists {dest}")
   #audio = AudioSegment.from_file(source, 'flac')
   #audio.export(dest, 'wav')

    y, sr = librosa.load(source, sr=16000)
    librosa.output.write_wav(dest, y, sr)


def transfer_transcripted_audio(
        search_folder: str, dataset_folder: str) -> List[Tuple[str, str]]:
    """
    Recursive procedure goes over all files in search folder.
    If it meets *.trans.txt then it adds all of the files to
    dataset in dataset_folder.
    If it meets another folder then recursively executes on it.
    :param search_folder:
    :param dataset_folder:
    :return: transcripts
    """
    transcripts = []
    file_sizes = []
    for file in os.listdir(search_folder):
        if os.path.isdir(f'{search_folder}/{file}'):
            # Go recursive
            rec_transcripts, rec_filesizes = transfer_transcripted_audio(
                f'{search_folder}/{file}', dataset_folder)
            transcripts.extend(rec_transcripts)
            file_sizes.extend(rec_filesizes)
        elif file.endswith('.trans.txt'):
            # Read transcript and move each audio to final folder
            # converting in the same time
            current_transcript = read_transcript(
                f'{search_folder}/{file}')
            for audio_name, _ in current_transcript:
                convert_flac_to_wav(
                    f'{search_folder}/{audio_name}.flac',
                    f'{dataset_folder}/{audio_name}.wav')
                file_sizes.append(
                    os.path.getsize(f'{dataset_folder}/{audio_name}.wav'))
            transcripts.extend(current_transcript)
    return transcripts, file_sizes


def create_index_data(index_path: str, dataset_path: str) -> pd.DataFrame:
    """
    Creates the index file which is suitable for the pipeline.
    The file contains paths to audiofiles and the transcripts

    :param index_path: current work directory.
                All paths will be relative to this directory.
    :param dataset_path: path of the dataset. May be either absolute or
                relative
    :return: pandas DataFrame with path, transcript and file size
    """

    def walk_dirs(current_folder: str):
        file_paths = []
        transcripts = []
        file_sizes = []
        for item in os.listdir(current_folder):
            if os.path.isdir(os.path.join(current_folder, item)):
                # Go recursive
                (item_file_paths,
                 item_transcripts, item_filesizes) = walk_dirs(
                     os.path.join(current_folder, item))

                file_paths.extend(item_file_paths)
                transcripts.extend(item_transcripts)
                file_sizes.extend(item_filesizes)

            elif item.endswith('.trans.txt'):
                # Read transcript
                item_file_paths, item_transcripts = zip(*read_transcript(
                    os.path.join(current_folder, item)))
                transcripts.extend(item_transcripts)

                for file_name in item_file_paths:
                    file_paths.append(
                        os.path.relpath(
                            os.path.join(
                                current_folder, f'{file_name}.wav'),
                            start=index_path)
                        )
                    file_sizes.append(
                        os.path.getsize(
                            os.path.join(
                                current_folder, f'{file_name}.wav')
                        )
                    )
        return file_paths, transcripts, file_sizes

    file_paths, transcripts, file_sizes = walk_dirs(dataset_path)
    index_data = pd.DataFrame(
        zip(file_paths, transcripts, file_sizes),
        columns=['path', 'transcript', 'filesize'])

    return index_data


def transcode_flac_wav_recursive(path, keep_original=False):
    """
    For any FLAC file found in path creates a corresponding WAV file
    :param path: top dir of the dataset
    :param keep_original: if original FLAC file should be kept
    :return: None if successfull
    """
    for item in sorted(os.listdir(path)):
        current_path = os.path.join(path, item)
        if os.path.isdir(current_path):
            print(f'Current: {current_path}')
            transcode_flac_wav_recursive(current_path, keep_original=keep_original)
        elif item.endswith('.flac'):
            convert_flac_to_wav(
                current_path,
                os.path.join(path, os.path.splitext(item)[0] + '.wav')
            )
            if not keep_original:
                os.remove(current_path)


def create_augmentation_data(
        dataset_path: str, speed_augment_by: float = 0.1):
    """
    For each audio file in the dataset creates its slower and faster version.

    :param dataset_path: path of the dataset. May be either absolute or
                relative
    """
    def walk_dirs(current_folder: str):
        file_paths = []
        for item in sorted(os.listdir(current_folder)):
            if os.path.isdir(os.path.join(current_folder, item)):
                # Go recursive
                item_file_paths = walk_dirs(
                     os.path.join(current_folder, item))

                file_paths.extend(item_file_paths)

            elif item.endswith('.trans.txt'):
                db_filename = os.path.join(current_folder, item)
                print(f"Current: {db_filename}")

                # Read transcript
                item_file_paths, transcripts = zip(*read_transcript(
                    db_filename))

                # generate augmented sound files
                additional_file_paths = []
                additional_transcripts = []
                for file_name, transcript in tqdm(zip(
                        item_file_paths, transcripts)):
                    if (file_name.endswith('FAST') or
                        file_name.endswith('SLOW') or
                        f'{file_name}-FAST' in item_file_paths
                        or f'{file_name}-SLOW' in item_file_paths):
                        continue
                    in_filename = os.path.join(
                                current_folder, f'{file_name}.wav')
                    file_paths.append(in_filename)

                    out_filename_f = os.path.join(
                                current_folder, f'{file_name}-FAST.wav')
                    change_sound_speed(in_filename, out_filename_f,
                                       1 + speed_augment_by)
                    additional_file_paths.append(f'{file_name}-FAST')
                    additional_transcripts.append(transcript)
                    file_paths.append(out_filename_f)

                    out_filename_s = os.path.join(
                                current_folder, f'{file_name}-SLOW.wav')
                    change_sound_speed(in_filename, out_filename_s,
                                       1 - speed_augment_by)
                    additional_file_paths.append(f'{file_name}-SLOW')
                    additional_transcripts.append(transcript)
                    file_paths.append(out_filename_s)

                # write updated transcripts
                write_transcript(
                    db_filename,
                    list(zip(item_file_paths+tuple(additional_file_paths),
                             transcripts+tuple(additional_transcripts)))
                )

        return file_paths

    file_paths = walk_dirs(dataset_path)

    return file_paths


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


def main(ds_name: str, index_dir: str, data_dir: str, augment: bool = False):
    """
    :param ds_name: name of the dataset to use
    :param index_dir: current work dir, where index will be placed
    :param data_dir: location where the dataset will be placed
    :param augment: if creation of the augmented sound files is requested
    """
    url = datasets[ds_name]['url']

    tar_file_name = url.split('/')[-1]
    audio_data_dir = tar_file_name.split('.', 1)[0]

    # Prepare main dataset
    if not os.path.isdir(
            os.path.join(data_dir, 'LibriSpeech', audio_data_dir)):
        if not os.path.isfile(tar_file_name):
            logging.info(f'Not found tar file {tar_file_name}.'
                         ' Downloading it.')
            urllib.request.urlretrieve(url, tar_file_name)
            logging.info(f'Successfully downloaded tar file.')

        logging.info(f'Extracting into {data_dir}')
        tar = tarfile.open(tar_file_name, "r:gz")
        tar.extractall(data_dir)
        tar.close()

    # Transcode FLAC to WAV and create an index
    logging.info('Transcoding FLAC files')
    transcode_flac_wav_recursive(
        os.path.join(data_dir, 'LibriSpeech', audio_data_dir))

    if augment:
        logging.info('Creating augmented sound files')
        create_augmentation_data(
            os.path.join(data_dir, 'LibriSpeech', audio_data_dir),
            speed_augment_by=0.1)

    logging.info('Generating index data')
    index_data = create_index_data(
        index_dir, os.path.join(data_dir, 'LibriSpeech', audio_data_dir))
    index_data.to_csv(os.path.join(index_dir, f'{ds_name}-index.csv'))


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

    args = parser.parse_args()
    main(args.type, args.index_dir, args.data_dir, args.augment)
