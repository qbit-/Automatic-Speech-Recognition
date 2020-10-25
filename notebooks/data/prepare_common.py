import os
import pandas as pd
import argparse
import logging
import sys
from pydub import AudioSegment
import librosa
import pyrubberband
from tqdm import tqdm
import soundfile as sf


datasets = {
    'validated',
    'invalidated',
    'reported',
    'train',
    'test',
    'dev'
}


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
    sf.write(dest, yy, sr)

    
def create_augmentation_data(clips, wav_dir: str, speed_augment_by: float = 0.1):
    augmented_clips = []
    for i, row in tqdm(clips.iterrows()):
        file_name = row.path.split('.')[0]
        if f'{file_name}-FAST.wav' not in os.listdir(wav_dir):
            change_sound_speed(row.path, f'{file_name}-FAST.wav', 1 + speed_augment_by)
        if f'{file_name}-SLOW.wav' not in os.listdir(wav_dir):
            change_sound_speed(row.path, f'{file_name}-SLOW.wav', 1 - speed_augment_by)
        augmented_clips.append({'path': f'{file_name}-FAST.wav', 'sentence': row.sentence})
        augmented_clips.append({'path': f'{file_name}-SLOW.wav', 'sentence': row.sentence})
        
    return pd.DataFrame.from_records(augmented_clips)


def create_index_data(df, wav_dir: str) -> pd.DataFrame:
    """
    Creates the index file which is suitable for the pipeline.
    The file contains paths to audiofiles and the transcripts

    :param index_path: current work directory.
                All paths will be relative to this directory.
    :param wav_dir: path to wav dataset
    :return: pandas DataFrame with path, transcript and file size
    """
    index_data = pd.DataFrame(
        zip(df.path, df.sentence),
        columns=['path', 'transcript'])
    index_data['filesize'] = index_data.path.apply(os.path.getsize)

    return index_data


def transcode_mp3_wav(clips, data_dir, wav_dir):
    print(wav_dir)
    if not os.path.isdir(wav_dir):
        os.mkdir(wav_dir)
    for i, row in tqdm(clips.iterrows()):
        wav_name = row.path.split('.')[0] + '.wav'
        if wav_name not in os.listdir(wav_dir):
            sound = AudioSegment.from_mp3(os.path.join(data_dir, 'clips', row.path))
            sound.export(os.path.join(wav_dir, wav_name), format="wav")

        
def main(ds_name: str, index_dir: str, data_dir: str, wav_dir: str, augment: bool = False):
    """
    :param ds_name: name of the dataset to use
    :param index_dir: current work dir, where index will be placed
    :param data_dir: location where the original dataset is stored on disk
    :param data_dir: where to store wav files
    :param augment: if creation of the augmented sound files is requested
    """
    
    ds_csv = os.path.join(data_dir, ds_name + '.tsv')
    

    # Check if dataset exists on disk
    if not os.path.isfile(ds_csv):
        print(f'Could not find file {ds_csv}.'
                         ' Not able to download yet.')
        sys.exit(1)
        
    clips = pd.read_csv(ds_csv, sep='\t')

    # Convert mp3 to wav
    logging.info('Transcoding MP3 files')
    transcode_mp3_wav(clips, data_dir=data_dir, wav_dir=wav_dir)
    clips['path'] = clips.path.apply(lambda f: os.path.join(wav_dir, f.split('.')[0]+'.wav'))
    print(len(clips))
    
    if augment:
        logging.info('Creating augmented sound files')
        augmented_clips = create_augmentation_data(clips, wav_dir, speed_augment_by=0.1)
        clips = pd.concat([clips, augmented_clips], ignore_index=True)
    print(len(clips))

    logging.info('Generating index data')
    index_data = create_index_data(clips, wav_dir)
    index_data.to_csv(os.path.join(index_dir, f'{ds_name}-index-commonvoice.csv'), index=False)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare CommonVoice data')
    parser.add_argument('--type', type=str,
                        help='which dataset to download',
                        default='dev',
                        choices=datasets)
    parser.add_argument('--data_dir', type=str,
                        help='where the original dataset is stored on disk',
                        default='.')
    parser.add_argument('--index_dir', type=str,
                        help='path relative to which all'
                             'audio paths will be indexed',
                        default='.')
    parser.add_argument('--wav_dir', type=str,
                        help='path to store wav files',
                        default='./common_voice_wavs')
    parser.add_argument('--augment', type=bool,
                        help='if generation of augmented'
                        ' files is requested',
                        default=False)

    args = parser.parse_args()
    main(args.type, args.index_dir, args.data_dir, args.wav_dir, args.augment)