import glob
import h5py
import numpy as np
import webrtcvad
from pydub import AudioSegment

from core.common import DATA_FOLDER, SAMPLE_CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE, FRAME_SIZE

OBJ_PREPARE_AUDIO = True


class FileManager:
    '''
    Keeps track of audio-files from a data-set.
    Provides support for formatting the wav-files into a desired format.
    Also provides support for conversion of .flac files (as we have in the LibriSpeech data-set).
    '''

    def __init__(self, name, directory):

        self.name = name
        self.data = h5py.File(DATA_FOLDER + '/' + name + '.hdf5', 'a')

        # Setup file names.
        if ('files' not in self.data) or (self.data['files'].shape[0]==0):

            # Get files.
            files = glob.glob(directory + '/**/*.wav', recursive=True)
            files.extend(glob.glob(directory + '/**/*.flac', recursive=True))
            files = [f for f in files]

            # Setup data set.
            dt = h5py.special_dtype(vlen=str)
            self.data.create_dataset('files', (len(files),), dtype=dt)

            # Add file names.
            for i, f in enumerate(files):
                self.data['files'][i] = f

    def get_track_count(self):
        return len(self.data['files'])

    def prepare_files(self, normalize=False):
        '''
        Prepares the files for the project.
        Will do the following check for each file:
        1. Check if it has been converted already to the desired format.
        2. Converts all files to WAV with the desired properties.
        3. Stores the converted files in a separate folder.
        '''

        if not OBJ_PREPARE_AUDIO:
            print(f'Skipping check for {self.name}.')
            return

        print('Found {0} tracks to check.'.format(self.get_track_count()))
        progress = 1

        # Setup raw data set.
        if 'raw' not in self.data:
            dt = h5py.special_dtype(vlen=np.dtype(np.int16))
            self.data.create_dataset('raw', (self.get_track_count(),), dtype=dt)

        # Convert files to desired format and save raw content.
        for i, file in enumerate(self.data['files']):

            print('Processing {0} of {1}'.format(progress, self.get_track_count()), end='\r', flush=True)
            progress += 1

            # Already converted?
            if len(self.data['raw'][i]) > 0:
                continue

            # Convert file.
            try:
                track = (AudioSegment.from_file(file)
                         .set_frame_rate(SAMPLE_RATE)
                         .set_sample_width(SAMPLE_WIDTH)
                         .set_channels(SAMPLE_CHANNELS))
            except AttributeError:
                track = (AudioSegment.from_file(file.decode("utf-8"))
                         .set_frame_rate(SAMPLE_RATE)
                         .set_sample_width(SAMPLE_WIDTH)
                         .set_channels(SAMPLE_CHANNELS))

            # Normalize?
            if normalize:
                track = track.apply_gain(-track.max_dBFS)

            # Store data.
            self.data['raw'][i] = np.array(track.get_array_of_samples(), dtype=np.int16)

        self.data.flush()
        print('\nDone!')

    def collect_frames(self):
        '''
        Takes all the audio files and merges their frames together into one long array
        for use with the sample generator.
        '''

        if 'frames' in self.data:
            print('Frame merging already done. Skipping.')
            return

        if 'raw' not in self.data:
            print('Could not find raw data!')
            return

        frame_count = 0
        progress = 1

        # Calculate number of frames needed.
        for raw in self.data['raw']:
            frame_count += int((len(raw) + (FRAME_SIZE - (len(raw) % FRAME_SIZE))) / FRAME_SIZE)
            print('Counting frames ({0} of {1})'.format(progress, self.get_track_count()), end='\r', flush=True)
            progress += 1

        # Create data set for frames.
        dt = np.dtype(np.int16)
        self.data.create_dataset('frames', (frame_count, FRAME_SIZE), dtype=dt)

        progress = 0

        # Buffer to speed up merging as HDF5 is not fast with lots of indexing.
        buffer = np.array([])
        buffer_limit = FRAME_SIZE * 4096

        # Merge frames.
        for raw in self.data['raw']:

            # Setup raw data with zero padding on the end to fit frame size.
            raw = np.concatenate((raw, np.zeros(FRAME_SIZE - (len(raw) % FRAME_SIZE))))

            # Add to buffer.
            buffer = np.concatenate((buffer, raw))

            # If buffer is not filled up and we are not done, keep filling the buffer up.
            if len(buffer) < buffer_limit and progress + (len(buffer) / FRAME_SIZE) < frame_count:
                continue

            # Get frames.
            frames = np.array(np.split(buffer, len(buffer) / FRAME_SIZE))
            buffer = np.array([])

            # Add frames to list.
            self.data['frames'][progress: progress + len(frames)] = frames

            progress += len(frames)
            print('Merging frames ({0} of {1})'.format(progress, frame_count), end='\r', flush=True)

        self.data.flush()
        print('\nDone!')

    def label_frames(self):
        '''
        Takes all audio frames and labels them using the WebRTC VAD.
        '''

        if 'labels' in self.data:
            print('Frame labelling already done. Skipping.')
            return

        if 'frames' not in self.data:
            print('Could not find any frames!')
            return

        vad = webrtcvad.Vad(0)

        frame_count = len(self.data['frames'])
        progress = 0
        batch_size = 65536

        # Create data set for labels.
        dt = np.dtype(np.uint8)
        self.data.create_dataset('labels', (frame_count,), dtype=dt)

        # Label all the frames.
        for pos in range(0, frame_count, batch_size):
            frames = self.data['frames'][pos: pos + batch_size]
            labels = [1 if vad.is_speech(f.tobytes(), sample_rate=SAMPLE_RATE) else 0 for f in frames]
            self.data['labels'][pos: pos + batch_size] = np.array(labels)

            progress += len(labels)
            print('Labelling frames ({0} of {1})'.format(progress, frame_count), end='\r', flush=True)

        self.data.flush()
        print('\nDone!')

def prepare_files():

    speech_dataset = FileManager('speech', '/home/kevin/Documents/voice-activity-detection/data/data/LibriSpeech')
    speech_dataset.prepare_files()
    speech_dataset.collect_frames()
    speech_dataset.label_frames()
    print('Speech dataset labeled')

    noise_dataset = FileManager('noise', '/home/kevin/Documents/voice-activity-detection/data/data/QUT-NOISE')
    noise_dataset.prepare_files(normalize=True)
    noise_dataset.collect_frames()

    return speech_dataset, noise_dataset
