import glob
from os import path, listdir
import sys
import h5py
import numpy as np
import webrtcvad
from pydub import AudioSegment

from matplotlib import pyplot as plt

# Add the parent directory to the PYTHONPATH
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from core.common import DATA_FOLDER, SAMPLE_CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE, NOISE_FOLDER, SPEECH_FOLDER

OBJ_PREPARE_AUDIO = True

# Frame size to use for the labelling.
FRAME_SIZE_MS = 30  
BATCH_SIZE = 65536

# Calculate frame size in data points.
WINDOW_SIZE = int(SAMPLE_RATE * (FRAME_SIZE_MS / 1000.0))
N_WINDOWS_BUFFER = 4096

# Labeling aggressiveness.
VAD_AGGRESSIVENESS = 3


class FileManager:
    """
    Keeps track of audio-files from a data-set.
    Provides support for formatting the wav-files into a desired format.
    Also provides support for conversion of .flac files (as we have in the LibriSpeech data-set).
    """

    def __init__(self, name, directory):

        self.name = name
        self.data = h5py.File(DATA_FOLDER + '/' + name + '.hdf5', 'a')

        # Setup file names.
        if ('files' not in self.data) or (self.data['files'].shape[0]==0):

            # Get files.
            files = glob.glob(directory + '/**/*.wav', recursive=True)
            files.extend(glob.glob(directory + '/**/*.flac', recursive=True))
            files = [f for f in files]
            files.sort()

            # Setup data set.
            dt = h5py.special_dtype(vlen=str)
            self.data.create_dataset('files', (len(files),), dtype=dt)

            # Add file names.
            for i, f in enumerate(files):
                self.data['files'][i] = f

    def get_track_count(self):
        return len(self.data['files'])

    def prepare_files(self, normalize=False):
        """
        Prepares the files for the project.
        Will do the following check for each file:
        1. Check if it has been converted already to the desired format.
        2. Converts all files to WAV with the desired properties.
        3. Stores the converted files in a separate folder.
        """

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
                track = (AudioSegment.from_file(file.decode("utf-8")).set_frame_rate(SAMPLE_RATE).set_sample_width(SAMPLE_WIDTH).set_channels(SAMPLE_CHANNELS))

            # Normalize?
            if normalize:
                track = track.apply_gain(-track.max_dBFS)

            # Store data.
            self.data['raw'][i] = np.array(track.get_array_of_samples(), dtype=np.int16)

        self.data.flush()

    def collect_frames(self):
        """
        Takes all the audio files and merges their frames together into one long array
        for use with the sample generator.
        """

        if 'frames' in self.data:
            print('Frame merging already done. Skipping.')
            return

        if 'raw' not in self.data:
            print('Could not find raw data!')
            return

        frame_count = 0
        frame_count_progress = 1

        # Calculate number of frames in all the raw data.
        for raw in self.data['raw']:
            frame_count += int((len(raw) + (WINDOW_SIZE - (len(raw) % WINDOW_SIZE))) / WINDOW_SIZE)
            print('Counting frames ({0} of {1})'.format(frame_count_progress, self.get_track_count()), end='\r', flush=True)
            frame_count_progress += 1

        # Create data set for frames and frame times.
        dt = np.dtype(np.int16)
        self.data.create_dataset('frames', (frame_count, WINDOW_SIZE), dtype=dt)
        self.data.create_dataset('frame_times', (frame_count,), dtype=np.float64)

        frame_merge_progress = 0
        current_time = 0.0

        # Buffer to speed up merging as HDF5 is not fast with lots of indexing.
        buffer = np.array([])
        buffer_limit = WINDOW_SIZE * N_WINDOWS_BUFFER
        frame_duration = WINDOW_SIZE / SAMPLE_RATE

        # Merge frames.
        total_sample_count = 0
        for raw in self.data['raw']:

            # Setup raw data with zero padding on the end to fit frame size. STUFF AN EXTRA FRAME OF ZEROS IN THERE. SOME WILL GIT CLIPPED CHOOM
            raw = np.concatenate((raw, np.zeros(WINDOW_SIZE - (len(raw) % WINDOW_SIZE))))
            total_sample_count += len(raw)
            
            # Add to buffer.
            buffer = np.concatenate((buffer, raw))

            # If buffer is not filled up and we are not done, keep filling the buffer up.
            if len(buffer) < buffer_limit and frame_merge_progress + (len(buffer) / WINDOW_SIZE) < frame_count:
                continue

            # Get frames.
            frames = np.array(np.split(buffer, len(buffer) / WINDOW_SIZE))
            buffer = np.array([])

            # Calculate frame times.
            frame_times = np.arange(current_time, current_time + len(frames) * frame_duration, frame_duration)
            current_time += len(frames) * frame_duration

            # Ensure the lengths match
            if len(frame_times) != len(frames):
                print(f"Length mismatch: frame_times={len(frame_times)}, frames={len(frames)}")
                frame_times = frame_times[:len(frames)]  # Adjust frame_times to match frames length

            # Add frames and frame times to the dataset.
            self.data['frames'][frame_merge_progress: frame_merge_progress + len(frames)] = frames
            self.data['frame_times'][frame_merge_progress: frame_merge_progress + len(frames)] = frame_times

            frame_merge_progress += len(frames)
            print('Merging frames ({0} of {1})'.format(frame_merge_progress, frame_count), end='\r', flush=True)

        self.data.flush()
        print('Split {0} samples within {1} tracks into {2} frames of size {3}.'.format(total_sample_count, self.get_track_count(), frame_count, WINDOW_SIZE))

    def label_frames(self):
        """
        Takes all audio frames and labels them using the WebRTC VAD.
        """

        if 'labels' in self.data:
            print('Frame labelling already done. Skipping.')
            return

        if 'frames' not in self.data:
            print('Could not find any frames!')
            return

        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        frame_count = len(self.data['frames'])
        progress = 0

        # Create data set for labels.
        dt = np.dtype(np.uint8)
        self.data.create_dataset('labels', (frame_count,), dtype=dt)

        # Label all the frames. 1 label per frame.
        for pos in range(0, frame_count, BATCH_SIZE):
            frames = self.data['frames'][pos: pos + BATCH_SIZE]
            labels = [1 if vad.is_speech(f.tobytes(), sample_rate=SAMPLE_RATE) else 0 for f in frames]
            self.data['labels'][pos: pos + BATCH_SIZE] = np.array(labels)

            progress += len(labels)
            print('Labelling frames ({0} of {1})'.format(progress, frame_count), end='\r', flush=True)

        self.data.flush()


def prepare_files():

    speech_dataset = FileManager('speech', SPEECH_FOLDER)
    speech_dataset.prepare_files()
    speech_dataset.collect_frames()
    speech_dataset.label_frames()
    print('Speech dataset labeled')

    noise_dataset = FileManager('noise', NOISE_FOLDER)
    noise_dataset.prepare_files(normalize=True)
    noise_dataset.collect_frames()

    return speech_dataset, noise_dataset
