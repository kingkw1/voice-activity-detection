import array
import numpy as np
import webrtcvad
from pydub import AudioSegment

from core.common import NOISE_LEVELS_DB, SAMPLE_RATE, SAMPLE_WIDTH, SAMPLE_CHANNELS
from core.visualization import Vis

OBJ_SHOW_PLAYABLE_TRACKS = True


class DataGenerator:

    def __init__(self, data, size_limit=0):
        self.data = data
        self.size = size_limit if size_limit > 0 else len(data['labels'])
        self.data_mode = 0  # Default to training data

    def set_noise_level_db(self, level, reset_data_mode=True):

        if level not in NOISE_LEVELS_DB:
            raise Exception(f'Noise level "{level}" not supported! Options are: {list(NOISE_LEVELS_DB.keys())}')

        self.noise_level = level

        # Optionally reset data mode and position in file
        if reset_data_mode:
            if self.data_mode == 0:
                self.use_train_data()
            elif self.data_mode == 1:
                self.use_validate_data()
            elif self.data_mode == 2:
                self.use_test_data()

    def setup_generation(self, frame_count, step_size, batch_size, val_part=0.1, test_part=0.1):

        self.frame_count = frame_count
        self.step_size = step_size
        self.batch_size = batch_size

        # Setup indexes and sizes for data splits.
        self.train_index = 0
        self.val_index = int((1.0 - val_part - test_part) * self.size)
        self.test_index = int((1.0 - test_part) * self.size)

        self.train_size = self.val_index
        self.val_size = self.test_index - self.val_index
        self.test_size = self.size - self.test_index

    def use_train_data(self):

        # Calculate how many batches we can construct from our given parameters.
        n = int((self.train_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.train_index
        self.data_mode = 0

    def use_validate_data(self):

        # Calculate how many batches we can construct from our given parameters.
        n = int((self.val_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.val_index
        self.data_mode = 1

    def use_test_data(self):

        # Calculate how many batches we can construct from our given parameters.
        n = int((self.test_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.test_index
        self.data_mode = 2

    def get_data(self, index_from, index_to):
        frames = self.data['frames-' + self.noise_level][index_from: index_to]
        mfcc = self.data['mfcc-' + self.noise_level][index_from: index_to]
        delta = self.data['delta-' + self.noise_level][index_from: index_to]
        labels = self.data['labels'][index_from: index_to]
        return frames, mfcc, delta, labels

    def get_batch(self, index):

        # Get current position.
        pos = self.initial_pos + (self.batch_size * index) * self.step_size

        # Get all data needed.
        l = self.frame_count + self.step_size * self.batch_size
        frames, mfcc, delta, labels = self.get_data(pos, pos + l)

        x, y, i = [], [], 0

        # Get batches
        while len(y) < self.batch_size:
            # Get data for the window.
            X = np.hstack((mfcc[i: i + self.frame_count], delta[i: i + self.frame_count]))

            # Append sequence to list of frames
            x.append(X)

            # Select label from center of sequence as label for that sequence.
            y_range = labels[i: i + self.frame_count]
            y.append(int(y_range[int(self.frame_count / 2)]))

            # Increment window using set step size
            i += self.step_size

        return x, y

    def plot_data(self, index_from, index_to, show_track=False):

        frames, mfcc, delta, labels = self.get_data(index_from, index_to)

        Vis.plot_sample(frames, labels)
        Vis.plot_sample_webrtc(frames)
        Vis.plot_features(mfcc, delta)

        # By returning a track and having this as the last statement in a code cell,
        # the track will appear as an audio track UI element (not supported by Windows).
        if show_track and OBJ_SHOW_PLAYABLE_TRACKS:
            return (AudioSegment(data=array.array('h', frames.flatten()),
                                 sample_width=SAMPLE_WIDTH, frame_rate=SAMPLE_RATE,
                                 channels=SAMPLE_CHANNELS))


def test_generator(data):
    # Test generator features.
    generator = DataGenerator(data, size_limit=10000)

    generator.setup_generation(frame_count=3, step_size=1, batch_size=2)
    generator.set_noise_level_db('-3')
    generator.use_train_data()
    X, y = generator.get_batch(0)

    print(f'Load a few frames into memory:\n{X[0]}\n\nCorresponding label: {y[0]}')

    # generator.plot_data(0, 1000)


def webrtc_vad_accuracy(data, sensitivity, noise_level):
    vad = webrtcvad.Vad(sensitivity)
    generator = DataGenerator(data, size_limit=0)

    # Not needed but must be set.
    generator.setup_generation(frame_count=1, step_size=1, batch_size=1)

    # Setup noise level and test data.
    generator.set_noise_level_db(noise_level)
    generator.use_test_data()

    correct = 0
    batch_size = 1000

    for pos in range(0, generator.size, batch_size):

        frames, _, _, labels = generator.get_data(pos, pos + batch_size)

        for i, frame in enumerate(frames):
            if vad.is_speech(frame.tobytes(), sample_rate=SAMPLE_RATE) == labels[i]:
                correct += 1

    return (correct / generator.size)
