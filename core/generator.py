import array
import numpy as np
import webrtcvad
from pydub import AudioSegment
import sys
from os import path
from sklearn.utils import resample

# Add the parent directory to the PYTHONPATH
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from core.common import NOISE_LEVELS_DB, SAMPLE_RATE, SAMPLE_WIDTH, SAMPLE_CHANNELS
from core.visualization import Vis
from core.prepare_strong_files import STRONGFileManager

OBJ_SHOW_PLAYABLE_TRACKS = True
SEED = 1337


class DataGenerator:

    def __init__(self, data, size_limit=0):
        # Initialize the DataGenerator with data and size limit
        self.data = data
        self.size = size_limit if size_limit > 0 else len(data['labels'])
        self.data_mode = 0  # Default to training data

        # Shuffle data indices
        self.indices = np.arange(self.size)
        np.random.seed(SEED)
        np.random.shuffle(self.indices)

    def set_noise_level_db(self, level, reset_data_mode=True):
        # Set the noise level in dB and optionally reset data mode

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
        # Setup the parameters for data generation

        self.frame_count = frame_count
        self.step_size = step_size
        self.batch_size = batch_size

        # Calculate indexes for data splits
        self.train_index = 0
        self.val_index = int(self.size * (1 - val_part - test_part))
        self.test_index = int(self.size * (1 - test_part))

        # Calculate sizes for each data split
        self.train_size = self.val_index
        self.val_size = self.test_index - self.val_index
        self.test_size = self.size - self.test_index
        assert self.size == self.train_size + self.val_size + self.test_size
        
        # Print data split sizes for debugging
        print(f"Train size: {self.train_size}, Validation size: {self.val_size}, Test size: {self.test_size}")

    def use_train_data(self):
        # Switch to training data mode

        # Calculate how many batches we can construct from our given parameters.
        n = int((self.train_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.train_index
        self.data_mode = 0

    def use_validate_data(self):
        # Switch to validation data mode

        # Calculate how many batches we can construct from our given parameters.
        n = int((self.val_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.val_index
        self.data_mode = 1

    def use_test_data(self):
        # Switch to test data mode

        # Calculate how many batches we can construct from our given parameters.
        n = int((self.test_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.test_index
        self.data_mode = 2

    def get_data(self, index_from, index_to):
        # plot_labels(self.data, index_from, index_to)
        assert(index_from >= 0 and index_to <= self.size)

        # Retrieve data between specified indexes
        try:
            frames = self.data['frames-' + self.noise_level][index_from: index_to]
            mfcc = self.data['mfcc-' + self.noise_level][index_from: index_to]
            delta = self.data['delta-' + self.noise_level][index_from: index_to]
            labels = self.data['labels'][index_from: index_to]

            print(f'Loaded data from {index_from} to {index_to} for noise level {self.noise_level}')
            print(f'Class distribution in loaded data: {np.bincount(labels)}')
            return frames, mfcc, delta, labels
        except KeyError:
            frames = self.data['frames'][index_from: index_to]
            labels = self.data['labels'][index_from: index_to]
            print(f'Loaded data from {index_from} to {index_to} without noise level')
            print(f'Class distribution in loaded data: {np.bincount(labels)}')
            return frames, None, None, labels
        
    def get_batch(self, index, skip_single_class=False):
        # Get a batch of data based on the current index

        # Get current position.
        pos = self.initial_pos + (self.batch_size * index) * self.step_size

        # Further increase the size of data slices
        l = self.frame_count + self.step_size * self.batch_size
        frames, mfcc, delta, labels = self.get_data(pos, pos + l)

        # Stratified sampling to ensure balanced batches
        class_0_indices = np.where(labels == 0)[0]
        class_1_indices = np.where(labels == 1)[0]

        # Check if both classes are present
        if skip_single_class and (len(class_0_indices) == 0 or len(class_1_indices) == 0):
            print(f"Batch {index} - Skipping due to missing class")
            return [], []

        # Determine the number of samples to take from each class
        num_samples_per_class = self.batch_size // 2

        # Sample from each class
        sampled_class_0_indices = np.random.choice(class_0_indices, num_samples_per_class, replace=True)
        sampled_class_1_indices = np.random.choice(class_1_indices, num_samples_per_class, replace=True)

        # Combine the sampled indices
        balanced_indices = np.hstack((sampled_class_0_indices, sampled_class_1_indices))
        np.random.shuffle(balanced_indices)

        x, y = [], []
        inconsistent_slices = 0
        for i in balanced_indices:
            # Ensure the slicing is correct and results in consistent shapes
            mfcc_slice = mfcc[i: i + self.frame_count]
            delta_slice = delta[i: i + self.frame_count]
            if mfcc_slice.shape[0] == self.frame_count and delta_slice.shape[0] == self.frame_count:
                X = np.hstack((mfcc_slice, delta_slice))
                x.append(X)
                y.append(labels[i])
            else:
                # print(f"Skipping index {i} due to inconsistent slice shapes")
                inconsistent_slices += 1
        print(f"Batch {index} - Skipped {inconsistent_slices} inconsistent slices out of {len(balanced_indices)}")

        # Convert to numpy array and add debug print to verify the shape
        x = np.array(x)
        y = np.array(y)
        print(f"Batch {index} - Data shape: {x.shape}, Labels shape: {y.shape}")

        # Print batch class distribution for debugging
        print(f"Batch {index} - Class distribution:", np.bincount(y))

        return x, y

    def plot_data(self, index_from, index_to, show_track=False):
        # Plot data between specified indexes

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

    # Adjust frame_count to a larger value to increase the likelihood of including class 1 samples
    generator.setup_generation(frame_count=200, step_size=1, batch_size=2)
    generator.set_noise_level_db('-3')
    generator.use_train_data()

    # Print overall class distribution
    labels = data['labels']
    label_counts = np.bincount(labels)
    for label, count in enumerate(label_counts):
        print(f'Label {label}: {count}')

    for i in range(3):
        X, y = generator.get_batch(i)
        print(f'Batch {i} - Class distribution: {np.bincount(y)}')

        # Check if batch is not empty before accessing elements
        if X.size > 0 and y.size > 0:
            print(f'Load a few frames into memory:\n{X[0][:5]}\n\nCorresponding label: {y[0]}')
        else:
            print(f'Batch {i} is empty.')

    # generator.plot_data(0, 1000)


def webrtc_vad_accuracy(data, sensitivity, noise_level):
    # Calculate the accuracy of WebRTC VAD

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


if __name__ == '__main__':
    
    strong_dataset = STRONGFileManager('processed_strong_data')
    
    test_generator(strong_dataset.data)