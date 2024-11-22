import glob
from os import path, listdir
import sys
import h5py
import numpy as np
import webrtcvad
from pydub import AudioSegment
import os
from matplotlib import pyplot as plt

# Add the parent directory to the PYTHONPATH
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from core.common import DATA_FOLDER, SAMPLE_CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE, NOISE_FOLDER, SPEECH_FOLDER, \
    STRONG_PROCESSED_MIC_FOLDER, STRONG_VIDEO_AUDIO_FOLDER, create_dictionary

OBJ_PREPARE_AUDIO = True

# Frame size to use for the labelling.
FRAME_SIZE_MS = 30  
BATCH_SIZE = 65536

# Calculate frame size in data points.
WINDOW_SIZE = int(SAMPLE_RATE * (FRAME_SIZE_MS / 1000.0))
N_WINDOWS_BUFFER = 4096

# Labeling aggressiveness.
VAD_AGGRESSIVENESS = 3


class STRONGFileManager():
    def __init__(self, name):
        self.name = name
        output_file_name = DATA_FOLDER + '/' + name + '.hdf5'

        # # if file already exists, delete it
        # if path.exists(output_file_name):
        #     print(f"Deleting existing {output_file_name} file")
        #     os.remove(output_file_name)

        self.data = h5py.File(output_file_name, 'a')
        
        mic_files = listdir(STRONG_PROCESSED_MIC_FOLDER)
        video_files = listdir(STRONG_VIDEO_AUDIO_FOLDER)

        file_dict = create_dictionary(mic_files, video_files)
        file_path_dict = {path.join(STRONG_VIDEO_AUDIO_FOLDER,key):path.join(STRONG_PROCESSED_MIC_FOLDER, value) for key, value in file_dict.items()}
        self.file_dict = file_path_dict

        # Setup video files as "files" variable in data
        if ('files' not in self.data) or (self.data['files'].shape[0]==0):
            files = list(self.file_dict.keys())
            files.sort()
            self.data['files'] = files

    def get_track_count(self):
        return len(self.data['files'])

    def prepare_files(self, normalize_=False):
        """
        Prepares the files for the project.
        Will do the following check for each file:
        1. Check if it has been converted already to the desired format.
        2. Converts all files to WAV with the desired properties.
        3. Stores the converted files in a separate folder.
        """

        print('Found {0} tracks to check.'.format(self.get_track_count()))
        progress = 1

        # Setup raw data set.
        if 'raw' not in self.data:
            dt = h5py.special_dtype(vlen=np.dtype(np.int16))
            self.data.create_dataset('raw', (self.get_track_count(),), dtype=dt)

        # Setup raw data set.
        if 'mic' not in self.data:
            dt = h5py.special_dtype(vlen=np.dtype(np.int16))
            self.data.create_dataset('mic', (self.get_track_count(),), dtype=dt)

        # Convert files to desired format and save raw content.
        for i, file in enumerate(self.data['files']):

            print('Processing {0} of {1}'.format(progress, self.get_track_count()), end='\r', flush=True)
            progress += 1
            
            # Get the data
            video_track = self.get_track(file, normalize_)
            mic_track = self.get_track(self.file_dict[file.decode("utf-8")], normalize_)

            # Format the data
            video_data = np.array(video_track.get_array_of_samples(), dtype=np.int16)
            mic_data = np.array(mic_track.get_array_of_samples(), dtype=np.int16)

            min_length = min(len(video_data), len(mic_data))
            video_data = video_data[:min_length]
            mic_data = mic_data[:min_length]

            # Save the data
            self.data['raw'][i] = video_data
            self.data['mic'][i] = mic_data

            # Plot both data
            # x_window = [11, 12]
            # start_sample = int(x_window[0] * SAMPLE_RATE)
            # end_sample = int(x_window[1] * SAMPLE_RATE)
            # time = np.arange(start_sample, end_sample) / SAMPLE_RATE
            # plt.figure(figsize=(10, 5))
            # plt.plot(time, video_data[start_sample:end_sample], label='Video Data')
            # plt.plot(time, mic_data[start_sample:end_sample], label='Mic Data')
            # plt.xlim(x_window)
            # plt.title('Video and Mic Data')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Amplitude')
            # plt.legend()
            # plt.savefig(f'{DATA_FOLDER}/plot_{i}.png')
            # plt.close()

        self.data.flush()

    def collect_frames(self):
        """
        Takes all the audio files and merges their frames together into one long array
        for use with the sample generator.
        """

        # Calculate number of frames in all the raw data.
        frame_count = self.count_frames()
        
        # Create data set for frames and frame times.
        dt = np.dtype(np.int16)
        self.data.create_dataset('frames', (frame_count, WINDOW_SIZE), dtype=dt)
        self.data.create_dataset('mic_frames', (frame_count, WINDOW_SIZE), dtype=dt)
        self.data.create_dataset('frame_times', (frame_count,), dtype=np.float64)

        # Buffer to speed up merging as HDF5 is not fast with lots of indexing.
        buffer = np.array([])
        mic_buffer = np.array([])
        buffer_limit = WINDOW_SIZE * N_WINDOWS_BUFFER
        frame_duration = WINDOW_SIZE / SAMPLE_RATE

        # Merge frames.
        total_sample_count = 0
        frame_merge_progress = 0
        current_time = 0.0
        for i, raw in enumerate(self.data['raw']):
            raw = self.data['raw'][i]
            mic_raw = self.data['mic'][i]

            # Setup raw data with zero padding on the end to fit frame size. STUFF AN EXTRA FRAME OF ZEROS IN THERE. SOME WILL GIT CLIPPED CHOOM
            raw = np.concatenate((raw, np.zeros(WINDOW_SIZE - (len(raw) % WINDOW_SIZE))))
            mic_raw = np.concatenate((mic_raw, np.zeros(WINDOW_SIZE - (len(mic_raw) % WINDOW_SIZE))))
            assert len(raw) == len(mic_raw)
            total_sample_count += len(raw)
            
            # Add to buffer.
            buffer = np.concatenate((buffer, raw))
            mic_buffer = np.concatenate((mic_buffer, mic_raw))

            # If buffer is not filled up and we are not done, keep filling the buffer up.
            if len(buffer) < buffer_limit and frame_merge_progress + (len(buffer) / WINDOW_SIZE) < frame_count:
                continue

            # Get frames.
            frames = np.array(np.split(buffer, len(buffer) / WINDOW_SIZE))
            mic_frames = np.array(np.split(mic_buffer, len(mic_buffer) / WINDOW_SIZE))
            buffer = np.array([])
            mic_buffer = np.array([])

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
            self.data['mic_frames'][frame_merge_progress: frame_merge_progress + len(mic_frames)] = mic_frames

            frame_merge_progress += len(frames)
            print('Merging frames ({0} of {1})'.format(frame_merge_progress, frame_count), end='\r', flush=True)

        self.data.flush()
        print('Split {0} samples within {1} tracks into {2} frames of size {3}.'.format(total_sample_count, self.get_track_count(), frame_count, WINDOW_SIZE))

    def count_frames(self):
        # Calculate number of frames in all the raw data.
        frame_count = 0
        frame_count_progress = 1
        for raw in self.data['raw']:
            frame_count += int((len(raw) + (WINDOW_SIZE - (len(raw) % WINDOW_SIZE))) / WINDOW_SIZE)
            print('Counting frames ({0} of {1})'.format(frame_count_progress, self.get_track_count()), end='\r', flush=True)
            frame_count_progress += 1

        return frame_count

    def label_frames(self):
        """
        Takes all audio frames and labels them using the WebRTC VAD.
        """

        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        frame_count = len(self.data['mic_frames'])
        progress = 0

        # Create data set for labels.
        dt = np.dtype(np.uint8)
        self.data.create_dataset('labels', (frame_count,), dtype=dt)

        # Label all the frames. 1 label per frame.
        for pos in range(0, frame_count, BATCH_SIZE):
            frames = self.data['mic_frames'][pos: pos + BATCH_SIZE]
            labels = [1 if vad.is_speech(f.tobytes(), sample_rate=SAMPLE_RATE) else 0 for f in frames]
            self.data['labels'][pos: pos + BATCH_SIZE] = np.array(labels)

            progress += len(labels)
            print('Labelling frames ({0} of {1})'.format(progress, frame_count), end='\r', flush=True)

        self.data.flush()

    def get_track(self, file, normalize=False):
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
        
        return track


def plot_audio_and_labels(strong_dataset, segment_index=0, start_time=10, end_time=25):
    """
    Plot video audio data, mic audio data, and labels for a given segment.
    
    Parameters:
    - strong_dataset (STRONGFileManager): Instance of STRONGFileManager with prepared data.
    - segment_index (int): Index of the segment to plot (default is 0).
    - start_time (float): Start time of the segment in seconds (default is 0).
    - end_time (float): End time of the segment in seconds (default is 5).
    """
    # Retrieve sample rate and calculate the sample range for the specified time window.
    sample_rate = SAMPLE_RATE
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Extract data from the STRONGFileManager datasets
    video_data = strong_dataset.data['raw'][segment_index][start_sample:end_sample]
    mic_data = strong_dataset.data['mic'][segment_index][start_sample:end_sample]
    
    # Calculate frame indices for the labels based on the frame size and sample range
    start_frame = start_sample // WINDOW_SIZE
    end_frame = end_sample // WINDOW_SIZE
    labels = strong_dataset.data['labels'][start_frame:end_frame]
    frame_times = strong_dataset.data['frame_times'][start_frame:end_frame]

    # Convert sample indices to time for plotting
    time_audio = np.arange(start_sample, end_sample) / sample_rate

    # Plot each subplot
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Plot video audio data
    axs[0].plot(time_audio, video_data, color='blue', label='Video Audio')
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Video Audio Data")
    axs[0].legend(loc="upper right")

    # Plot mic audio data
    axs[1].plot(time_audio, mic_data, color='green', label='Mic Audio')
    axs[1].set_ylabel("Amplitude")
    axs[1].set_title("Mic Audio Data")
    axs[1].legend(loc="upper right")

    # Plot labels
    axs[2].plot(frame_times, labels, color='red', label='Speech Activity')
    axs[2].set_ylabel("Label")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title("Speech Activity Labels")
    axs[2].legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()


def prepare_strong_files():
    # Prepare the portion of the video data with associated mic audio that can be used for ground truth labels.
    strong_dataset = STRONGFileManager('strong')
    strong_dataset.prepare_files(normalize_=True)
    strong_dataset.collect_frames()
    strong_dataset.label_frames()


if __name__ == '__main__':
    #  prepare_strong_files()

    # Load the dataset
    strong_dataset = STRONGFileManager('strong')
    plot_audio_and_labels(strong_dataset)