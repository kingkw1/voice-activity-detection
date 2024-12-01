import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.io.wavfile import write
import subprocess
from tqdm import tqdm

from core.common import SAMPLE_RATE, DATA_FOLDER
from core.prepare_strong_files import STRONGFileManager, FRAME_SIZE_MS

# Parameters
FPS = 30  # Frames per second
WINDOW_SIZE = int(SAMPLE_RATE * (FRAME_SIZE_MS / 1000.0))


def main(file_index=0, start_time=0, end_time=None, plot_window_size=10):
    """
    :param plot_window_size: Size of the plot window in seconds (how much of the plot to display at once).
    """
    # Load data from STRONGFileManager
    strong_dataset = STRONGFileManager('strong')
    sample_rate = SAMPLE_RATE
    video_data = strong_dataset.data['raw'][file_index]
    mic_data = strong_dataset.data['mic'][file_index]
    labels = strong_dataset.data['labels']
    frame_times = strong_dataset.data['frame_times']

    # Total duration of the audio in seconds
    total_duration = len(video_data) / sample_rate

    # Validate and adjust start and stop times
    if end_time is None:
        end_time = total_duration
    if start_time < 0 or end_time > total_duration or start_time >= end_time:
        raise ValueError("Invalid start_time or end_time values.")

    # Calculate start and stop samples
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Subset the data based on start and stop times
    video_data = video_data[start_sample:end_sample]
    mic_data = mic_data[start_sample:end_sample]
    labels = labels[start_sample // WINDOW_SIZE:end_sample // WINDOW_SIZE]
    frame_times = frame_times[start_sample // WINDOW_SIZE:end_sample // WINDOW_SIZE]

    # Derive the duration for the snippet
    snippet_duration = (end_sample - start_sample) / sample_rate
    samples_per_frame = int(sample_rate / FPS)  # Samples per video frame
    n_frames = int(FPS * snippet_duration)  # Total number of frames

    # Save audio to a .wav file
    audio_output_file = "audio_snippet.wav"
    audio_combined = (video_data + mic_data) / 2  # Combine audio tracks (optional)
    write(audio_output_file, sample_rate, audio_combined.astype(np.int16))

    # Create figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Initialize plots
    video_line, = axs[0].plot([], [], color='blue', lw=2, label="Video Audio")
    mic_line, = axs[1].plot([], [], color='green', lw=2, label="Mic Audio")
    label_line, = axs[2].plot([], [], color='red', lw=2, label="Speech Activity")

    axs[0].set_title("Video Audio Data")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend(loc="upper right")
    axs[1].set_title("Mic Audio Data")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend(loc="upper right")
    axs[2].set_title("Speech Activity Labels")
    axs[2].set_ylabel("Label")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(loc="upper right")

    fig.tight_layout()

    # Create a tqdm progress bar
    progress_bar = tqdm(total=n_frames, desc="Generating animation frames")

    # Update function for animation
    def update(frame):
        current_start = frame * samples_per_frame
        current_end = current_start + samples_per_frame

        video_segment = video_data[current_start:current_end]
        mic_segment = mic_data[current_start:current_end]
        label_segment = labels[current_start // WINDOW_SIZE:current_end // WINDOW_SIZE]
        time_audio = np.arange(current_start, current_end) / sample_rate + start_time
        time_labels = frame_times[current_start // WINDOW_SIZE:current_end // WINDOW_SIZE]

        video_line.set_data(np.arange(len(video_data)) / sample_rate + start_time, video_data)
        mic_line.set_data(np.arange(len(mic_data)) / sample_rate + start_time, mic_data)
        label_line.set_data(frame_times, labels)

        # Center the plot window around the current time, displaying past and future data
        current_time = time_audio[-1]  # Time of the current frame's last point
        half_window = plot_window_size / 2  # Half of the window size

        # Set x-axis limits to center the plot window around the current time
        x_min = max(current_time - half_window, start_time)  # Avoid going before the start
        x_max = min(current_time + half_window, end_time)  # Avoid going beyond the end

        axs[0].set_xlim(x_min, x_max)
        axs[1].set_xlim(x_min, x_max)
        axs[2].set_xlim(x_min, x_max)

        # Update the progress bar
        progress_bar.update(1)

        return video_line, mic_line, label_line

    try:
        # Generate animation
        ani = FuncAnimation(fig, update, frames=n_frames, blit=False)

        # Save animation as a video without audio
        video_output_file = f"{DATA_FOLDER}/live_plot_snippet.mp4"
        writer = FFMpegWriter(fps=FPS, metadata={"artist": "Matplotlib"}, codec='libx264', bitrate=1800)
        ani.save(video_output_file, writer=writer)
        print(f"Video saved as {video_output_file}")

        # Merge video and audio using ffmpeg
        final_output_file = f"{DATA_FOLDER}/live_plot_snippet_with_audio.mp4"
        ffmpeg_command = [
            'ffmpeg', '-i', video_output_file, '-i', audio_output_file, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', final_output_file
        ]
        subprocess.run(ffmpeg_command, check=True)
        print(f"Final video with audio saved as {final_output_file}")

        # Delete the old video without audio
        subprocess.run(['rm', video_output_file], check=True)
        print(f"Deleted the old video without audio: {video_output_file}")

        # Delete the audio file
        subprocess.run(['rm', audio_output_file], check=True)
        print(f"Deleted the audio file: {audio_output_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while merging video and audio: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        progress_bar.close()


if __name__ == '__main__':
    # Example: Generate snippet between 10 seconds and 20 seconds with a plot window size of 10 seconds (5s past and 5s future)
    main(file_index=0, start_time=10, end_time=40, plot_window_size=10)
