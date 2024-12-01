import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.io.wavfile import write
import subprocess
from tqdm import tqdm

from core.common import SAMPLE_RATE
from core.prepare_strong_files import STRONGFileManager, FRAME_SIZE_MS


# Parameters
FPS = 30  # Frames per second
WINDOW_SIZE = int(SAMPLE_RATE * (FRAME_SIZE_MS / 1000.0))


def main(file_index=0):  
    # Load data from STRONGFileManager
    strong_dataset = STRONGFileManager('strong')
    sample_rate = SAMPLE_RATE
    video_data = strong_dataset.data['raw'][file_index]
    mic_data = strong_dataset.data['mic'][file_index]
    labels = strong_dataset.data['labels']
    frame_times = strong_dataset.data['frame_times']

    # Derive the duration from the audio length
    total_samples = len(video_data)
    DURATION = total_samples / sample_rate  # Audio duration in seconds
    samples_per_frame = int(sample_rate / FPS)  # Samples per video frame
    n_frames = frames = int(FPS * DURATION)  # Total number of frames

    # Save audio to a .wav file
    audio_output_file = "audio_output.wav"
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
        start_sample = frame * samples_per_frame
        end_sample = start_sample + samples_per_frame

        video_segment = video_data[start_sample:end_sample]
        mic_segment = mic_data[start_sample:end_sample]
        label_segment = labels[start_sample // WINDOW_SIZE:end_sample // WINDOW_SIZE]
        time_audio = np.arange(start_sample, end_sample) / sample_rate
        time_labels = frame_times[start_sample // WINDOW_SIZE:end_sample // WINDOW_SIZE]

        video_line.set_data(time_audio, video_segment)
        mic_line.set_data(time_audio, mic_segment)
        label_line.set_data(time_labels, label_segment)

        axs[0].set_xlim(time_audio[0], time_audio[-1])
        axs[1].set_xlim(time_audio[0], time_audio[-1])
        axs[2].set_xlim(time_audio[0], time_audio[-1])

        # Update the progress bar
        progress_bar.update(1)

        return video_line, mic_line, label_line

    try:
        # Generate animation
        frames = int(FPS * DURATION)
        ani = FuncAnimation(fig, update, frames=frames, blit=False)

        # Debugging: Reduce DPI or frame size
        fig.set_size_inches(8, 6)
        plt.rcParams["savefig.dpi"] = 100

        # Save animation as a video without audio
        video_output_file = "live_plot.mp4"
        writer = FFMpegWriter(fps=FPS, metadata={"artist": "Matplotlib"}, codec='libx264', bitrate=1800)
        ani.save(video_output_file, writer=writer)
        print(f"Video saved as {video_output_file}")

        # Merge video and audio using ffmpeg
        audio_output_file = "audio.wav"  # Replace with your actual audio file
        final_output_file = "live_plot_with_audio.mp4"
        ffmpeg_command = [
            'ffmpeg', '-i', video_output_file, '-i', audio_output_file, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', final_output_file
        ]
        subprocess.run(ffmpeg_command, check=True)
        print(f"Final video with audio saved as {final_output_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while merging video and audio: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
