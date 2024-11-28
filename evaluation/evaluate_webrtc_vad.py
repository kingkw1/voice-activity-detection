from core.common import SAMPLE_RATE
from core.generator import DataGenerator
from core.prepare_files import prepare_files
from core.process_data import process_training_data

from matplotlib import pyplot as plt
import webrtcvad


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
        if pos + batch_size > generator.size:
            batch_size = generator.size - pos
        frames, _, _, labels = generator.get_data(pos, pos + batch_size)

        for i, frame in enumerate(frames):
            if vad.is_speech(frame.tobytes(), sample_rate=SAMPLE_RATE) == labels[i]:
                correct += 1

    return (correct / generator.size)


def main():
    # Set up data for use in neural networks
    speech_dataset, noise_dataset = prepare_files()
    data = process_training_data(speech_dataset, noise_dataset)

    print('Accuracy (sensitivity 0, no noise):', webrtc_vad_accuracy(data, 0, 'None'))
    print('Accuracy (sensitivity 0, -15 dB noise level):', webrtc_vad_accuracy(data, 0, '-15'))
    print('Accuracy (sensitivity 0, -3 dB noise level):', webrtc_vad_accuracy(data, 0, '-3'))

    print('Accuracy (sensitivity 1, no noise):', webrtc_vad_accuracy(data, 1, 'None'))
    print('Accuracy (sensitivity 1, -15 dB noise level):', webrtc_vad_accuracy(data, 1, '-15'))
    print('Accuracy (sensitivity 1, -3 dB noise level):', webrtc_vad_accuracy(data, 1, '-3'))

    print('Accuracy (sensitivity 2, no noise):', webrtc_vad_accuracy(data, 2, 'None'))
    print('Accuracy (sensitivity 2, -15 dB noise level):', webrtc_vad_accuracy(data, 2, '-15'))
    print('Accuracy (sensitivity 2, -3 dB noise level):', webrtc_vad_accuracy(data, 2, '-3'))

    plt.show()


if __name__ == '__main__':
    main()
