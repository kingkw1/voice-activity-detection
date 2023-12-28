from matplotlib import pyplot as plt
from core.common import num_params
from core.generator import test_generator, webrtc_vad_accuracy
from core.prepare_files import prepare_files
from core.process_data import process_data
from core.train import test_network, initialize_network, set_seed, train_models
from core.models import Net, NickNet, DenseNet


def main():
    # Prepare the audio files
    speech_dataset, noise_dataset = prepare_files()

    # Set up data for use in neural networks
    data = process_data(speech_dataset, noise_dataset)

    # Define data generator
    test_generator(data)

    initialize_network()
    test_network(data)

    net = Net(large=False)
    num_params(net)
    print(net)

    gru = NickNet(large=True)
    num_params(gru)
    print(gru)

    set_seed()
    densenet = DenseNet(large=True)
    num_params(densenet)
    print(densenet)

    train_models(data)

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
