from matplotlib import pyplot as plt
from core.generator import test_generator, webrtc_vad_accuracy
from core.process_data import process_training_data, process_test_data
from core.train import test_network, initialize_network, set_seed, train_all_models, get_model, netvad
from core.models import Net, NickNet, DenseNet
from core.prepare_strong_files import prepare_strong_files
from core.prepare_files import prepare_files


def main():
    # Prepare the STRONG Data
    labeled_strong_dataset = prepare_strong_files()
    train_data = process_test_data(labeled_strong_dataset)

    test_generator(train_data)
    initialize_network()
    test_network(train_data)
    train_all_models(train_data)

    # Evaluate model with sample data
    # netvad(model, train_data, init_pos=400, title='STRONG Data')

    # print('Accuracy (sensitivity 0, no noise):', webrtc_vad_accuracy(data, 0, 'None'))
    # print('Accuracy (sensitivity 0, -15 dB noise level):', webrtc_vad_accuracy(data, 0, '-15'))
    # print('Accuracy (sensitivity 0, -3 dB noise level):', webrtc_vad_accuracy(data, 0, '-3'))

    # print('Accuracy (sensitivity 1, no noise):', webrtc_vad_accuracy(data, 1, 'None'))
    # print('Accuracy (sensitivity 1, -15 dB noise level):', webrtc_vad_accuracy(data, 1, '-15'))
    # print('Accuracy (sensitivity 1, -3 dB noise level):', webrtc_vad_accuracy(data, 1, '-3'))

    # print('Accuracy (sensitivity 2, no noise):', webrtc_vad_accuracy(data, 2, 'None'))
    # print('Accuracy (sensitivity 2, -15 dB noise level):', webrtc_vad_accuracy(data, 2, '-15'))
    # print('Accuracy (sensitivity 2, -3 dB noise level):', webrtc_vad_accuracy(data, 2, '-3'))

    # plt.show()


if __name__ == '__main__':
    main()
