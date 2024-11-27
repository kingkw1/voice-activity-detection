from matplotlib import pyplot as plt
from core.generator import test_generator, webrtc_vad_accuracy
from core.process_data import process_training_data, process_test_data
from core.train import test_network, initialize_network, set_seed, train_all_models, get_model, netvad
from core.models import Net, NickNet, DenseNet
from core.prepare_strong_files import prepare_strong_files
from core.prepare_files import prepare_files


def main():
    # Set up data for use in neural networks
    speech_dataset, noise_dataset = prepare_files()
    data = process_training_data(speech_dataset, noise_dataset)

    # Define data generator 
    test_generator(data)

    initialize_network()
    test_network(data)

    # Train all data
    train_all_models(data)
    plt.show()

    # Initialize the model & Load/Train it
    model = NickNet()
    model_name = 'gru'
    model = get_model(data, model, model_name)

    # Evaluate model with sample data
    netvad(model, data, title='Training Data')
    # netvad(model, strong_data, init_pos=400, title='STRONG Data')

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
