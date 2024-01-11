from matplotlib import pyplot as plt
from core.common import num_params
from core.generator import test_generator, webrtc_vad_accuracy
from core.prepare_files import prepare_files, prepare_strong_files
from core.process_data import process_training_data, process_test_data
from core.train import test_network, initialize_network, set_seed, train_all_models, get_model, netvad
from core.models import Net, NickNet, DenseNet


def main():
    # Prepare the audio files
    speech_dataset, noise_dataset = prepare_files()

    # Prepare my labeled files
    strong_video_audio_dataset, _ = prepare_strong_files()

    # Set up data for use in neural networks
    data = process_training_data(speech_dataset, noise_dataset)
    test_data = process_test_data(strong_video_audio_dataset)

    # Define data generator
    test_generator(data)

    initialize_network()
    test_network(data)

    # Initialize the model & Load/Train it
    model = NickNet()
    model_name = 'gru'
    model = get_model(data, model, model_name, gamma=2)

    # Evaluate model with sample data
    netvad(model, data, title='Training Data')

    netvad(model, test_data, init_pos=400, title='STRONG Data')

    train_all_models(data)
    # train_all_models(strong_video_audio_dataset.data)
    # plt.show()

    # print('Accuracy (sensitivity 0, no noise):', webrtc_vad_accuracy(data, 0, 'None'))
    # print('Accuracy (sensitivity 0, -15 dB noise level):', webrtc_vad_accuracy(data, 0, '-15'))
    # print('Accuracy (sensitivity 0, -3 dB noise level):', webrtc_vad_accuracy(data, 0, '-3'))
    #
    # print('Accuracy (sensitivity 1, no noise):', webrtc_vad_accuracy(data, 1, 'None'))
    # print('Accuracy (sensitivity 1, -15 dB noise level):', webrtc_vad_accuracy(data, 1, '-15'))
    # print('Accuracy (sensitivity 1, -3 dB noise level):', webrtc_vad_accuracy(data, 1, '-3'))
    #
    # print('Accuracy (sensitivity 2, no noise):', webrtc_vad_accuracy(data, 2, 'None'))
    # print('Accuracy (sensitivity 2, -15 dB noise level):', webrtc_vad_accuracy(data, 2, '-15'))
    # print('Accuracy (sensitivity 2, -3 dB noise level):', webrtc_vad_accuracy(data, 2, '-3'))

    plt.show()


if __name__ == '__main__':
    main()
