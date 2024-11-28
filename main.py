from matplotlib import pyplot as plt
from core.generator import test_generator
from core.process_data import process_training_data, process_test_data
from core.train import test_network, initialize_network, set_seed, train_all_models, get_model, netvad
from core.models import Net, NickNet, DenseNet
from core.prepare_strong_files import prepare_strong_files
from core.prepare_files import prepare_files
from evaluation.evaluate_webrtc_vad import webrtc_vad_accuracy


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
    # model = NickNet()
    # model_name = 'gru'
    # model = get_model(data, model, model_name)

    # # Evaluate model with sample data
    # netvad(model, data, title='Training Data')
    # # netvad(model, strong_data, init_pos=400, title='STRONG Data')

    # plt.show()


if __name__ == '__main__':
    main()
