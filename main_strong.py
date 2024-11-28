from matplotlib import pyplot as plt
from core.generator import test_generator
from core.process_data import process_training_data, process_test_data
from core.train import test_network, initialize_network, set_seed, train_all_models, get_model, netvad
from core.models import Net, NickNet, DenseNet
from core.prepare_strong_files import prepare_strong_files
from core.prepare_files import prepare_files
from evaluation.evaluate_webrtc_vad import webrtc_vad_accuracy


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
    # plt.show()


if __name__ == '__main__':
    main()
