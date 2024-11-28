from matplotlib import pyplot as plt
from core.generator import test_generator
from core.process_data import process_training_data
from core.train import test_network, initialize_network, train_all_models, get_model, netvad
from core.models import MODEL_STACK
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

    # Select model for testing
    model_name = 'net_large'
    model_dict = MODEL_STACK[model_name]
    model_type = model_dict['model']
    model = get_model(data, model_type, model_name)

    # # Evaluate model with sample data
    netvad(model, data, title='Training Data')
    # netvad(model, strong_data, init_pos=400, title='STRONG Data')

    plt.show()


if __name__ == '__main__':
    main()
