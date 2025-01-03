"""
This script provides functionality for training, testing, and evaluating various neural network models for voice activity detection (VAD). It includes the following key components:

1. **Imports and Constants**:
    - Imports necessary libraries and modules.
    - Defines constants and configurations for training and testing.

2. **Model Definitions**:
    - Defines a stack of models with different configurations and parameters.

3. **Utility Functions**:
    - `test_network(data)`: Tests the network with given data.
    - `initialize_network()`: Initializes the network and prints the number of parameters.
    - `FocalLoss`: Custom loss function for handling class imbalance.
    - `net_path(epoch, title)`: Generates the file path for saving/loading network models.
    - `save_net(net, epoch, title)`: Saves the network model to disk.
    - `load_net(epoch, title)`: Loads the network model from disk.
    - `train_net(net, data, ...)`: Trains the network with specified parameters.
    - `set_seed(seed)`: Sets the random seed for reproducibility.
    - `test_predict(net, data, size_limit, noise_level)`: Computes predictions on test data using the given network.
    - `roc_auc(nets, data, noise_lvl, size_limit)`: Generates a ROC curve for the given network and data.
    - `far(net, data, size_limit, frr, model_name)`: Computes the confusion matrix for a given network.
    - `netvad(net, data, noise_level, init_pos, length, title, timeit)`: Generates a sample and runs it through the network, plotting the results.
    - `get_model(data, model, model_name)`: Trains or loads a model based on the configuration.
    - `train_all_models(data)`: Trains all models in the stack and evaluates them.

4. **MFCC (Mel-Frequency Cepstral Coefficients)**:
    - MFCC is a feature extraction technique commonly used in audio processing. It represents the short-term power spectrum of sound and is used to capture the characteristics of audio signals.
    - In this script, MFCC is used to extract features from audio data, which are then fed into the neural network models for training and evaluation.
"""

import os
import time
import torch
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch import nn as nn, optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

from core.common import num_params, accuracy, BATCH_SIZE, FRAMES, FEATURES, OBJ_CUDA, NOISE_LEVELS_DB
from core.generator import DataGenerator
from core.models import Net, MODEL_STACK
from core.prepare_strong_files import prepare_strong_files
from core.process_data import process_test_data
from core.visualization import Vis


OBJ_TRAIN_MODELS = True
NOISE_LEVELS = list(NOISE_LEVELS_DB.keys())
STEP_SIZE = 6
MAX_EPOCHS = 14
SEED = 1337


def test_network(data):

    # Test generator
    generator = DataGenerator(data)
    generator.setup_generation(frame_count=FRAMES, step_size=STEP_SIZE, batch_size=BATCH_SIZE)
    generator.use_train_data()
    generator.set_noise_level_db('None')

    print(generator.batch_count, 'training batches were found.')

    # Compact instantiation of untrained network on CPU
    OBJ_CUDA = torch.cuda.is_available()
    temp, OBJ_CUDA = OBJ_CUDA, False
    net, OBJ_CUDA = Net(large=False), temp
    del temp

    # Run a few batches
    for i in range(3):
        # Get batch
        X, y = generator.get_batch(i)
        X = torch.from_numpy(np.array(X)).float()
        y = torch.from_numpy(np.array(y)).long()

        if OBJ_CUDA:
            X = X.cuda()
            y = y.cuda()

        # Run through network
        out = net(X)
        acc = accuracy(out, y).data.cpu().numpy()

    print('Successfully ran the network!\n\nExample output:', out.data.cpu().numpy()[0])


def initialize_network():

    net = Net(large=False)
    num_params(net)
    print(net)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=0)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def net_path(epoch, title):
    part = os.getcwd() + '/models/' + title
    if epoch >= 0:
        return part + '_epoch' + str(epoch).zfill(3) + '.net'
    else:
        return part + '.net'


def save_net(net, epoch, title='net'):
    if not os.path.exists(os.getcwd() + '/models'):
        os.makedirs(os.getcwd() + '/models')
    torch.save(net, net_path(epoch, title))


def load_net(epoch=MAX_EPOCHS, title='net'):
    if OBJ_CUDA:
        return torch.load(net_path(epoch, title))
    else:
        return torch.load(net_path(epoch, title), map_location='cpu')


def train_net(net, data, size_limit=0, noise_level='None', epochs=15, lr=1e-3, use_adam=True,
              weight_decay=1e-5, momentum=0.9, use_focal_loss=True, gamma=0.0,
              early_stopping=False, patience=25, frame_count=FRAMES, step_size=STEP_SIZE,
              auto_save=True, title='net', verbose=True):
    """
    Full-featured training of a given neural network.
    A number of training parameters are optionally adjusted.
    If verbose is True, the training progress is continously
    plotted at the end of each epoch.
    If auto_save is True, the model will be saved every epoch.
    """

    # Set up an instance of data generator using default partitions
    generator = DataGenerator(data, size_limit)
    generator.setup_generation(frame_count, step_size, BATCH_SIZE)

    if noise_level not in NOISE_LEVELS:
        print('Error: invalid noise level!')
        return

    if generator.train_size == 0:
        print('Error: no training data was found!')
        return

    # Move network to GPU if available
    if OBJ_CUDA:
        net.cuda()

    # Instantiate the chosen loss function
    if use_focal_loss:
        criterion = FocalLoss(gamma)
        levels = NOISE_LEVELS
    else:
        criterion = nn.CrossEntropyLoss()
        levels = [noise_level]

    # Move criterion to GPU if available
    if OBJ_CUDA:
        criterion.cuda()

    # Instantiate the chosen optimizer with the parameters specified
    if use_adam:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # If verbose, print starting conditions
    if verbose:
        print(f'Initiating training of {title}...\n\nLearning rate: {lr}')
        _trsz = generator.train_size * 3 if use_focal_loss else generator.train_size
        _vlsz = generator.val_size * 3 if use_focal_loss else generator.val_size
        print(f'Model parameters: {sum(p.numel() for p in net.parameters())}')
        print(f'Frame partitions: {_trsz} | {_vlsz}')
        _critstr = f'Focal Loss (γ = {gamma})' if use_focal_loss else f'Cross-Entropy ({noise_level} dB)'
        _optmstr = f'Adam (decay = {weight_decay})' if use_adam else f'SGD (momentum = {momentum})'
        _earlstr = f'Early Stopping (patience = {patience})' if early_stopping else str(epochs)
        _autostr = 'Enabled' if auto_save else 'DISABLED'
        print(f'Criterion: {_critstr}\nOptimizer: {_optmstr}')
        print(f'Max epochs: {_earlstr}\nAuto-save: {_autostr}')

    net.train()
    stalecount, maxacc = 0, 0

    def plot(losses, accs, val_losses, val_accs):
        """
        Continously plots the training/validation loss and accuracy
        of the model being trained. This functions is only called if
        verbose is True for the training session.
        """
        e = [i for i in range(len(losses))]
        # fig = plt.figure(figsize=(12, 4))

        fig = plt.gcf()
        fig.set_figwidth(12)
        fig.set_figheight(4)

        plt.subplot(1, 2, 1)
        plt.ion()
        plt.show()

        for ax in fig.axes:
            # lines = ax.get_lines()
            for line in ax.get_lines():  # ax.lines:
                line.remove()

        plt.plot(e, losses, label='Loss (Training)', color='r')

        if generator.val_size != 0:
            plt.plot(e, val_losses, label='Loss (Validation)', color='b')

        plt.legend()
        plt.title('Loss')
        plt.subplot(1, 2, 2)
        plt.plot(e, accs, label='Accuracy (Training)', color='r')

        if generator.val_size != 0:
            plt.plot(e, val_accs, label='Accuracy (Validation)', color='b')

        plt.legend()
        plt.title('Accuracy')
        plt.suptitle(f'Training progress of {title}')
        # plt.show()
        plt.draw()
        plt.pause(0.001)
        clear_output(wait=True)

    def run(net, optimize=False):
        '''
        This function constitutes a single epoch.
        Snippets are loaded into memory and their associated
        frames are loaded as generators. As training progresses
        and new frames are needed, they are generated by the iterator,
        and are thus not stored in memory when not used.
        If optimize is True, the associated optimizer will backpropagate
        and adjust network weights.
        Returns the average sample loss and accuracy for that epoch.
        '''
        epoch_loss, epoch_acc, level_acc = 0, [], []

        # In case we apply focal loss, we want to include all noise levels
        batches = generator.batch_count
        num_batches = batches * len(levels)

        if num_batches == 0:
            raise ValueError('Not enough data to create a full batch!')

        # Helper function responsible for running a batch
        def run_batch(X, y, epoch_loss, epoch_acc):

            X = Variable(torch.from_numpy(np.array(X)).float())
            y = Variable(torch.from_numpy(np.array(y))).long()

            if OBJ_CUDA:
                X = X.cuda()
                y = y.cuda()

            out = net(X)

            # Compute loss and accuracy for batch
            batch_loss = criterion(out, y)
            batch_acc = accuracy(out, y)

            # If training session, initiate backpropagation and optimization
            if optimize == True:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            if OBJ_CUDA:
                batch_acc = batch_acc.cpu()
                batch_loss = batch_loss.cpu()

            # Accumulate loss and accuracy for epoch metrics
            epoch_loss += batch_loss.data.numpy() / float(BATCH_SIZE)
            epoch_acc.append(batch_acc.data.numpy())

            return epoch_loss, epoch_acc

        # For each noise level scheduled
        for lvl in levels:

            # Set up generator for iteration
            generator.set_noise_level_db(lvl)

            # For each batch in noise level
            for i in range(batches):
                # Get a new batch and run it
                X, y = generator.get_batch(i, skip_single_class=True)
                if len(X)==0 or len(y)==0:
                    continue
                temp_loss, temp_acc = run_batch(X, y, epoch_loss, epoch_acc)
                epoch_loss += temp_loss / float(num_batches)
                level_acc.append(np.mean(temp_acc))

        return epoch_loss, np.mean(level_acc)

    losses, accs, val_losses, val_accs = [], [], [], []

    if verbose:
        start_time = time.time()

    # Iterate over training epochs
    for epoch in range(epochs):

        # Calculate loss and accuracy for that epoch and optimize
        generator.use_train_data()
        loss, acc = run(net, optimize=True)
        losses.append(loss)
        accs.append(acc)

        # If validation data is available, calculate validation metrics
        if generator.val_size != 0:
            net.eval()
            generator.use_validate_data()
            val_loss, val_acc = run(net)
            # print(val_loss, val_acc)
            # return
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            net.train()

            # Early stopping algorithm.
            # If validation accuracy does not improve for
            # a set amount of epochs, abort training and retrieve
            # the best model (according to validation accuracy)
            if epoch > 0 and val_accs[-1] <= maxacc:
                stalecount += 1
                if stalecount > patience and early_stopping:
                    return
            else:
                stalecount = 0
                maxacc = val_accs[-1]

        if auto_save:
            save_net(net, epoch, title)

        # Optionally plot performance metrics continuously
        if verbose:

            # Print measured wall-time of first epoch
            if epoch == 0:
                dur = str(int((time.time() - start_time) / 60))
                print(f'\nEpoch wall-time: {dur} min')

            plot(losses, accs, val_losses, val_accs)
        
    # Save the figure
    fig = plt.gcf()
    fig.savefig(os.path.join(os.getcwd(), 'models', f'{title}_training_plot.png'))
    plt.close(fig)


def set_seed(seed = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if OBJ_CUDA:
        torch.cuda.manual_seed_all(seed)


def test_predict(net, data, size_limit, noise_level):
    """
    Computes predictions on test data using given network.
    """

    # Set up an instance of data generator using default partitions
    generator = DataGenerator(data, size_limit)
    generator.setup_generation(FRAMES, STEP_SIZE, BATCH_SIZE)

    if noise_level not in NOISE_LEVELS:
        print('Error: invalid noise level!')
        return

    if generator.test_size == 0:
        print('Error: no test data was found!')
        return

    net.eval()
    generator.use_test_data()
    generator.set_noise_level_db(noise_level)

    y_true, y_score = [], []

    for i in range(generator.batch_count):
        X, y = generator.get_batch(i)

        # Check if batch is not empty before accessing elements
        if len(X) == 0 or len(y) == 0:
            continue

        # print(f"Batch {i} - Class distribution:", np.bincount(y))
        X = Variable(torch.from_numpy(np.array(X)).float())
        y = Variable(torch.from_numpy(np.array(y))).long()

        try:
            out = net(X)
        except RuntimeError:
            out = net(X.cuda())

        if OBJ_CUDA:
            out = out.cpu()
            y = y.cpu()

        # Add true labels.
        y_true.extend(y.data.numpy())

        # Add probabilities for positive labels.
        y_score.extend(out.data.numpy()[:, 1])

    print("y_true distribution:", np.bincount(y_true))
    print("y_score distribution:", np.unique(y_score, return_counts=True))

    return y_true, y_score


def roc_auc(nets, data, noise_lvl, size_limit=0):
    """
    Generates a ROC curve for the given network and data for each noise level.
    """

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    plt.title('Receiver Operating Characteristic (%s)' % noise_lvl, fontsize=16)

    # For each network
    for key in nets:
        net = nets[key]

        # Make predictions
        y_true, y_score = test_predict(net, data, size_limit, noise_lvl)

        # Compute ROC metrics and AUC
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc_res = metrics.auc(fpr, tpr)

        # Plots the ROC curve and show area.
        plt.plot(fpr, tpr, label='%s (AUC = %0.3f)' % (key, auc_res))

    plt.xlim([0, 0.2])
    plt.ylim([0.6, 1])
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(loc='lower right', prop={'size': 16})

    return fig


def far(net, data, size_limit=0, frr=1, model_name=''):
    """
    Computes the confusion matrix for a given network.
    """

    # Evaluate predictions using threshold
    def apply_threshold(y_score, t=0.5):
        return [1 if y >= t else 0 for idx, y in enumerate(y_score)]

    def fix_frr(y_true, y_score, frr_target, noise_level):

        # Quick hack for initial threshold level to hit 1% FRR a bit faster.
        if noise_level == 'None':
            t = 1e-4
        elif noise_level == '-15':
            t = 1e-5
        else:
            t = 1e-9

        # Compute FAR for a fixed FRR
        while t < 1.0:
            cm = confusion_matrix(y_true, apply_threshold(y_score, t), labels=[0, 1])

            # Check if confusion matrix has the expected shape
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle the case where the confusion matrix does not have the expected shape
                tn, fp, fn, tp = 0, 0, 0, 0
                if cm.shape == (1, 1):
                    if y_true[0] == 0:
                        tn = cm[0, 0]
                    else:
                        tp = cm[0, 0]
                elif cm.shape == (1, 2):
                    tn, fp = cm[0, 0], cm[0, 1]
                elif cm.shape == (2, 1):
                    fn, tp = cm[1, 0], cm[0, 0]

            far = (fp * 100) / (fp + tn) if (fp + tn) > 0 else 0
            frr = (fn * 100) / (fn + tp) if (fn + tp) > 0 else 0

            if frr >= frr_target:
                return far, frr

            t *= 1.1

        # Return closest result if no good match found.
        return far, frr

    print('Network metrics: ' + model_name)

    # For each noise level
    for lvl in NOISE_LEVELS:
        # Make predictions
        y_true, y_score = test_predict(net, data, size_limit, lvl)
        print('FAR: %0.2f%% for fixed FRR at %0.2f%% and noise level' % fix_frr(y_true, y_score, frr, lvl), lvl)


def netvad(net, data, noise_level='-3', init_pos=50, length=700, title=None, timeit=True):
    """
    Generates a sample of specified length and runs it through
    the given network. By default, the network output is plotted
    alongside the original labels and WebRTC output for comparison.
    """

    # Set up an instance of data generator using default partitions
    generator = DataGenerator(data)
    generator.setup_generation(FRAMES, STEP_SIZE, BATCH_SIZE)

    if noise_level not in NOISE_LEVELS:
        print('Error: invalid noise level!')
        return

    if generator.test_size == 0:
        print('Error: no test data was found!')
        return

    net.eval()
    generator.use_test_data()
    generator.set_noise_level_db(noise_level)

    raw_frames, mfcc, delta, labels = generator.get_data(init_pos, init_pos + length)

    if mfcc is not None:
        # Convert sample to list of frames
        def get_frames():
            i = 0
            while i < length - FRAMES:
                yield np.hstack((mfcc[i: i + FRAMES], delta[i: i + FRAMES]))
                i += 1

        # Creates batches from frames
        frames = list(get_frames())
    else:
        # Convert sample to list of frames
        def get_frames():
            i = 0
            while i < length - FRAMES:
                yield np.hstack((mfcc[i: i + FRAMES], delta[i: i + FRAMES]))
                i += 1
        frames = list(get_frames())

    batches, i, num_frames = [], 0, -1
    while i < len(frames):
        full = i + BATCH_SIZE >= len(frames)
        end = i + BATCH_SIZE if not full else len(frames)
        window = frames[i:end]
        if full:
            num_frames = len(window)
            while len(window) < BATCH_SIZE:
                window.append(np.zeros((FRAMES, FEATURES)))
        batches.append(np.stack(window))
        i += BATCH_SIZE

    # Start timer
    if timeit:
        start_net = time.time()

    # Predict for each frame
    offset = 15
    accum_out = [0] * offset
    for batch in batches:
        X = Variable(torch.from_numpy(batch).float())
        try:
            out = torch.max(net(X.cuda()), 1)[1].cpu().float().data.numpy()
        except RuntimeError:
            out = torch.max(net(X), 1)[1].float().data.numpy()

        accum_out.extend(out)

    # Stop timer
    if timeit:
        dur_net = str((time.time() - start_net) * 1000).split('.')[0]
        device = 'GPU' if OBJ_CUDA else 'CPU'
        seq_dur = int((length / 100) * 3)
        print(f'Network processed {len(batches) * BATCH_SIZE} frames ({seq_dur}s) in {dur_net}ms on {device}.')

    # Adjust padding
    if num_frames > 0:
        accum_out = accum_out[:len(accum_out) - (BATCH_SIZE - num_frames)]
    accum_out = np.array(accum_out)

    # Cut frames outside of prediction boundary
    raw_frames = raw_frames[offset:-offset]
    labels = labels[offset:-offset]
    accum_out = accum_out[offset:]

    # Plot results
    print('Displaying results for noise level:', noise_level)
    Vis.plot_evaluation(raw_frames, labels, accum_out, title=title)


def get_model(data, model, model_name):
    model_path = net_path(MAX_EPOCHS, model_name)
    if OBJ_TRAIN_MODELS and not os.path.exists(model_path):
        set_seed()
        model_dict = MODEL_STACK[model_name]
        train_net(model, data, title=model_name, **model_dict['kwargs'])
    else:
        model = load_net(title=model_name)

    return model


def train_all_models(data):
    trained_models = {}
    for model_name in MODEL_STACK.keys():
        model_path = net_path(MAX_EPOCHS, model_name)
        if OBJ_TRAIN_MODELS and not os.path.exists(model_path):
            set_seed()
            model_dict = MODEL_STACK[model_name]
            model = model_dict['model']
            train_net(model, data, title=model_name, **model_dict['kwargs'])
        else:
            model = load_net(title=model_name)

        # Print model summary
        print(model)

        trained_models[model_name] = model

    for model_name in trained_models.keys():
        fig = roc_auc(trained_models, data, 'None')
        fig.savefig(os.path.join(os.getcwd(), 'models', f'{model_name}_roc_auc_None.png'))
        plt.close(fig)

        fig = roc_auc(trained_models, data, '-15')
        fig.savefig(os.path.join(os.getcwd(), 'models', f'{model_name}_roc_auc_-15.png'))
        plt.close(fig)

        fig = roc_auc(trained_models, data, '-3')
        fig.savefig(os.path.join(os.getcwd(), 'models', f'{model_name}_roc_auc_-3.png'))
        plt.close(fig)

    # Fixed FRR
    print('\nFixed FRR:')
    for model_name in trained_models.keys():
        model = trained_models[model_name]
        far(model, data, frr=1, model_name=model_name)

    # Qualitative results
    print('\nQualitative results:')
    for model_name in trained_models.keys():
        model = trained_models[model_name]
        netvad(model, data, title=model_name)


if __name__ == "__main__":

    # Prepare the STRONG Data
    labeled_strong_dataset = prepare_strong_files()
    train_data = process_test_data(labeled_strong_dataset)

    # test_generator(train_data)
    initialize_network()
    train_all_models(train_data)