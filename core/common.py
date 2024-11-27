# Name of folder to save the data files in.
import numpy as np
from matplotlib import pyplot as plt
import torch

DATA_FOLDER = "/home/kevin/Documents/voice-activity-detection/data"
NOISE_FOLDER = '/home/kevin/Documents/voice-activity-detection/data/QUT-NOISE'
SPEECH_FOLDER = '/home/kevin/Documents/voice-activity-detection/data/LibriSpeech'
STRONG_PROCESSED_MIC_FOLDER = "/home/kevin/Documents/voice-activity-detection/data/my_audio_files/processedMicAudio"
STRONG_VIDEO_AUDIO_FOLDER = "/home/kevin/Documents/voice-activity-detection/data/my_audio_files/videoAudio"

# Specify the desired WAV-format.
SAMPLE_RATE = 16000
SAMPLE_CHANNELS = 1
SAMPLE_WIDTH = 2

BATCH_SIZE = 2048
FRAMES = 30
FEATURES = 24

NOISE_LEVELS_DB = {'None': None, '-15': -15, '-3': -3}

OBJ_CUDA = torch.cuda.is_available()
if OBJ_CUDA:
    print('CUDA has been enabled.')
else:
    print('CUDA has been disabled.')


def num_params(net, verbose = True):
    count = sum(p.numel() for p in net.parameters())
    if verbose:
        print(f'Model parameters: {count}')
    return count


def accuracy(out, y):
    '''
    Calculate accuracy of model where
    out.shape = (64, 2) and y.shape = (64)
    '''
    if OBJ_CUDA:
        y = y.cuda()
    out = torch.max(out, 1)[1].float()
    eq = torch.eq(out, y.float()).float()
    return torch.mean(eq)


def traverse_datasets(hdf_file):

    """
    Traverse all datasets across all groups in HDF5 file.
    reference: https://stackoverflow.com/questions/50117513/can-you-view-hdf5-files-in-pycharm
    """

    import h5py

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, 'r') as f:
        for (path, dset) in h5py_dataset_iterator(f):
            print(path, dset)

    return None


def create_dictionary(list1, list2):
    dictionary = {}
    for file1 in sorted(list1):
        for file2 in sorted(list2):
            experiment = file2[0:6]
            if file1.startswith(experiment):
                dictionary[file2] = file1
                break
    return dictionary


def plot_labels(data, index_from=None, index_to=None):
    temp_labels = np.array(data['labels'])

    plt.figure(figsize=(12, 8))
    plt.plot(range(len(temp_labels)), temp_labels, label='All Labels')

    # Ensure index_from and index_to are within the bounds of temp_labels
    if index_from is not None and index_to is not None:
        index_from = max(0, min(index_from, len(temp_labels) - 1))
        index_to = max(index_from, min(index_to, len(temp_labels)))
        plt.plot(range(index_from, index_to), temp_labels[index_from:index_to], 'r', label='Selected Labels')

    plt.xlim(0, len(temp_labels))
    plt.legend()
    plt.savefig('labels.png')
    plt.close()
