# Name of folder to save the data files in.
import torch

DATA_FOLDER = "/home/kevin/Documents/voice-activity-detection/data"
NOISE_FOLDER = '/home/kevin/Documents/voice-activity-detection/data/QUT-NOISE'
SPEECH_FOLDER = '/home/kevin/Documents/voice-activity-detection/data/LibriSpeech'

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
    out = torch.max(out, 1)[1].float()
    eq = torch.eq(out, y.float()).float()
    return torch.mean(eq)

