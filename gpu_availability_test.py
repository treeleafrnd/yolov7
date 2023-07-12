import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU is available')
    print('CUDA device:', torch.cuda.get_device_name(device))
else:
    device = torch.device('cpu')
    print('CPU')
    
