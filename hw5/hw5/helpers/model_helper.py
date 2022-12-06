import torch
import numpy as np

def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def freeze_conv_layer(model):
    for name, param in model.named_parameters():
        if name.startswith('conv'):
            param.requires_grad = False

def init_conv_kernel_with_edge_detector(model):
    # Get kernel size
    kernel_size = model.conv1.kernel_size[0]
    
    # number of filters should be 3
    num_filters = model.conv1.out_channels
    assert num_filters == 3, "Number of filters should be 3"

    if kernel_size == 2:
        # 2 x 2 edge detector
        horizontal_edge_detector = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32)
        vertical_edge_detector = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32)
        none_edge_detector = torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
    
    else:
        horizontal_edge_detector = torch.from_numpy(custom_sobel((kernel_size, kernel_size), 0))
        vertical_edge_detector = torch.from_numpy(custom_sobel((kernel_size, kernel_size), 1))
        none_edge_detector = torch.from_numpy(np.zeros((kernel_size, kernel_size)))

    edge_detector = torch.stack([horizontal_edge_detector, vertical_edge_detector, none_edge_detector])
    model.conv1.weight.data = edge_detector.view(model.num_filter, 1, model.kernel_size, model.kernel_size)
    model.conv2.weight.data = torch.cat([model.conv1.weight.data, model.conv1.weight.data, model.conv1.weight.data], dim=1)

    # type casting
    model.conv1.weight.data = model.conv1.weight.data.type(torch.FloatTensor)
    model.conv2.weight.data = model.conv2.weight.data.type(torch.FloatTensor)

    # bias
    model.conv1.bias.data = torch.tensor([0, 0, 0], dtype=torch.float32)
    model.conv2.bias.data = torch.tensor([0, 0, 0], dtype=torch.float32)

def custom_sobel(shape, axis):
    """
    shape must be odd: eg. (5,5)
    axis is the direction, with 0 to positive x and 1 to positive y
    """
    k = np.zeros(shape, dtype=np.float32)
    p = [(j,i) for j in range(shape[0]) 
           for i in range(shape[1]) 
           if not (i == (shape[1] -1)/2. and j == (shape[0] -1)/2.)]

    for j, i in p:
        j_ = int(j - (shape[0] -1)/2.)
        i_ = int(i - (shape[1] -1)/2.)
        k[j,i] = (i_ if axis==0 else j_)/float(i_*i_ + j_*j_)
    return k