
from collections import OrderedDict
import torch

recognition_models = {'cyrillic_g2': {
    'filename': 'cyrillic_g2.pth',
    'model_script': 'cyrillic',
    'url': 'https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.1/cyrillic_g2.zip',
    'md5sum': '19f85f43d9128a89ac21b8d6a06973fe',
    'symbols': '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €₽',
    'characters': '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €₽ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяЂђЃѓЄєІіЇїЈјЉљЊњЋћЌќЎўЏџҐґҒғҚқҮүҲҳҶҷӀӏӢӣӨөӮӯ'
}}


def load_weights(model, device, weight_path):
    state_dict = torch.load(weight_path, map_location=device)
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:]
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)


def load_model(device, backend='vgg'):
    """
    Load CRNN architecture based on VGG_FeatureExtractor or ResNet_FeatureExtractor
    :param weight_path:
    :param model_path:
    :return: net
    """

    # network params
    input_channel = 1 # for Feature extractor
    output_channel = 256
    hidden_size = 256
    num_classes = 208


    if backend == 'resnet':
        from resnet_crnn_model import CRNN
        net = CRNN(input_channel, output_channel, hidden_size, num_classes)
        weight_path = './crnn_pytorch_master/data/crnn.pt'
        load_weights(net, device, weight_path)

    if backend == 'vgg':
        from .vgg_crnn_model import VGG
        weight_path = 'D:/Sveta/Screenshots_project/easy_ocr_detection/weights/cyrillic_g2.pth'
        net = VGG(input_channel, output_channel, hidden_size, num_classes)
        load_weights(net, device, weight_path)
        print("VGG loaded")

    model_size_bytes = sum(p.numel() * p.element_size() for p in net.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)  # Convert bytes to megabytes
    print("Model VGG size:", model_size_mb, "MB")

    if device == 'cuda':
        net = net.cuda()
    return net

