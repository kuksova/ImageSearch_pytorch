# Quantization pre-trained weights for inference only
# FP32 -> INT8

# Dynamic quantization doesn't decrease the model size

# 1. Load Pre-trained Weights
# 2. Prepare a Representative Dataset
# 3. Specify quantization configuration
# 4. Calibrate with the Representative Dataset
# 5. Convert to quantized model

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms
from collections import OrderedDict
import io

from torch.quantization import convert, prepare, quantize_dynamic, default_qconfig, default_qat_qconfig


from detection_textboxes_my.craft import CRAFT

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class ImagesDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = sorted(os.listdir(self.data_path))
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((1280, 608)),  # h,w
                # transforms.Resize((32,100)), # h,w
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_path_im = os.path.join(self.data_path, self.data[idx])

        image = Image.open(full_path_im).convert('RGB')
        image = image.resize((608, 1280))  # w, h
        image = np.array(image)  # h, w

        if self.transform:
            image = self.transform(image)
        return image


def load_model(model_file):
    model = CRAFT()

    model.load_state_dict(copyStateDict(torch.load(model_file, map_location='cpu')))
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def quatization(model_path, data):

    data_set = ImagesDataset(data)
    b_size = len(data_set)
    for i in data_set:
        print(i.shape)
    data_loader = DataLoader(data_set, shuffle=False, batch_size=b_size)

    #model = load_model(model_path)
    model = CRAFT()

    model.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')))
    #model = torch.load(model_path, map_location='cpu')
    model.to('cpu')

    model.eval()

    print("Size of model before quantization")
    print_size_of_model(model)


    # Fuse Conv, bn and relu
    #model.fuse_model()


    model.qconfig = torch.ao.quantization.default_qconfig
    print(model.qconfig)
    qmodel = torch.quantization.prepare(model, inplace=False)

    qmodel.eval()
    # Collect Calibration statistics
    with torch.no_grad():
        for image in data_loader:
            output = model(image)

    # Convert to quantized model
    torch.quantization.convert(qmodel, inplace=True)

    ##qmodel.load_state_dict(torch.load(model_path, map_location='cpu')) # https://discuss.pytorch.org/t/how-to-load-quantized-model-for-inference/140283

    print("Size of model after quantization")
    print_size_of_model(qmodel)

    # Save to file
    #torch.jit.script(qmodel).save("quantized.pt")
    #torch.jit.save(torch.jit.script(model), './weights/quantized_craft_mlt_25k.pt')


    #model1 = load_model('./weights/quantized_craft_mlt_25k.pth')
    #model1.eval()

    print("Size of model uploaded quantizated model")
    print_size_of_model(qmodel)

    # Evaluation using quantizated model
    with torch.no_grad():
        for image in data_loader:
            output = qmodel(image)

def quatization1(model_path, data):
    data_set = ImagesDataset(data)
    b_size = len(data_set)
    #for i in data_set:
    #    print(i.shape)
    data_loader = DataLoader(data_set, shuffle=False, batch_size=b_size)

    #model = load_model(model_path)
    model = CRAFT()


    model.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')))
    #model = torch.load(model_path, map_location='cpu')
    state_dict = torch.load(model_path, map_location='cpu')
    #model.load_state_dict(state_dict)

    model.to('cpu')
    model.eval()
    model.fuse_model()

    print("Size of model before quantization")
    print_size_of_model(model)


    # model.qconfig = torch.quantization.get_default_config('fbgemm')
    model.qconfig = torch.ao.quantization.default_qconfig
    # insert observers
    torch.quantization.prepare(model, inplace=True)

    # convert to quantized version
    torch.quantization.convert(model, inplace=True)

    print("Size of model after quantization")
    print_size_of_model(model)

    # Evaluation using quantizated model
    with torch.no_grad():
        for image in data_loader:
            output = model(image)

if __name__ == "__main__":

    detection_model_path = "./weights/craft_mlt_25k.pth"
    data = '../_IMAGESTORAGE/09_Sep/'

    quatization1(detection_model_path, data)

