import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from .models.model_loader import load_model
#from .converte import decoding
#from .converter2 import decoding
from .converter3 import decoding_boxes

device = 'cpu'
class WordDataset(Dataset):
    def __init__(self, data, aug_transform=False, allign_ratio = None):
        self.text_boxes = data
        self.aug_transform = aug_transform
        self.augment_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((32, 250)),  # h,w or add collate_with_padding for creating batches
                            transforms.Lambda(lambda x: F.adjust_contrast(x, contrast_factor=0.5)),  # Adjust contrast
                            transforms.Normalize((0.1307,), (0.3081,)) ]) # transforms.Normalize((0.5), (0.5))])

        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((32, 100)),  # h,w (32, 100)
                            transforms.Normalize((0.1307,), (0.3081,)) ]) # transforms.Normalize((0.5), (0.5))])

    def __len__(self):
        return len(self.text_boxes)

    def __getitem__(self, idx):

        sample = self.text_boxes[idx]
        if self.aug_transform:
            return self.augment_transform(sample)
        else:
            sample = self.transform(sample)
        return sample


def run(data, augment_transform):
    """
    Perform batch recognition. From one image we have a set of extracted cropped images.
    :param data: List of cropped images with a text only
    :return:
    """
    data_set = WordDataset(data, augment_transform)
    b_size = len(data_set)
    data_loader = DataLoader(data_set, shuffle=False, batch_size=b_size)

    model = load_model(device, backend='vgg')
    model.eval()
    model = model.to(device)

    # evaluate CRNN
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            preds = model(batch)

    # I need to calibrate this model first, save it and upload it again


    # decode
    #print("Decoding CRNN output", preds.shape)

    #res_text = decoding(preds) # from one image

    res_text, r = decoding_boxes(preds)  # from one image



    return res_text








