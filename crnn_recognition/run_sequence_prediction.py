import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import transforms
import os

from .models.model_loader import load_model
from .utils_easyocr import CTCLabelConverter

characters = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €₽ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяЂђЃѓЄєІіЇїЈјЉљЊњЋћЌќЎўЏџҐґҒғҚқҮүҲҳҶҷӀӏӢӣӨөӮӯ'

def converter_init():
    characters = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €₽ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяЂђЃѓЄєІіЇїЈјЉљЊњЋћЌќЎўЏџҐґҒғҚқҮүҲҳҶҷӀӏӢӣӨөӮӯ'
    separator_list = {
        'ru': ['\xa2', '\xa3'], # there was th
        'en': ['\xa4', '\xaf  zns5']
    }

    dict_list = {}
    lang_list = ['ru', 'en']
    for lang in lang_list:
        dict_list[lang] = os.path.join('crnn_recognition_my', 'dict', lang + ".txt")

    converter = CTCLabelConverter(characters, separator_list, dict_list)
    num_class = len(converter.character)
    return converter


def decoding(preds, batch_size=1, device='cpu', decoder='greedy'):
    converter = converter_init()
    # Select max probabilty (greedy decoding) then decode index to character
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)

    ######## filter ignore_char, rebalance
    preds_prob = F.softmax(preds, dim=2)
    preds_prob = preds_prob.cpu().detach().numpy()
    # preds_prob[:,:,ignore_idx] = 0.
    pred_norm = preds_prob.sum(axis=2)
    preds_prob = preds_prob / np.expand_dims(pred_norm, axis=-1)
    preds_prob = torch.from_numpy(preds_prob).float().to(device)

    if decoder == 'greedy':
        # Select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds_prob.max(2)
        preds_index = preds_index.view(-1)
        preds_str = converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)


    preds_prob = preds_prob.cpu().detach().numpy()
    values = preds_prob.max(axis=2)
    indices = preds_prob.argmax(axis=2)
    preds_max_prob = []
    for v, i in zip(values, indices):
        max_probs = v[i != 0]
        if len(max_probs) > 0:
            preds_max_prob.append(max_probs)
        else:
            preds_max_prob.append(np.array([0]))
    return preds_str, preds_max_prob

def custom_mean(x):
    return x.prod()**(2.0/np.sqrt(len(x)))


def run():
    device = 'cpu'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 100)),  # h,w
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img_path = 'D:/Sveta/Screenshots_project'
    image_list = []
    for i in range(0, 3):
        nm = str(i) + 'cropped_image.jpg'
        im_path = os.path.join(img_path, nm)
        image = Image.open(im_path).convert('L')
        # print(image.size)
        # image = image.resize(1,100, 32))
        # image = np.array(image)
        image = transform(image)
        image_list.append(image)


    model = load_model(device, backend='vgg')
    model.eval()

    batch_size = 1
    imgW = 100
    batch_max_length = int(imgW / 10)

    result = []

    for x in image_list:
        x = torch.unsqueeze(x, 0)
        # x = x.view(1, *x.size())
        #x = Variable(x)
        #print(x.shape)

        # For max length prediction
        length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)

        # [1, 1, 32, 100] -> [1, 256, 1, 24] -> [1, 24, 256] -> [1, 24, 208]
        preds = model(x) # text_for_pred
        """
        print(preds.shape)


        preds_prob = F.softmax(preds, dim=2)
        print(preds_prob.shape)
        preds_prob = preds_prob.cpu().detach().numpy()
        max_inds = np.argmax(preds_prob, axis=2)[0]
        print(max_inds)

        decoded_text = []
        for m in max_inds:
            char = characters[m]
            decoded_text.append(char)
        print(decoded_text)

        # check identical charactes follows to each other
        merged_text = []
        prev_char = ''
        for char in decoded_text:
            if char != prev_char:
                merged_text.append(char)
            prev_char = char
        decoded_text = ''.join(merged_text)
        print(decoded_text)
        print(len(characters))
        #break
        """




        pred_srt, preds_max_prob = decoding(preds)



    for pred, pred_max_prob in zip(pred_srt, preds_max_prob):
        confidence_score = custom_mean(pred_max_prob)
        result.append([pred, confidence_score])

        print(pred_srt, preds_max_prob)
        print(len(pred_srt))
        print(result)
        break

    #print("Recognition time per the whole image = ", time.time() - t0)

def run_one_sample(img_box):
    device = 'cpu'
    batch_size = 1
    imgW = 100
    batch_max_length = int(imgW / 10)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 100)),  # h,w
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img_box = transform(img_box)

    x = torch.unsqueeze(img_box, 0)
    # x = x.view(1, *x.size())
    # x = Variable(x)
    # print(x.shape)

    model = load_model(device, backend='vgg')
    model.eval()

    # [1, 1, 32, 100] -> [1, 256, 1, 24] -> [1, 24, 256] -> [1, 24, 208]
    preds = model(x)

    pred_srt, preds_max_prob = decoding(preds)


    for pred, pred_max_prob in zip(pred_srt, preds_max_prob):
        confidence_score = custom_mean(pred_max_prob)

    return pred_srt, confidence_score

if __name__ == "__main__":

    run()