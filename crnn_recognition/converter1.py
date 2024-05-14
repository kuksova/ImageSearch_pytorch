# This I am trying ti run converter from  crnn.pytorch

import torch.nn.functional as F
import torch
import numpy as np

from .srtlabel_conv import strLabelConverter

characters = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €₽ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяЂђЃѓЄєІіЇїЈјЉљЊњЋћЌќЎўЏџҐґҒғҚқҮүҲҳҶҷӀӏӢӣӨөӮӯ'

device = 'cpu'

def custom_mean(x):
    return x.prod()**(2.0/np.sqrt(len(x)))

def decode_greedy(preds, preds_size):
    # greedy approach . It's the same naive decodong from the original repository
    #print("greedy ", preds)
    char_list = []
    for i in range(preds_size):
        if preds[i].item() != 0 and (not (i > 0 and preds[i - 1].item() == preds[i].item())):
            char_list.append(characters[preds[i].item() - 1])
    char_list = ''.join(char_list)

    return char_list


def naive_decoding(preds_in, decoder='greedy'):
    """
    --[[ Create LSTM unit, adapted from https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua

    This is from original repository CRNN.
    CTC out probabilities decoding. We adopt the conditional probability defined in the Connectionist Temporal Classification (CTC) layer from paper.
    Naive, lexicon-free decodong


    ARGS:
      - `preds_in`   : float tensor [nFrame x inputLength x nClasses] [3,24,208]
    RETURNS:
       - `pred`    : int tensor [nFrame x inputLength]
    return result batch text
    """

    res_text = []
    batch_size = preds_in.shape[0]
    for i in range(batch_size):
        preds = torch.squeeze(preds_in[i])

        preds_size = preds.shape[0]

        preds_prob = F.softmax(preds, dim=1)
        preds_index = torch.argmax(preds_prob, dim=1)
        preds_index = preds_index.view(-1)


        preds_str = decode_greedy(preds_index.data.cpu().detach(), preds_size)

        # confidence score
        preds_prob = preds_prob.cpu().detach().numpy()
        values = np.max(preds_prob, axis=1)
        preds_max_prob = values[np.where(values != 0)]

        confidence_score = custom_mean(preds_max_prob)

        print('final text: ', preds_str, 'score ', confidence_score)
        if confidence_score > 0.90:
            #res_text.append([preds_str, confidence_score])
            res_text.append(preds_str.lower())
    return res_text




