import torch.nn.functional as F
import torch
import numpy as np
import os

from.CTCLabelConverter import CTCLabelConverter

characters = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €₽ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяЂђЃѓЄєІіЇїЈјЉљЊњЋћЌќЎўЏџҐґҒғҚқҮүҲҳҶҷӀӏӢӣӨөӮӯ'

recognition_models = {'cyrillic_g2': {
    'filename': 'cyrillic_g2.pth',
    'model_script': 'cyrillic',
    'url': 'https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.1/cyrillic_g2.zip',
    'md5sum': '19f85f43d9128a89ac21b8d6a06973fe',
    'symbols': '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €₽',
    'characters': '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €₽ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяЂђЃѓЄєІіЇїЈјЉљЊњЋћЌќЎўЏџҐґҒғҚқҮүҲҳҶҷӀӏӢӣӨөӮӯ'
}}

device = 'cpu'

BASE_PATH = './dict'

# 1. Get decoder
# 2. Decoding algorithm
# 3. Predict first round
# 4. Predict 2nd round
# 5.     # default mode will try to process multiple boxes at the same time
#     # or without gpu/parallelization, it is faster to process image one by one

def custom_mean(x):
    return x.prod()**(2.0/np.sqrt(len(x)))


def decoding_boxes(preds):
    batch_size = preds.shape[0]
    device=('cpu')
    character = recognition_models['cyrillic_g2']['characters']

    ignore_char = ''
    ignore_idx = []
    for char in ignore_char:
        try: ignore_idx.append(character.index(char)+1)
        except: pass

    #separator_list = {
    #    'th': ['\xa2', '\xa3'],
    #    'en': ['\xa4', '\xa5']
    #}
    separator_list = {}
    dict_list = {}
    lang_list = ['en', 'ru']
    for lang in lang_list:
        dict_list[lang] = os.path.join(BASE_PATH, 'dict', lang + ".txt")
    result = []

    # Don't forget about quantization and paralellism
    converter = CTCLabelConverter(character, separator_list, dict_list)
    num_class = len(converter.character)


    # Select max probabilty (greedy decoding) then decode index to character
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)


    ######## filter ignore_char, rebalance
    preds_prob = F.softmax(preds, dim=2)
    preds_prob = preds_prob.cpu().detach().numpy()
    preds_prob[:, :, ignore_idx] = 0.
    pred_norm = preds_prob.sum(axis=2)
    preds_prob = preds_prob / np.expand_dims(pred_norm, axis=-1)
    preds_prob = torch.from_numpy(preds_prob).float().to(device)

    # Select max probabilty (greedy decoding) then decode index to character
    _, preds_index = preds_prob.max(2)
    preds_index = preds_index.view(-1)
    preds_str = converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)

    # Choose converter
    decoder = 'wordbeamsearch'
    beamWidth = 5

    if decoder == 'greedy':
        # Select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds_prob.max(2)
        preds_index = preds_index.view(-1)
        preds_str = converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)
    elif decoder == 'beamsearch':
        k = preds_prob.cpu().detach().numpy()
        preds_str = converter.decode_beamsearch(k, beamWidth=beamWidth)
    elif decoder == 'wordbeamsearch':
        k = preds_prob.cpu().detach().numpy()
        preds_str = converter.decode_wordbeamsearch(k, beamWidth=beamWidth)

    preds_prob = preds_prob.cpu().detach().numpy()
    values = preds_prob.max(axis=2)
    indices = preds_prob.argmax(axis=2)
    preds_max_prob = []
    for v,i in zip(values, indices):
        max_probs = v[i!=0]
        if len(max_probs)>0:
            preds_max_prob.append(max_probs)
        else:
            preds_max_prob.append(np.array([0]))
    r = []
    for pred, pred_max_prob in zip(preds_str, preds_max_prob):
        confidence_score = custom_mean(pred_max_prob)
        result.append([pred, confidence_score])
        r.append(pred)


    return result, r # or I can preds_str already for the 1st round

# Extracted text  ['8:08', 'гся', 'вперед', 'не', 'ОСТаВЛЯ', 'ЮОреШ', 'ам', 'ни', 'СДИНГО', 'шанса', 'send', '(lldii', 'message', 'sh', 'все', '7', 'вами', 'яснО', 'досуг', '24', 'вырывае']
# Extracted text  [['8:08', 0.8213773369789124], ['гся', 0.9730203148890963], ['вперед', 0.941273426176592], ['не', 0.428929593000009], ['ОСТаВЛЯ', 0.1554080473983978], ['ЮОреШ', 0.30067028977420945], ['ам', 0.99996965440393], ['ни', 0.9904330747788211], ['СДИНГО', 0.43017323134095126], ['шанса', 0.6575094925833336], ['send', 0.9279707670211792], ['(lldii', 0.0378410396171497], ['message', 0.9864752591646576], ['sh', 0.9644514771438927], ['все', 0.9764503615327602], ['7', 0.4169343884030674], ['вами', 0.9999416470527649], ['яснО', 0.6975758671760559], ['досуг', 0.7231102010664203], ['24', 0.17136925281304888], ['вырывае', 0.9623472042518221]]