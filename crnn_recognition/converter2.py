import torch.nn.functional as F
import torch
import numpy as np
from scipy.special import logsumexp  # log(p1 + p2) = logsumexp([log_p1, log_p2])

NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01

characters = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €₽ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяЂђЃѓЄєІіЇїЈјЉљЊњЋћЌќЎўЏџҐґҒғҚқҮүҲҳҶҷӀӏӢӣӨөӮӯ'

def _reconstruct(labels, blank=0):
    new_labels = []
    # merge same labels
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # delete blank
    new_labels = [l for l in new_labels if l != blank]

    char_list = []
    for l in new_labels:
        char_list.append(characters[l - 1])
    char_list = ''.join(char_list)

    return char_list

def beam_search_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [([], 0)]  # (prefix, accumulated_log_prob)
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue
                new_prefix = prefix + [c]
                # log(p1 * p2) = log_p1 + log_p2
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))

        # sorted by accumulated_log_prob
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # sum up beams to produce labels
    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix, blank))
        # log(p1 + p2) = logsumexp([log_p1, log_p2])
        total_accu_log_prob[labels] = \
            logsumexp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])

    labels_beams = [(list(labels), accu_log_prob)
                    for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    return labels

def custom_mean(x):
    return x.prod()**(2.0/np.sqrt(len(x)))

def ctc_decode(log_probs, label2char=None, blank=0, method='beam_search', beam_size=10):
    emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
    # size of emission_log_probs: (batch, length, class)

    decoders = {
        #'greedy': greedy_decode,
        'beam_search': beam_search_decode,
        #'prefix_beam_search': prefix_beam_decode,
    }
    decoder = decoders[method]

    decoded_list = []
    for emission_log_prob in emission_log_probs:
        decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size)
        if label2char:
            decoded = [label2char[l] for l in decoded]
        decoded_list.append(decoded)
    return decoded_list

def decoding(preds_in, decoder='beam_search'):
    #decode_method = config['decode_method'],
    #beam_size = config['beam_size']



    decode_method = 'beam_search'
    beam_size = 10

    res_text = []
    batch_size = preds_in.shape[0]
    for i in range(batch_size):
        #preds = torch.squeeze(preds_in[i])

        log_probs = torch.nn.functional.log_softmax(preds_in, dim=2)

        input_lengths = torch.LongTensor([preds_in.size(0)] * batch_size)

        preds_text = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)

        # confidence score
        preds_prob = log_probs.cpu().detach().numpy()
        values = np.max(preds_prob, axis=1)
        preds_max_prob = values[np.where(values != 0)]

        confidence_score = custom_mean(preds_max_prob)

        print('final text: ', preds_text, 'score ', confidence_score)

        """
        reals = targets.cpu().numpy().tolist()
        target_lengths = target_lengths.cpu().numpy().tolist()
    
        tot_count += batch_size
        tot_loss += loss.item()
        target_length_counter = 0
        for pred, target_length in zip(preds, target_lengths):
            real = reals[target_length_counter:target_length_counter + target_length]
            target_length_counter += target_length
            if pred == real:
                tot_correct += 1
            else:
                wrong_cases.append((real, pred))
        evaluation = {
            'loss': tot_loss / tot_count,
            'acc': tot_correct / tot_count,
            'wrong_cases': wrong_cases
        }
        """
