# Code of CTC labelconverter is taken from easyOCR

import torch
import numpy as np

from .utils import ctcBeamSearch, word_segmentation

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, separator_list = {}, dict_pathlist = {}):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

        self.separator_list = separator_list
        separator_char = []
        for lang, sep in separator_list.items():
            separator_char += sep
        self.ignore_idx = [0] + [i+1 for i,item in enumerate(separator_char)]

        ####### latin dict
        if len(separator_list) == 0:
            dict_list = []
            for lang, dict_path in dict_pathlist.items():
                try:
                    with open(dict_path, "r", encoding = "utf-8-sig") as input_file:
                        word_count =  input_file.read().splitlines()
                    dict_list += word_count
                except:
                    pass
        else:
            dict_list = {}
            for lang, dict_path in dict_pathlist.items():
                with open(dict_path, "r", encoding = "utf-8-sig") as input_file:
                    word_count =  input_file.read().splitlines()
                dict_list[lang] = word_count

        self.dict_list = dict_list

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode_greedy(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]
            # Returns a boolean array where true is when the value is not repeated
            a = np.insert(~((t[1:]==t[:-1])),0,True)
            # Returns a boolean array where true is when the value is not in the ignore_idx list
            b = ~np.isin(t,np.array(self.ignore_idx))
            # Combine the two boolean array
            c = a & b
            # Gets the corresponding character according to the saved indexes
            text = ''.join(np.array(self.character)[t[c.nonzero()]])
            texts.append(text)
            index += l
        return texts

    def decode_beamsearch(self, mat, beamWidth=5):
        texts = []
        for i in range(mat.shape[0]):
            t = ctcBeamSearch(mat[i], self.character, self.ignore_idx, None, beamWidth=beamWidth)
            texts.append(t)
        return texts

    def decode_wordbeamsearch(self, mat, beamWidth=5):
        texts = []
        argmax = np.argmax(mat, axis = 2)

        for i in range(mat.shape[0]):
            string = ''
            # without separators - use space as separator
            if len(self.separator_list) == 0:
                space_idx = self.dict[' ']

                data = np.argwhere(argmax[i]!=space_idx).flatten()
                group = np.split(data, np.where(np.diff(data) != 1)[0]+1)
                group = [ list(item) for item in group if len(item)>0]

                for j, list_idx in enumerate(group):
                    matrix = mat[i, list_idx,:]
                    t = ctcBeamSearch(matrix, self.character, self.ignore_idx, None,\
                                      beamWidth=beamWidth, dict_list=self.dict_list)
                    if j == 0: string += t
                    else: string += ' '+t

            # with separators
            else:
                words = word_segmentation(argmax[i])

                for word in words:
                    matrix = mat[i, word[1][0]:word[1][1]+1,:]
                    if word[0] == '': dict_list = []
                    else: dict_list = self.dict_list[word[0]]
                    t = ctcBeamSearch(matrix, self.character, self.ignore_idx, None, beamWidth=beamWidth, dict_list=dict_list)
                    string += t
            texts.append(string)
        return texts
