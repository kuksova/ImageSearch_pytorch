#Thanks to easyocr code

from modules import ResNet_FeatureExtractor, BidirectionalLSTM
import torch.nn as nn



class BidirectionalLSTM1(nn.Module):
    def __init__(self, nIn, nOut, nHidden):
        super(BidirectionalLSTM, self).__init__()


        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view( T *b, h)

        output = self.embedding(t_rec) # [T*b, nOut]
        output = output.veiw(T, b, -1)

        return output


class CRNN1(nn.Module):
    def __init__(self, CNN):
        super(CRNN, self).__init__()
        self.cnn = CNN



    def forward(self ,x):
        conv_feature_map = self.cnn(x)
        b, c, h, w = conv_feature_map.size()
        assert h== 1, "the height of conv must be 1"

        conv_feature_map = conv_feature_map.squeeze(2)
        conv_feature_map = conv_feature_map.permute(2, 0, 1)  # [w,b,c]

        output = self.rnn(conv_feature_map)

        return output

class CRNN(nn.Module):
    def __init__(self, in_ch, out_ch, n_hidden, n_classes):
        """

        :param in_ch: for FeatureExtractor
        :param out_ch: for FeatureExtractor
        :param num_classes: for CRNN prediction output
        """
        super(CRNN, self).__init__()

        """Feature Extraction"""
        self.FeatureExtraction = ResNet_FeatureExtractor(in_ch, out_ch)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """Sequence modeling using RNN"""
        # LSTM ot GRU ????
        self.SequenceModeling = nn.Sequential(
                                BidirectionalLSTM(out_ch, n_hidden, n_hidden),
                                BidirectionalLSTM(n_hidden, n_hidden, n_hidden))

        """Prediction"""
        self.Prediction = nn.Linear(n_hidden, n_classes)
        self.Softmax = nn.Softmax(dim=2)

    def forward(self, input_x):
        """Feature Extraction Stage"""
        visual_feature = self.FeatureExtraction(input_x)
        b, c, h, w = visual_feature.size()
        assert h == 1, "the height of conv must be 1"

        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """Sequence modeling Stage"""
        contextual_feature = self.SequenceModeling(visual_feature)

        """Predcition Stage"""
        prediction = self.Prediction(contextual_feature.contiguous())

        #pred = self.Softmax(prediction)

        return prediction







#import string
# characters = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €₽ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl'
# characters = string.digits
# num_class = len(character)
