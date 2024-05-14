import torch.nn as nn
from .modules import VGG_FeatureExtractor, BidirectionalLSTM

class VGG(nn.Module):

    def __init__(self, in_ch, out_ch, hidden_size, num_class):
        super(VGG, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(in_ch, out_ch)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(out_ch, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))

        """ Prediction """
        self.Prediction = nn.Linear(hidden_size, num_class)
        self.Softmax = nn.Softmax(dim=2)


    def forward(self, input):
        """ Feature extraction stage """
        #print('bn')
        #print(input.shape)
        visual_feature = self.FeatureExtraction(input)
        #print("a ", visual_feature.shape)

        b, c, h, w = visual_feature.size()
        assert h == 1, "the height of conv must be 1"

        # [b, c, h, w] -> [b, w, c, h]
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)
        #print(visual_feature.shape)
        #print('bn1')

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())
        # pred = self.Softmax(prediction)

        return prediction