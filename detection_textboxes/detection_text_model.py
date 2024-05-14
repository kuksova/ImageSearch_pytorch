from detection_textboxes_my.craft import CRAFT
from detection_textboxes_my.craft_utils import getDetBoxes, adjustResultCoordinates

import torch

from collections import OrderedDict
import numpy as np
import os




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

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def get_detector(device, quantize=True, cudnn_benchmark=False):

    net = CRAFT()
    trained_model_path = "./detection_textboxes_my/weights/craft_mlt_25k.pth"
    net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location=device)))

    model_size_bytes = sum(p.numel() * p.element_size() for p in net.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)  # Convert bytes to megabytes
    print_size_of_model(net)
    print("Model CRAFT size:", model_size_mb, "MB")


    # if device == 'cpu':
    #    net.load_state_dict(torch.load(trained_model, map_location=device))
    #    if quantize:
    #        try:
    #            torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
    #        except:
    #            pass
    # else:
    #    net.load_state_dict(torch.load(trained_model, map_location=device))
    #    net = torch.nn.DataParallel(net).to(device)
    #    cudnn.benchmark = cudnn_benchmark

    net = net.to(device)

    net.eval()

    return net

def test_net(net, x, device, ratio_w, ratio_h):
    #if isinstance(image, np.ndarray) and len(image.shape) == 4: # as a batch

    # resize
    #resize_aspect_ratio(img, canvas_size

    #preprocessing
    # add if you need it for the model

    x = x.to(device)

    with torch.no_grad():
        y, feature = net(x)

    result = process_out(y, feature, ratio_w, ratio_h)

    return result

def process_out(y, fea, ratio_w, ratio_h):
    text_threshold = 0.7
    low_text = 0.4
    link_threshold = 0.4
    estimate_num_chars=False
    poly = False
    boxes_list, polys_list = [], []
    for out in y:
        # make score and link map
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys, mapper = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if estimate_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

    boxes_list = []
    for polys in polys_list:
        single_img_result = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)
        boxes_list.append(single_img_result)

    return boxes_list  # polys_list

