import torch

from crnn_recognition_my.run_batch_recogn import run

import torchvision.transforms.functional as F

def predict_text(bboxes):

    # Predict first round
    out_result1 = run(bboxes, augment_transform=False)
    print(out_result1)

    # Predict 2nd round for boxes with low confidence score
    # Add augmentations: Change contrast augmentation and Ratio increasing (?) augmentation
    contrast_ths = 0.1
    low_confident_idx = [i for i, item in enumerate(out_result1) if (item[1] < contrast_ths)]

    if len(low_confident_idx) > 0:
        bboxes_list2 = [bboxes[i] for i in low_confident_idx]

        out_result2 = run(bboxes_list2, augment_transform=True)


    # Compare and choose the best
    result = []
    coord = list(range(0,len(bboxes)))
    for i, zipped in enumerate(zip(coord, out_result1)):
        box, pred1 = zipped
        if i in low_confident_idx:
            pred2 = out_result2[low_confident_idx.index(i)]
            if pred1[1]>pred2[1]:
                #result.append( (box, pred1[0], pred1[1]) )
                result.append( pred1[0])
            else:
                # result.append( (box, pred2[0], pred2[1]) )
                result.append(pred2[0])
        else:
            # result.append( (box, pred1[0], pred1[1]) )
            result.append(pred1[0])
    print(result)
    return result