import time
import onnxruntime as ort
import numpy as np
import cv2


dic_labels = {
          0: 'person',
          1: 'bicycle',
          2: 'car',
          3: 'motorcycle',
          4: 'airplane',
          5: 'bus',
          6: 'train',
          7: 'truck',
          8: 'boat',
          9: 'traffic light',
          10: 'fire hydrant',
          11: 'stop sign',
          12: 'parking meter',
          13: 'bench',
          14: 'bird',
          15: 'cat',
          16: 'dog',
          17: 'horse',
          18: 'sheep',
          19: 'cow',
          20: 'elephant',
          21: 'bear',
          22: 'zebra',
          23: 'giraffe',
          24: 'backpack',
          25: 'umbrella',
          26: 'handbag',
          27: 'tie',
          28: 'suitcase',
          29: 'frisbee',
          30: 'skis',
          31: 'snowboard',
          32: 'sports ball',
          33: 'kite',
          34: 'baseball bat',
          35: 'baseball glove',
          36: 'skateboard',
          37: 'surfboard',
          38: 'tennis racket',
          39: 'bottle',
          40: 'wine glass',
          41: 'cup',
          42: 'fork',
          43: 'knife',
          44: 'spoon',
          45: 'bowl',
          46: 'banana',
          47: 'apple',
          48: 'sandwich',
          49: 'orange',
          50: 'broccoli',
          51: 'carrot',
          52: 'hot dog',
          53: 'pizza',
          54: 'donut',
          55: 'cake',
          56: 'chair',
          57: 'couch',
          58: 'potted plant',
          59: 'bed',
          60: 'dining table',
          61: 'toilet',
          62: 'tv',
          63: 'laptop',
          64: 'mouse',
          65: 'remote',
          66: 'keyboard',
          67: 'cell phone',
          68: 'microwave',
          69: 'oven',
          70: 'toaster',
          71: 'sink',
          72: 'refrigerator',
          73: 'book',
          74: 'clock',
          75: 'vase',
          76: 'scissors',
          77: 'teddy bear',
          78: 'hair drier',
          79: 'toothbrush'
    }


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
        device='cpu'
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if isinstance(prediction, (list, tuple)):  # YOLO model in validation model, output = (inference_out, loss_out) (1, 84, 6300)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - nm - 4  # number of classes
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].max(axis=1) > conf_thres

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 2.5 + 0.05 * bs  # seconds to quit after

    t = time.time()
    # output = [torch.zeros((0, 6 + nm), device=device)] * bs
    output = [np.zeros((0, 6 + nm), dtype=np.float32) for _ in range(bs)]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.T[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            # v = torch.zeros((len(lb), nc + nm + 5), device=device)
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            # x = torch.cat((x, v), 0)
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box = x[:, :4]      # 取前4列，形状为 (50, 4)
        cls = x[:, 4:84]    # 取第5列到第84列，形状为 (50, 80)
        mask = np.zeros((x.shape[0], 1))

        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        conf = np.max(cls, axis=1, keepdims=True)
        j = np.argmax(cls, axis=1)

        # x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        j_expanded = np.expand_dims(j, axis=1)  # 这将j从一维扩展到二维
        x = np.concatenate((box, conf, j_expanded.astype(np.float32), mask), axis=1)     # 将数组按列连接起来
        conf_mask = conf.ravel() > conf_thres   # 重塑conf数组，然后应用阈值过滤
        x = x[conf_mask]    # 使用布尔索引来选择满足条件的行

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            indices = np.argsort(-x[:, 4])  # 对第5列取负值后进行argsort
            x = x[indices]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        output[xi] = output[xi]
        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
    return output


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (np.minimum(a2, b2) - np.maximum(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def getdata():
    im0 = cv2.imread('E:\\arm_workspace\\yolov9\\data\\images\\horses.jpg')  # BGR

    # [array([[  5.8523,  83.2554, 477.5521, 553.6018,  0.9344,   0.,   0.]])]
    # [tensor([[  5.8523,  83.2554, 477.5521, 553.6018,  0.9344,   0.0000]])]
    im = letterbox(im0, stride=32, auto=False)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = im.astype(np.float32)
    im /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im, im0

def create_show_img(pred_list, img):
    for infos_row in pred_list:
        for box in infos_row:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            conf = box[4]
            c = int(box[5])  # integer class
            cv2.rectangle(img, p1, p2, color=(255, 56, 203), thickness=3, lineType=cv2.LINE_AA)
            label = f'{dic_labels[c]} {conf:.2f}'

            tf = max(3 - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=1, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(img, p1, p2, (255, 56, 203), -1, cv2.LINE_AA)  # filled
            cv2.putText(img,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        1,
                        color=(255, 255, 255),
                        thickness=tf,
                        lineType=cv2.LINE_AA)
    return img

def tstInfoV9():
    # 加载ONNX模型 python torch2onnx.py --weights './yolov9-m.pt' --output yolov9-m-tst.onnx
    session = ort.InferenceSession('./yolov9-m.onnx')  # (1, 8400, 1, 4) (1, 8400, 80)
    im, im0 = getdata()
    outputs = session.run(None, {'images': im})
    print(type(outputs), len(outputs))
    print(type(outputs[0]), type(outputs[1]))
    print(outputs[0].shape, outputs[1].shape)

    pred = outputs[1]
    pred_list = non_max_suppression(pred)
    print('______nested_list________>', pred_list)

    imt = create_show_img(pred_list, im0)
    cv2.imwrite('img.jpg', imt)


if __name__ == "__main__":
    tstInfoV9()

