import os
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox # LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class Detector:
    def __init__(self, weights='yolov5s.pt', device="cpu", img_size=640):
        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device)
        stride = int(self.model.stride.max())  # model stride
        print(stride)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.img_size = check_img_size(img_size, s=stride)  # check img_size
        print("half : {}, imgsz : {}".format(self.half, self.img_size))
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.names_map = {n: i for i, n in enumerate(self.names)}
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        print("names : {}, colors : {}".format(len(self.names), len(self.colors)))
        print(self.names_map)

        self.augment = False
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.agnostic_nms = False
        self.classes = None
        self.stride = 32

    def detect_car(self, img, debug=False):
        """
        input:
            img     opencv
        output:
            box
            score
            draw_img
        """
        im0 = img.copy()

        img, ratio, (dw, dh) = letterbox(img, self.img_size, stride=self.stride)
        print(img.shape, ratio, (dw, dh))

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        print(img.shape)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        print(img.shape)
        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]
        print(len(pred), pred[0].shape)

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t2 = time_synchronized()
        print("elapse time : ", t2 - t1)
        print(len(pred), pred[0].shape, pred)

        det = pred[0]
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det) < 1:
            return None, 0.0, im0
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        boxes = det[:, :4].cpu().numpy()
        labels = det[:, 5].cpu().numpy()
        scores = det[:, 4].cpu().numpy()
        print(boxes.shape, boxes)
        print(labels.shape, labels)
        print(scores.shape, scores)

        if len(boxes) < 1:
            return None, 0, im0

        indies = np.where(labels == self.names_map['car'])
        if len(indies[0]) < 1:
            print("no car")
            indies = np.where(labels == self.names_map['bus'])
        if len(indies[0]) < 1:
            print("no bus")
            indies = np.where(labels == self.names_map['truck'])
        if len(indies[0]) < 1:
            print("no truck")
            return None, 0, im0
        boxes = boxes[indies]
        scores = scores[indies]
        labels = labels[indies]
        print("boxes    :{}".format(boxes))
        print("labels   :{}".format(labels))
        print("scores   :{}".format(scores))

        max_box = np.array(boxes[0]).astype(np.int32).tolist()
        max_area = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
        max_score = scores[0]
        if len(boxes) > 1:
            for i in range(1, len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2 + 0.5), int(y2 + 0.5)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area and scores[i] + 0.3 > max_score:
                    max_area = area
                    max_box = [x1, y1, x2, y2]
                    max_score = scores[i]
        print(max_box, max_score, max_area)
        if max_box and debug:
            x1, y1, x2, y2 = max_box
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), thickness=4)
        return max_box, float(max_score), im0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--input', type=str, default='data/images/bus.jpg', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    return opt


def main(args):
    print(args)
    det = Detector(weights=args.weights, device=args.device)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    print(args.input, img.shape)
    t0 = time.time()
    box, score, img_draw = det.detect_car(img, debug=True)
    print(box, score, time.time() - t0)
    cv2.imwrite(os.path.join(args.project, os.path.basename(args.input)), img_draw)


if __name__ == '__main__':
    main(get_args())
    '''
    opt = get_args()
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
    '''