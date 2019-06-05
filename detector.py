import torch
import cv2
import numpy as np
from torch.autograd import Variable
from darknet import Darknet
from util import process_result, load_images, resize_image, cv_image2tensor, transform_result
import pickle as pkl
import argparse
import math
import random
import os.path as osp
import os
import sys
from datetime import datetime
from time import sleep

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,100)
fontScale              = 3
fontColor              = (0,0,100)
lineType               = 2


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 object detection')
    parser.add_argument('-i', '--input', required=True, help='input image or directory or video')
    parser.add_argument('-t', '--obj-thresh', type=float, default=0.5, help='objectness threshold, DEFAULT: 0.5')
    parser.add_argument('-n', '--nms-thresh', type=float, default=0.4, help='non max suppression threshold, DEFAULT: 0.4')
    parser.add_argument('-o', '--outdir', default='detection', help='output directory, DEFAULT: detection/')
    parser.add_argument('-v', '--video', action='store_true', default=False, help='flag for detecting a video input')
    parser.add_argument('-w', '--webcam', action='store_true',  default=False, help='flag for detecting from webcam. Specify webcam ID in the input. usually 0 for a single webcam connected')
    parser.add_argument('--cuda', action='store_true', default=False, help='flag for running on GPU')
    parser.add_argument('--no-show', action='store_true', default=False, help='do not show the detected video in real time')

    args = parser.parse_args()

    return args

def create_batches(imgs, batch_size):
    num_batches = math.ceil(len(imgs) // batch_size)
    batches = [imgs[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]

    return batches

def state_trigger(labels):
    if "cell phone" in labels:
        return 1
    else:
        return 0

def draw_bbox(imgs, bbox, colors, classes, objects = ['all']):
    if objects == ['all']:
        objects = classes
    img = imgs[int(bbox[0])]
    try:
        loc = int(bbox[-1])
        label = classes[loc]
    except Exception as e:
        print(e)
        print(bbox)
        print("\n")
        label = "null"
    p1 = tuple(bbox[1:3].int())
    p2 = tuple(bbox[3:5].int())
    color = random.choice(colors)
    cv2.rectangle(img, p1, p2, color, 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    p3 = (p1[0], p1[1] - text_size[1] - 4)
    p4 = (p1[0] + text_size[0] + 4, p1[1])
    cv2.rectangle(img, p3, p4, color, -1)
    if label in objects:
        cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)

def detect_video(model, args):

    input_size = [int(model.net_info['height']), int(model.net_info['width'])]

    colors = pkl.load(open("pallete", "rb"))
    classes = load_classes("data/coco.names")
    colors = [colors[1]]
    if args.webcam:
        cap = cv2.VideoCapture(int(args.input))
        output_path = osp.join(args.outdir, 'gpu.avi')
    else:
        cap = cv2.VideoCapture(args.input)
        output_path = osp.join(args.outdir, 'det_' + osp.basename(args.input).rsplit('.')[0] + '.avi')

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.set(3, 1280)
    cap.set(4, 720)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    read_frames = 0

    start_time = datetime.now()
    print('Detecting...')
    while cap.isOpened():
        retflag, frame = cap.read()
        original_frame = frame
        read_frames += 1
        if retflag:
            frame_tensor = cv_image2tensor(frame, input_size).unsqueeze(0)
            frame_tensor = Variable(frame_tensor)

            if args.cuda:
                frame_tensor = frame_tensor.cuda()

            detections = model(frame_tensor, args.cuda).cpu()
            detections = process_result(detections, args.obj_thresh, args.nms_thresh)
            if len(detections) != 0:
                detections = transform_result(detections, [frame], input_size)
                labels = []
                for detection in detections:
                    labels.append(classes[int(detection[-1])])
                state = state_trigger(labels)
                if state == 0:
                    for detection in detections:
                        draw_bbox([frame], detection, colors, classes)
                    cv2.imshow('frame', frame)
                elif state == 1:
                    frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
                    cv2.putText(frame,'PHONE DETECTED', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                    for detection in detections:
                        if classes[int(detection[-1])] == "cell phone":
                            draw_bbox([frame], detection, colors, classes)
                    cv2.imshow('frame', frame)
                    #sleep(0.65)
                else:
                    cv2.imshow('frame', original_frame)
                   

            #if not args.no_show:
                #cv2.imshow('frame', frame)
            out.write(frame)
            if read_frames % 30 == 0:
                print('Number of frames processed:', read_frames)
            if not args.no_show and cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))
    print('Total frames:', read_frames)
    cap.release()
    out.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    print('Detected video saved to ' + output_path)

    return

def main():

    args = parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    print('Loading network...')
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights('yolov3.weights')
    if args.cuda:
        model.cuda()

    model.eval()
    print('Network loaded')

    if args.video:
        detect_video(model, args)

    else:
        detect_image(model, args)



if __name__ == '__main__':
    main()
