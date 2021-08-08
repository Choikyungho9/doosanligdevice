import argparse
import time
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box#, plot_one_box_center
from utils.torch_utils import select_device, load_classifier, time_synchronized
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tkinter as tk
#import Jetson.GPIO as GPIO
from datetime import datetime
#---------monitor.py---------------
#import pygame
#from pygame import Surface
import sys
import time
import os
#from pygame.locals import QUIT,Rect
import serial
import select
import sys
#------------------------------------
###odometer###
#import smbus
#---------------File Save Part--------------------#

address = 0x48 #// ADC Converter's 0x40~0x48
A0 = 0x40 #// Set the address of the A0 pin as input
A1 = 0x41 #// Set the address of the A1 pin as input
A2 = 0x42 #// Set address of A2 pin as input
A3 = 0x43 #// Set the address of the A3 pin as input
#bus = smbus.SMBus(1)

os.environ['SDL_VIDEO_WINDOW_POS']=str('0,-10')
#-----------------------------------

# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup([21,22,23,24], GPIO.OUT, initial=GPIO.HIGH)
#GPIO.setup(22, GPIO.OUT)
#GPIO.setup(23, GPIO.OUT)
#GPIO.setup(24, GPIO.OUT)

#f1 = open('test/0805_co1.csv','w')
#f2 = open('test/0805_ts1.csv','w')

def distance(x1, y1, x2, y2):
    """
    Calculate distance between two points
    """
    dist = math.sqrt(math.fabs(x2 - x1) ** 2 + math.fabs(y2 - y1) ** 2)
    return dist


###--- angle correction ---###
@torch.no_grad()
def detect(weights='yolov5s.pt',  # model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=320,  # inference size (pixels)
           conf_thres=0.35,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=2000,  # maximum detections per image
           device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           ):
    global x1,y1,x2,y2
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    ###--- distance average array definition ---###
    readings_dist_m=np.array([])
    max_dist_m_samples=6

    ###--- angle average array definition ---###
    readings_pallet_width=np.array([])  # width average
    readings_pallet_height=np.array([]) # height average

    max_pallet_width_samples=1         # width average 
    max_pallet_height_samples=1        # height average 
    approch_angle=0

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # bus.read_byte_data(address, A2) #// measure the signal of pin A1
        # value = bus.read_byte_data(address, A2) #// Measured signal
        # Process detections
        # v = (0.2512+(1.5860*value))
        # #v = (0.2512+(1.6560*value))
        # v2=float("{:.1f}".format(v))

        for i, det in enumerate(pred):  # detections per image
            #-----odometer_run-----
            #bus.read_byte_data(address, A2)
            #value = bus.read_byte_data(address, A2)
            #----------------------
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    ###----box coordinate----
                    x_top = float(xyxy[0])
                    y_top = float(xyxy[1])
                    width = float(xyxy[2] - xyxy[0])
                    height = float(xyxy[3] - xyxy[1])
                    pallet_x = float(xyxy[2])
                    pallet_y = float(xyxy[3])
                    # print(x_top," ",y_top, " ", width, " ", height)

                    ###--- realtime average dist_m ---###
                    #time.sleep(0.1)
                    #readings_dist_m=np.append(readings_dist_m,dist_i)
                    #avg=[np.mean(readings_dist_m)]
                    #avg_dist_m=int(avg[0])

                    ####====angle average code====####
                    ###--- angle estimation ---###
                    readings_pallet_width=np.append(readings_pallet_width,width)     # width average
                    avg_width=[np.mean(readings_pallet_width)]                       # width average
                    avg_width2=int(avg_width[0])                                 # width average
                        
                    readings_pallet_height=np.append(readings_pallet_height,height)  # height average
                    avg_height=[np.mean(readings_pallet_height)]                     # height average
                    avg_height2=int(avg_height[0])*2                                 # height average
                    #angle = np.arcsin(vertical / hypotenuse) * 180.0 / math.pi      #old code
                    #angle_m=int(angle.astype(np.float)*10)/10   




                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        (h, w) = im0.shape[:2]
                        cv2.circle(im0, (w // 2, h - 5), 7, (0, 255, 0), -3)
                        ref_w = w // 2
                        ref_h = h
                        
                        ###----if detect pallet:----   
                        if c == 1:
                            #GPIO.output(21, GPIO.LOW)  #signal tower
                            distancei2 = (2 * 31.4 * 180) * 5 / (width + height)
                            distancei3 = 43.2675+0.7286*distancei2  
                            distancei4 = 317.5033-6.8798*distancei3 + 0.0559*math.pow(distancei3, 2) - 0.0001*math.pow(distancei3, 3)
                            if 100<distancei4<180:
                                distamcei4= 83.9971 - 6.8798*distancei4 +0.0549*math.pow(distancei4,2) - 0.0001*math.pow(distancei4, 3)
                            elif 50<distancei4<100:
                                distamcei4= 75.9971 - 6.8798*distancei4 +0.0549*math.pow(distancei4,2) - 0.0001*math.pow(distancei4, 3)
                            dist_i=int(distancei4*10)/10
                            #dist_m1 = (-1.7134 * math.pow(10, -22)) + (2.0906 * distancei) + (-0.0146 * math.pow(distancei, 2)) + (
#4.5538 * math.pow(10, -5) * math.pow(distancei, 3))
                            #if dist_m1 > 95:
                            #    dist_m2 = -22.8275 + 1.0849 * dist_m1
                                # print('1')
                            #    if 90 < dist_m2 < 186:
                            #        dist_m2 = -40.2318 + 1.1376 * dist_m2 + 40
                                    # print('2')
                                    # print(dist_m2,'33')
                            #elif dist_m1 <= 120:
                            #    dist_m2 = -1439.9452 + 27.3437 * dist_m1 - 0.1206 * math.pow(dist_m1, 2)
                                # print('3')
                            # print(dist_m2,'4')
                            #dist_m3 = 1.2715 + 0.9799 * dist_m2
                            #dist_m = round(dist_m3,2)
                            #print(dist_m)
                            plot_one_box(xyxy, im0, label=f'{dist_i:.2f}cm', color=colors(c, True), line_thickness=2)
                            # x,y,w,h = cv2.boundingRect(xyxy) 
                            # Draw circle in the center of the bounding box
                            x00 = int(x_top)  # + int(width / 2)
                            y00 = int(y_top)  # + int(height/2)
                            x01 = int(x_top + width)
                            y10 = int(y_top + height)
                            x01_h = int(x_top + width * 0.295)
                            y10_h = int(y_top + height * 0.53)
                            x01_h2 = int(x_top + width * 0.705)
                            x_center = int((x01 + x00) / 2)
                            y_center = int((y00 + y10) / 2)
                            holex_c = int((x00 + x01))
                            holey_c = int((y00 + y10))
                            cv2.circle(im0, (x00, y00), 4, (0, 255, 0), 0)  # 1-point
                            cv2.circle(im0, (x00, y10), 4, (0, 255, 0), 0)  # 2-point
                            cv2.circle(im0, (x01, y00), 4, (0, 255, 0), 0)  # 3-point
                            cv2.circle(im0, (x01, y10), 4, (0, 255, 0), 0)  # 4-point
                            # cv2.circle(im0, (x_center, y_center), 4, (0, 255, 0), -1) #center-point
                            cv2.circle(im0, (x01_h, y10_h), 4, (0, 255, 0), -1)  # hole-center-point
                            cv2.circle(im0, (x01_h2, y10_h), 4, (0, 255, 0), -1)  # hole-center-point
                            # A = (ref_w,ref_h)
                            # B = (x_center, y_center)
                            #---------------------------
                            
                            # trig stuff to get the line
                            #hypotenuse = distance(ref_w, ref_h, x_center, y_center)
                            #horizontal = distance(ref_w, ref_h, x_center, ref_h)
                            #vertical = distance(x_center, y_center, x_center, ref_h)
                            
#                            ###--- realtime average dist_m ---###
 #                           #time.sleep(0.1)
  #                          readings_dist_m=np.append(readings_dist_m,dist_m)
  #                          avg=[np.mean(readings_dist_m)]
  #                          avg_dist_m=int(avg[0])
#
#                            ####====angle average code====####
#                            ###--- angle estimation ---###
#                            readings_pallet_width=np.append(readings_pallet_width,width)     # width average
#                            avg_width=[np.mean(readings_pallet_width)]                       # width average
#                            avg_width2=int(avg_width[0])                                 # width average
#                            
#                            readings_pallet_height=np.append(readings_pallet_height,height)  # height average
#                            avg_height=[np.mean(readings_pallet_height)]                     # height average
#                            avg_height2=int(avg_height[0])*2                                 # height average
                            #angle = np.arcsin(vertical / hypotenuse) * 180.0 / math.pi      #old code
                            #angle_m=int(angle.astype(np.float)*10)/10                       #old code
                            ###---------------------- realtime approch_angle ------------------------------###
                            #print(cal_angle)
                            #--- pallet range ---#
       
                            # draw 3 angle lines
                            cv2.line(im0, (ref_w, ref_h), (x_center, y_center), (0, 0, 255), 2)
                            #cv2.line(im0, (ref_w, ref_h), (x_center, ref_h), (0, 0, 255), 2)
                            #cv2.line(im0, (x_center, y_center), (x_center, ref_h), (0, 0, 255), 2)

                            # put angle text (allow for calculations upto 180 degrees)
                            #angle_text = ""
                            #if y_center < ref_h and x_center > ref_w:
                            #    angle_text = str(int(angle))
                            #elif y_center < ref_h and x_center < ref_w:
                            #    angle_text = str(int(180 - angle))
                            #elif y_center > ref_h and x_center < ref_w:
                            #    angle_text = str(int(180 + angle))
                            #elif y_center > ref_h and x_center > ref_w:
                            #    angle_text = str(int(360 - angle))
                            #else:
                            #    pass
                            # CHANGE FONT HERE
                            #cv2.putText(im0, angle_text, (ref_w - 30, ref_h), cv2.FONT_HERSHEY_COMPLEX, 3,
                                  #      (0, 128, 229), 2)

                            print('distancei',dist_i,'dist_m',{dist_i}, 'time', datetime.utcnow().strftime('%S.%f')[:-3], 'width',width,'height',height)#,v2)
                           
                            #print(avg_dist_m,approch_angle,'------------') #,datetime.utcnow().strftime('%H:%M:%S.%f')[:-3],'avg!')
                            #f1.write(f"'time',{datetime.utcnow().strftime('%S.%f')[:-3]},'velocity',{v2},'dist',{distancei},'dist_m',{dist_m}, 'width',{width},'height',{height}\n")
                        else:
                            pass
                            #GPIO.output(21, GPIO.HIGH)
                            #GPIO.output(22, GPIO.HIGH)
                            #GPIO.output(23, GPIO.HIGH)
                            #GPIO.output(24, GPIO.HIGH)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            #f2.write(f"'time:',{datetime.utcnow().strftime('%S.%f')[:-3]},'velocity:',{v2}\n")
            # Print time (inference + NMS)
            #print(f'{s}Done. ({t2 - t1:.3f}s)')

            FPS=1000/(float(f'{t2 - t1:.3f}'))*0.001
            #print(f'Current FPS: 1000/(float(f'{t2 - t1:.3f}')')
            #print(f'({t2 - t1:.3f})')
            #print(f'Current FPS: ({FPS})' ,datetime.utcnow().strftime('%M:%S.%f')[:-3], v2)
            # Stream results

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
      
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    try:
        global px,py, x1, y1, x2, y2
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='ligdevice.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=288, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.35, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=3000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        opt = parser.parse_args()
        #print(opt)
        check_requirements(exclude=('tensorboard', 'thop'))
        detect(**vars(opt))
    except ValueError:
        print('NaN')
        pass
                            
