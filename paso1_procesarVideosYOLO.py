# LANZAR DESDE LA CARPETA YOLO!!
# Paso 1. Extraccion de objetos YOLO5 de videos de STAIRS
# Autor: Jaime Duque Domingo

# Este programa lee todos los videos del dataset STAIRS y extrae fichero a fichero los objetos 
# detectados en cada frame de cada video. El proceso crea el directorio de salida si no existe
# y detecta los videos que se han escrito. Si uno ya esta escrito, no se vuelve a escribir
# Sera necesario borrar la carpeta de salida para resetear el proceso

# Para cada directorio en la carpeta de entrada (accion) se crea un directorio en la carpeta de salida
# Para cada video xxxxx.mp4 se crea un fichero de datos xxxxx.dat que incluye los objetos detectados 
# con el siguiente formato:

# accion_STAIRS; clase_YOLO; nombre_fichero; numero_frame; ancho_frame; alto_frame; x1; y1; x2; y2; confianza
#        donde (x1,y1)-(x2,y2) representa la posicion del objeto YOLO

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#sys.path.remove('/home/disa/catkin_ws/devel/lib/python2.7/dist-packages')
#sys.path.append('/usr/local/lib/python3.7/site-packages')
#import cv2 as cv
#print("Version OpenCV: " + cv.__version__)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
#response = urllib.request.urlopen('https://www.python.org')
#print(response.read().decode('utf-8'))

import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

"""from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
#from utils.general import (
#    check_img_size, non_max_suppression, apply_classifier, scale_coords,
#    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)    
from utils.torch_utils import select_device, load_classifier, time_synchronized
"""
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

DIRECTORIO_VIDEOS_STAIRS="/datos/NTURGBD/VIDEOS_50_120"
DIRECTORIO_SALIDA="/datos/NTURGBD/DATOS-YOLO5"
weights = "/datos/NTURGBD/yolov5x.pt"

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# Inicializamos el modelo YOLO5
set_logging()
device = select_device('')
#if os.path.exists(out):
#    shutil.rmtree(out)  # delete output folder
#os.makedirs(out)  # make new output folder
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
# weights = "../YOLO_COLO_TEST/yolov5x.pt"
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = 640    
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16
    
# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pth', map_location=device)['model'])  # load weights
    modelc.to(device).eval()    

def detect(video, accion_STAIRS):
    resultado = ""
    nombre_fichero = os.path.basename(video)
    webcam = False
    
    # Set Dataloader
    source = video
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    #numero_frame = -1
    
    for path, img, im0s, vid_cap in dataset:
        #numero_frame = numero_frame + 1
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=True)[0]

        # Apply NMS
        opt_classes = None
        opt_conf_thres = 0.4 # object confidence threshold
        opt_iou_thres = 0.5 # IOU threshold for NMS
        pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres, classes=opt_classes, agnostic=True)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            #save_path = str(Path(out) / Path(p).name)
            #txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                """for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string"""

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c1_x = int(xyxy[0])
                    c1_y = int(xyxy[1])
                    c2_x = int(xyxy[2])
                    c2_y = int(xyxy[3])
                    ancho_img = im0.shape[1]
                    alto_img = im0.shape[0]
                    clase = names[int(cls)]
                    confianza = '%.2f' % (conf)
                    confianza = str(confianza)
                    numero_frame = dataset.frame
                    resultado = resultado + (accion_STAIRS + "; " + clase + "; " + nombre_fichero + "; " + str(numero_frame) + "; " +
                        str(ancho_img) + "; " + str(alto_img) + "; " + str(c1_x) + "; " + str(c1_y) + "; " +
                        str(c2_x) + "; " + str(c2_y) + "; "  + str(confianza) + "\n")
    return resultado

# Si el directorio de salida no existe, lo creamos
if not os.path.exists(DIRECTORIO_SALIDA):
    os.mkdir(DIRECTORIO_SALIDA)

empezar=sys.argv[1]
finalizar=sys.argv[2]
activar = False    

for base, dirs, files in sorted(os.walk(DIRECTORIO_VIDEOS_STAIRS)):
    accion_STAIRS = base.split(os.path.sep)[-1]
    if accion_STAIRS == empezar:
        activar = True
    if accion_STAIRS == finalizar:
        activar = False
    if activar == True:        
        if accion_STAIRS != "VIDEOS":    
            print("Procesando accion: " + accion_STAIRS + " ...")
            # Crear directorio si no existe
            if not os.path.exists(DIRECTORIO_SALIDA + '/' + accion_STAIRS):
                os.mkdir(DIRECTORIO_SALIDA + '/' + accion_STAIRS)        
    
            for archivo_video in files:
                print("--Procesando: " + archivo_video)            
                archivo_sin_extension = os.path.splitext(archivo_video)[0]
                
                # Si el archivo ya existe lo omitimos
                archivo_salida = DIRECTORIO_SALIDA + "/" + accion_STAIRS + "/" + archivo_sin_extension + ".dat"
                if not os.path.exists(archivo_salida):            
    
                    resultado = detect(DIRECTORIO_VIDEOS_STAIRS + '/' + accion_STAIRS + '/' + archivo_video, accion_STAIRS)
    
                    # Escribimos el fichero
                    with open(archivo_salida, 'a+') as f:
                        f.write(resultado)
                        f.close()
    
print("Proceso terminado correctamente.")
