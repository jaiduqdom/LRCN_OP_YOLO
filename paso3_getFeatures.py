"""
Paso 3. Extraccion de las features de STAIRS o NTURGBD.
Jaime Duque
Se ejecuta como en el paso 2:  python3 paso3_getFeatures.py accion1 accion2
    Ejemplo:    python3 paso3_getFeatures.py smoking taking_photo
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Ignoramos los mensajes de aviso de CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import time
import os.path
import sys
import numpy as np
import os.path
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import cv2
import ssl
import tensorflow.compat.v1 as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ssl._create_default_https_context = ssl._create_unverified_context

class Extractor():
    def __init__(self, image_shape=(299, 299, 3), weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        input_tensor = Input(image_shape)
        # Get model with pretrained weights.
        base_model = InceptionV3(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=True
        )

        # We'll extract features at the final pool layer.
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output
        )

    def extract_image(self, img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)
        return features[0]


DIRECTORIO_VIDEOS_STAIRS="/datos/NTURGBD/VIDEOS_50_120"
DIRECTORIO_SALIDA="/datos/NTURGBD/DATOS-FEATURES"
MAX_FRAMES = 50

def main():
    empezar=sys.argv[1]
    finalizar=sys.argv[2]
    activar = False    
    
    seq_length = 50
    image_height = 360
    image_width = 640

    if not os.path.exists(DIRECTORIO_SALIDA):
        os.mkdir(DIRECTORIO_SALIDA)

    image_shape = (image_height, image_width, 3)

    # get the model.
    model = Extractor(image_shape=image_shape)

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
                    archivo_salida = DIRECTORIO_SALIDA + "/" + accion_STAIRS + "/" + archivo_sin_extension + ".npy"
                    if not os.path.exists(archivo_salida):            
        
                        # Leemos los frames del video
                        nombre = DIRECTORIO_VIDEOS_STAIRS + '/' + accion_STAIRS + '/' + archivo_video
                        print(nombre)
                        cap = cv2.VideoCapture(nombre)

                        # Check if camera opened successfully
                        if (cap.isOpened()== False): 
                            print("Error opening video " + nombre + ".")
                        else:
                            nf = 0
                            sequence = []
                        
                            # Read until video is completed
                            while(nf < seq_length and cap.isOpened()):
                                ret, frame = cap.read()
                                if ret == True:
                                    print(str(nf))
                                    fr2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    features = model.extract_image(fr2)
                                    sequence.append(features)
                                    nf += 1

                            # Save the sequence.
                            np.save(archivo_salida, sequence)
                        
                        # When everything done, release the video capture object
                        cap.release()

if __name__ == '__main__':
    main()
