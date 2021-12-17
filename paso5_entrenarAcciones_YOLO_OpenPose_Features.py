"""
Train our LSTM on extracted features / OpenPose y Yolo.
Jaime Duque Domingo
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Ignoramos los mensajes de aviso de CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import sys
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from data_YOLO_OpenPose_Features import DataSet
# from extract_features import extract_features
from time import time
import os.path
import math
from matplotlib import pyplot
from matplotlib import pylab
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
#from keras.regularizers import l2

DATABASE='NTURGBD'
#DATABASE='STAIR_Train_Test'

# Para el caso de NTU-RGB
from accionesPermitidasNTURGB import NUMERO_ACCIONES
from accionesPermitidasNTURGB import objetos_validos_YOLO

# Para el caso de STAIRS
#   from accionesPermitidas import NUMERO_ACCIONES
#   from accionesPermitidas import objetos_validos_YOLO

# Para que coja la memoria de la GPU de manera progresiva
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
    
#import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


################################################################################################################
##########    DEFINIMOS EL MODELO
################################################################################################################
# Algunos parametros del modelo
# Sumamos 1 a los objetos YOLO ya que la persona aparece 2 veces
NUMERO_OBJETOS_YOLO = len(objetos_validos_YOLO) + 1
NUMERO_ARTICULACIONES_OPENPOSE = 18
NUMERO_FRAMES = 60  # 90 en el caso de STAIR
NUMERO_FEATURES_INCEPTION = 2048

output_dim = NUMERO_ACCIONES
input_dim_YOLO = NUMERO_OBJETOS_YOLO * 5 * NUMERO_FRAMES  # tenemos confianza, (x1,y1) - (x2,y2)
input_dim_OPENPOSE = NUMERO_ARTICULACIONES_OPENPOSE * 2 * NUMERO_FRAMES * 2 # tenemos (x1,y1) - (x18,y18) para 2 personas (2 articulaciones por persona)
input_dim_FEATURES_INCEPTION = NUMERO_FEATURES_INCEPTION * NUMERO_FRAMES

# Modelo 1: YOLO, OpenPose y Features a la vez
def crearModelo1():
    # CREAMOS EL MODELO
    # La entrada a la red esta unificada
    input_dim = input_dim_YOLO + input_dim_OPENPOSE + input_dim_FEATURES_INCEPTION
    inputs = keras.Input(shape=(input_dim, ), dtype=tf.float32, name='inputs')
    
    # Dividimos las entradas en 3 subgrafos
    inputsYOLO, inputsOpenPose, inputsFeatures = tf.split(inputs, num_or_size_splits=[input_dim_YOLO,input_dim_OPENPOSE, input_dim_FEATURES_INCEPTION], axis=1, name='split')
    
    dropOutInputYOLO = layers.Dropout(0.05, name='dropOutInputYOLO')(inputsYOLO)
    dropOutInputOpenPose = layers.Dropout(0.05, name='dropOutInputOpenPose')(inputsOpenPose)
    dropOutInputFeatures = layers.Dropout(0.05, name='dropOutInputFeatures')(inputsFeatures)    
    
    # Para un modelo LSTM necesitamos tensores 3D con los distintos frames en la tercera dimension
    # Lo haremos tanto para YOLO como para OpenPose como para las features
    # Tenemos 90 frames
    # YOLO
    list_Frames_YOLO_2D = tf.split(dropOutInputYOLO, num_or_size_splits=NUMERO_FRAMES, axis=1, name='splitFramesYOLO')
    # Trasladamos los vectores de 2D a 3D (t1 = layers.RepeatVector(1)(i1))
    list_Frames_YOLO_3D = [layers.RepeatVector(1)(list_Frames_YOLO_2D[i]) for i in range(NUMERO_FRAMES)]
    # Creamos una entrada 3D para LSTM de la forma: [batch, frame, features] T = layers.concatenate([t1,t2], axis=1)
    input_LSTM_YOLO = layers.concatenate([list_Frames_YOLO_3D[i] for i in range(NUMERO_FRAMES)], axis=1)

    # OpenPose
    list_Frames_OpenPose_2D = tf.split(dropOutInputOpenPose, num_or_size_splits=NUMERO_FRAMES, axis=1, name='splitFramesOpenPose')
    # Trasladamos los vectores de 2D a 3D (t1 = layers.RepeatVector(1)(i1))
    list_Frames_OpenPose_3D = [layers.RepeatVector(1)(list_Frames_OpenPose_2D[i]) for i in range(NUMERO_FRAMES)]
    # Creamos una entrada 3D para LSTM de la forma: [batch, frame, features] T = layers.concatenate([t1,t2], axis=1)
    input_LSTM_OpenPose = layers.concatenate([list_Frames_OpenPose_3D[i] for i in range(NUMERO_FRAMES)], axis=1)

    # Features
    list_Frames_Features_2D = tf.split(dropOutInputFeatures, num_or_size_splits=NUMERO_FRAMES, axis=1, name='splitFramesFeatures')
    # Trasladamos los vectores de 2D a 3D (t1 = layers.RepeatVector(1)(i1))
    list_Frames_Features_3D = [layers.RepeatVector(1)(list_Frames_Features_2D[i]) for i in range(NUMERO_FRAMES)]
    # Creamos una entrada 3D para LSTM de la forma: [batch, frame, features] T = layers.concatenate([t1,t2], axis=1)
    input_LSTM_Features = layers.concatenate([list_Frames_Features_3D[i] for i in range(NUMERO_FRAMES)], axis=1)

    # Creamos las capas LSTM
    # YOLO
    tam_YOLO = int(input_dim_YOLO / NUMERO_FRAMES)
    input_shape_YOLO = (NUMERO_FRAMES, tam_YOLO)
    LSTM_Yolo = layers.LSTM(tam_YOLO, return_sequences=False, 
                   input_shape=input_shape_YOLO,
                   dropout=0.5, name='LSTM_YOLO')(input_LSTM_YOLO)
    # OpenPose
    tam_OpenPose = int(input_dim_OPENPOSE / NUMERO_FRAMES)
    input_shape_OpenPose = (NUMERO_FRAMES, tam_OpenPose)
    LSTM_OpenPose = layers.LSTM(tam_OpenPose, return_sequences=False, 
                   input_shape=input_shape_OpenPose,
                   dropout=0.5, name='LSTM_OpenPose')(input_LSTM_OpenPose)
    # Features
    tam_Features = int(input_dim_FEATURES_INCEPTION / NUMERO_FRAMES)
    input_shape_Features = (NUMERO_FRAMES, tam_Features)
    LSTM_Features = layers.LSTM(tam_Features, return_sequences=False, 
                   input_shape=input_shape_Features,
                   dropout=0.5, name='LSTM_Features')(input_LSTM_Features)

    # Las LSTM ya vienen con dropout.
    # Ahora las concatenamos
    completeLayer = layers.concatenate([LSTM_Yolo, LSTM_OpenPose, LSTM_Features], name='concatenation')
    # Metemos una capa densa con 512 neuronas como en el modelo original de LRCN
    # dense1 = layers.Dense(512, use_bias=False, name='dense1')(completeLayer)
    # Proporcion 1/4 de las entradas siguiendo el modelo LRCN
    
    # ORIGINAL STAIRS: dense1 = layers.Dense(594, use_bias=False, name='dense1')(completeLayer)
    dense1 = layers.Dense(594, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='dense1')(completeLayer)
    
    #dense1 = layers.Dense(1661, use_bias=False, name='dense1')(completeLayer)
    normalizacion = layers.BatchNormalization(name='batchNorm1')(dense1)
    activacion = layers.Activation('relu', name='activation1')(normalizacion)
    dropOut2 = layers.Dropout(0.5, name='dropOut2')(activacion)

    # Finalmente construimos la salida
    outputs = layers.Dense(output_dim, activation='softmax', name='outputs')(dropOut2)

    # Set the metrics. Only use top k if there's a need.
    metrics = ['accuracy']
    if output_dim >= 10:
        metrics.append('top_k_categorical_accuracy')
    
    # Ahora ya podemos declara el modelo
    optimizer = Adam(lr=1e-5, decay=1e-6)
    model = keras.Model(inputs=inputs, outputs=outputs, name="modeloLSTM_YOLO_OpenPose_Features")
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    model.summary()
    # Devolvemos el modelo
    return model

# Modelo 2: Solo las features, como inicialmente
def crearModelo2():
    # CREAMOS EL MODELO
    # La entrada a la red esta unificada
    input_dim = input_dim_YOLO + input_dim_OPENPOSE + input_dim_FEATURES_INCEPTION
    inputs = keras.Input(shape=(input_dim, ), dtype=tf.float32, name='inputs')
    
    # Dividimos las entradas en 3 subgrafos
    inputsYOLO, inputsOpenPose, inputsFeatures = tf.split(inputs, num_or_size_splits=[input_dim_YOLO,input_dim_OPENPOSE, input_dim_FEATURES_INCEPTION], axis=1, name='split')
    
    dropOutInputFeatures = layers.Dropout(0.05, name='dropOutInputFeatures')(inputsFeatures)    
    
    # Para un modelo LSTM necesitamos tensores 3D con los distintos frames en la tercera dimension
    # Lo haremos para las features. Tenemos 90 frames
    # Features
    list_Frames_Features_2D = tf.split(dropOutInputFeatures, num_or_size_splits=NUMERO_FRAMES, axis=1, name='splitFramesFeatures')
    # Trasladamos los vectores de 2D a 3D (t1 = layers.RepeatVector(1)(i1))
    list_Frames_Features_3D = [layers.RepeatVector(1)(list_Frames_Features_2D[i]) for i in range(NUMERO_FRAMES)]
    # Creamos una entrada 3D para LSTM de la forma: [batch, frame, features] T = layers.concatenate([t1,t2], axis=1)
    input_LSTM_Features = layers.concatenate([list_Frames_Features_3D[i] for i in range(NUMERO_FRAMES)], axis=1)

    # Creamos las capas LSTM
    # Features
    tam_Features = int(input_dim_FEATURES_INCEPTION / NUMERO_FRAMES)
    input_shape_Features = (NUMERO_FRAMES, tam_Features)
    LSTM_Features = layers.LSTM(tam_Features, return_sequences=False, 
                   input_shape=input_shape_Features,
                   dropout=0.5, name='LSTM_Features')(input_LSTM_Features)

    # Las LSTM ya vienen con dropout.
    # Metemos una capa densa con 512 neuronas como en el modelo original de LRCN

    # ORIGINAL STAIRS: dense1 = layers.Dense(512, use_bias=False, name='dense1')(LSTM_Features)
    dense1 = layers.Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='dense1')(LSTM_Features)
    
    normalizacion = layers.BatchNormalization(name='batchNorm1')(dense1)
    activacion = layers.Activation('relu', name='activation1')(normalizacion)
    dropOut2 = layers.Dropout(0.5, name='dropOut2')(activacion)

    # Finalmente construimos la salida
    outputs = layers.Dense(output_dim, activation='softmax', name='outputs')(dropOut2)

    # Set the metrics. Only use top k if there's a need.
    metrics = ['accuracy']
    if output_dim >= 10:
        metrics.append('top_k_categorical_accuracy')
    
    # Ahora ya podemos declara el modelo
    optimizer = Adam(lr=1e-5, decay=1e-6)
    model = keras.Model(inputs=inputs, outputs=outputs, name="modeloLSTM_Features")
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    model.summary()
    # Devolvemos el modelo    
    return model    
################################################################################################################
################################################################################################################

# Modelo 3: OpenPose-YOLO, como inicialmente
def crearModelo3():
    # CREAMOS EL MODELO
    # La entrada a la red esta unificada
    input_dim = input_dim_YOLO + input_dim_OPENPOSE + input_dim_FEATURES_INCEPTION
    inputs = keras.Input(shape=(input_dim, ), dtype=tf.float32, name='inputs')
    
    # Dividimos las entradas en 3 subgrafos
    inputsYOLO, inputsOpenPose, inputsFeatures = tf.split(inputs, num_or_size_splits=[input_dim_YOLO,input_dim_OPENPOSE, input_dim_FEATURES_INCEPTION], axis=1, name='split')
    
    dropOutInputYOLO = layers.Dropout(0.05, name='dropOutInputYOLO')(inputsYOLO)
    dropOutInputOpenPose = layers.Dropout(0.05, name='dropOutInputOpenPose')(inputsOpenPose)
    
    # Para un modelo LSTM necesitamos tensores 3D con los distintos frames en la tercera dimension
    # Lo haremos tanto para YOLO como para OpenPose
    # Tenemos 90 frames
    # YOLO
    list_Frames_YOLO_2D = tf.split(dropOutInputYOLO, num_or_size_splits=NUMERO_FRAMES, axis=1, name='splitFramesYOLO')
    # Trasladamos los vectores de 2D a 3D (t1 = layers.RepeatVector(1)(i1))
    list_Frames_YOLO_3D = [layers.RepeatVector(1)(list_Frames_YOLO_2D[i]) for i in range(NUMERO_FRAMES)]
    # Creamos una entrada 3D para LSTM de la forma: [batch, frame, features] T = layers.concatenate([t1,t2], axis=1)
    input_LSTM_YOLO = layers.concatenate([list_Frames_YOLO_3D[i] for i in range(NUMERO_FRAMES)], axis=1)

    # OpenPose
    list_Frames_OpenPose_2D = tf.split(dropOutInputOpenPose, num_or_size_splits=NUMERO_FRAMES, axis=1, name='splitFramesOpenPose')
    # Trasladamos los vectores de 2D a 3D (t1 = layers.RepeatVector(1)(i1))
    list_Frames_OpenPose_3D = [layers.RepeatVector(1)(list_Frames_OpenPose_2D[i]) for i in range(NUMERO_FRAMES)]
    # Creamos una entrada 3D para LSTM de la forma: [batch, frame, features] T = layers.concatenate([t1,t2], axis=1)
    input_LSTM_OpenPose = layers.concatenate([list_Frames_OpenPose_3D[i] for i in range(NUMERO_FRAMES)], axis=1)

    # Creamos las capas LSTM
    # YOLO
    tam_YOLO = int(input_dim_YOLO / NUMERO_FRAMES)
    input_shape_YOLO = (NUMERO_FRAMES, tam_YOLO)
    LSTM_Yolo = layers.LSTM(tam_YOLO, return_sequences=False, 
                   input_shape=input_shape_YOLO,
                   dropout=0.5, name='LSTM_YOLO')(input_LSTM_YOLO)
    # OpenPose
    tam_OpenPose = int(input_dim_OPENPOSE / NUMERO_FRAMES)
    input_shape_OpenPose = (NUMERO_FRAMES, tam_OpenPose)
    LSTM_OpenPose = layers.LSTM(tam_OpenPose, return_sequences=False, 
                   input_shape=input_shape_OpenPose,
                   dropout=0.5, name='LSTM_OpenPose')(input_LSTM_OpenPose)

    # Las LSTM ya vienen con dropout.
    # Ahora las concatenamos
    completeLayer = layers.concatenate([LSTM_Yolo, LSTM_OpenPose], name='concatenation')
    # Metemos una capa densa con 512 neuronas como en el modelo original de LRCN
    # Cogemos 1/4 de las entradas como en el modelo LRCN
    dense1 = layers.Dense(82, use_bias=False, name='dense1')(completeLayer)
    # dense1 = layers.Dense(512, use_bias=False, name='dense1')(completeLayer)
    normalizacion = layers.BatchNormalization(name='batchNorm1')(dense1)
    activacion = layers.Activation('relu', name='activation1')(normalizacion)
    dropOut2 = layers.Dropout(0.5, name='dropOut2')(activacion)

    # Finalmente construimos la salida
    outputs = layers.Dense(output_dim, activation='softmax', name='outputs')(dropOut2)

    # Set the metrics. Only use top k if there's a need.
    metrics = ['accuracy']
    if output_dim >= 10:
        metrics.append('top_k_categorical_accuracy')
    
    # Ahora ya podemos declara el modelo
    optimizer = Adam(lr=1e-5, decay=1e-6)
    model = keras.Model(inputs=inputs, outputs=outputs, name="modeloLSTM_YOLO_OpenPose")
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    model.summary()
    # Devolvemos el modelo
    return model

# Modelo 4: OpenPose solamente
def crearModelo4():
    # CREAMOS EL MODELO
    # La entrada a la red esta unificada
    input_dim = input_dim_YOLO + input_dim_OPENPOSE + input_dim_FEATURES_INCEPTION
    inputs = keras.Input(shape=(input_dim, ), dtype=tf.float32, name='inputs')
    
    # Dividimos las entradas en 3 subgrafos
    inputsYOLO, inputsOpenPose, inputsFeatures = tf.split(inputs, num_or_size_splits=[input_dim_YOLO,input_dim_OPENPOSE, input_dim_FEATURES_INCEPTION], axis=1, name='split')
    
    dropOutInputOpenPose = layers.Dropout(0.05, name='dropOutInputOpenPose')(inputsOpenPose)
    
    # Para un modelo LSTM necesitamos tensores 3D con los distintos frames en la tercera dimension
    # Lo haremos para OpenPose
    # Tenemos 90 frames
    # OpenPose
    list_Frames_OpenPose_2D = tf.split(dropOutInputOpenPose, num_or_size_splits=NUMERO_FRAMES, axis=1, name='splitFramesOpenPose')
    # Trasladamos los vectores de 2D a 3D (t1 = layers.RepeatVector(1)(i1))
    list_Frames_OpenPose_3D = [layers.RepeatVector(1)(list_Frames_OpenPose_2D[i]) for i in range(NUMERO_FRAMES)]
    # Creamos una entrada 3D para LSTM de la forma: [batch, frame, features] T = layers.concatenate([t1,t2], axis=1)
    input_LSTM_OpenPose = layers.concatenate([list_Frames_OpenPose_3D[i] for i in range(NUMERO_FRAMES)], axis=1)

    # Creamos las capas LSTM
    # OpenPose
    tam_OpenPose = int(input_dim_OPENPOSE / NUMERO_FRAMES)
    input_shape_OpenPose = (NUMERO_FRAMES, tam_OpenPose)
    LSTM_OpenPose = layers.LSTM(tam_OpenPose, return_sequences=False, 
                   input_shape=input_shape_OpenPose,
                   dropout=0.5, name='LSTM_OpenPose')(input_LSTM_OpenPose)

    # Las LSTM ya vienen con dropout.
    # Metemos una capa densa con 512 neuronas como en el modelo original de LRCN
    # Como 1/4 de las neuronas es poco, ponemos las mismas que la salida
    # dense1 = layers.Dense(512, use_bias=False, name='dense1')(LSTM_OpenPose)
    dense1 = layers.Dense(NUMERO_ACCIONES, use_bias=False, name='dense1')(LSTM_OpenPose)
    normalizacion = layers.BatchNormalization(name='batchNorm1')(dense1)
    activacion = layers.Activation('relu', name='activation1')(normalizacion)
    dropOut2 = layers.Dropout(0.5, name='dropOut2')(activacion)

    # Finalmente construimos la salida
    outputs = layers.Dense(output_dim, activation='softmax', name='outputs')(dropOut2)

    # Set the metrics. Only use top k if there's a need.
    metrics = ['accuracy']
    if output_dim >= 10:
        metrics.append('top_k_categorical_accuracy')
    
    # Ahora ya podemos declara el modelo
    optimizer = Adam(lr=1e-5, decay=1e-6)
    model = keras.Model(inputs=inputs, outputs=outputs, name="modeloLSTM_OpenPose")
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    model.summary()
    # Devolvemos el modelo
    return model

# Modelo 5: YOLO solamente
def crearModelo5():
    # CREAMOS EL MODELO
    # La entrada a la red esta unificada
    input_dim = input_dim_YOLO + input_dim_OPENPOSE + input_dim_FEATURES_INCEPTION
    inputs = keras.Input(shape=(input_dim, ), dtype=tf.float32, name='inputs')
    
    # Dividimos las entradas en 3 subgrafos
    inputsYOLO, inputsOpenPose, inputsFeatures = tf.split(inputs, num_or_size_splits=[input_dim_YOLO,input_dim_OPENPOSE, input_dim_FEATURES_INCEPTION], axis=1, name='split')
    
    dropOutInputYOLO = layers.Dropout(0.05, name='dropOutInputYOLO')(inputsYOLO)
    
    # Para un modelo LSTM necesitamos tensores 3D con los distintos frames en la tercera dimension
    # Lo haremos para YOLO
    # Tenemos 90 frames
    # YOLO
    list_Frames_YOLO_2D = tf.split(dropOutInputYOLO, num_or_size_splits=NUMERO_FRAMES, axis=1, name='splitFramesYOLO')
    # Trasladamos los vectores de 2D a 3D (t1 = layers.RepeatVector(1)(i1))
    list_Frames_YOLO_3D = [layers.RepeatVector(1)(list_Frames_YOLO_2D[i]) for i in range(NUMERO_FRAMES)]
    # Creamos una entrada 3D para LSTM de la forma: [batch, frame, features] T = layers.concatenate([t1,t2], axis=1)
    input_LSTM_YOLO = layers.concatenate([list_Frames_YOLO_3D[i] for i in range(NUMERO_FRAMES)], axis=1)

    # Creamos las capas LSTM
    # YOLO
    tam_YOLO = int(input_dim_YOLO / NUMERO_FRAMES)
    input_shape_YOLO = (NUMERO_FRAMES, tam_YOLO)
    LSTM_Yolo = layers.LSTM(tam_YOLO, return_sequences=False, 
                   input_shape=input_shape_YOLO,
                   dropout=0.5, name='LSTM_YOLO')(input_LSTM_YOLO)

    # Las LSTM ya vienen con dropout.
    # Metemos una capa densa con 512 neuronas como en el modelo original de LRCN
    # Como 1/4 de las neuronas es poco, ponemos las mismas que la salida
    dense1 = layers.Dense(NUMERO_ACCIONES, use_bias=False, name='dense1')(LSTM_Yolo)
    # dense1 = layers.Dense(512, use_bias=False, name='dense1')(LSTM_Yolo)
    normalizacion = layers.BatchNormalization(name='batchNorm1')(dense1)
    activacion = layers.Activation('relu', name='activation1')(normalizacion)
    dropOut2 = layers.Dropout(0.5, name='dropOut2')(activacion)

    # Finalmente construimos la salida
    outputs = layers.Dense(output_dim, activation='softmax', name='outputs')(dropOut2)

    # Set the metrics. Only use top k if there's a need.
    metrics = ['accuracy']
    if output_dim >= 10:
        metrics.append('top_k_categorical_accuracy')
    
    # Ahora ya podemos declara el modelo
    optimizer = Adam(lr=1e-5, decay=1e-6)
    model = keras.Model(inputs=inputs, outputs=outputs, name="modeloLSTM_YOLO")
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    model.summary()
    # Devolvemos el modelo
    return model

################################################################################################################
################################################################################################################


def train(seq_length, modelName, class_limit=None, 
          batch_size=32, nb_epoch=100):

    # Tiempo de inicio
    start_time = time()

    # Helper: Save the model.
    # Cada 1000 samples grabamos batches
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('/datos','NTURGBD', 'checkpoints', modelName + '.{epoch:03d}-{val_accuracy:.3f}.hdf5'),
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_freq='epoch')

    """# Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('/datos','NTURGBD', 'logs', modelName))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('/datos','NTURGBD', 'logs', modelName + '-' + 'training-' + \
        str(timestamp) + '.log'))"""

    # Get the data and process it.
    data = DataSet(
        seq_length=seq_length,
        class_limit=class_limit)

    # Get samples per epoch.
    print("Longitud data")
    print(len(data.data))

    # Creamos los generadores
    train_generator = data.frame_generator(batch_size, 'train')    
    val_generator = data.frame_generator(batch_size, 'validation')
    test_generator = data.frame_generator(batch_size, 'test')

    # Calculamos los pasos por epoca
    steps_per_epoch = float(data.len_Training) / float(batch_size)
    steps_per_epoch = int(math.trunc(steps_per_epoch))

    val_steps_per_epoch = float(data.len_Validation) / float(batch_size)
    val_steps_per_epoch = int(math.trunc(val_steps_per_epoch))

    test_steps_per_epoch = float(data.len_Test) / float(batch_size)
    test_steps_per_epoch = int(math.trunc(test_steps_per_epoch))

    # Definimos diferentes modelos
    model = None
    if modelName == 'lstm_YOLO_OP_Features':
        model = crearModelo1()
    if modelName == 'lstm_Features':
        model = crearModelo2()
    if modelName == 'lstm_OpenPoseYOLO':
        model = crearModelo3()
    if modelName == 'lstm_OpenPose':
        model = crearModelo4()
    if modelName == 'lstm_YOLO':
        model = crearModelo5()

    # Fit!
    history = None
    # Use fit generator.
    # print("Comienzo del entrenamiento...")
    # print(steps_per_epoch)
    # print(val_steps_per_epoch)
    # print(test_steps_per_epoch)
    # print(nb_epoch)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[checkpointer],
        validation_data=val_generator,
        validation_steps=val_steps_per_epoch) 

    """rm.model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger, checkpointer],
        validation_data=val_generator,
        validation_steps=40,
        workers=4)"""
    # Guardamos el modelo
    model.save(os.path.join('/datos','NTURGBD', 'checkpoints', modelName + '-final.hdf5'))
    # Representamos graficamente
        
    # Tiempo transcurrido
    elapsed_time = time() - start_time
    print("Tiempo total: %0.10f seconds." % elapsed_time)

    """# plot learning curves
    params = {'legend.fontsize': '15',
             'axes.labelsize': '15',
             'axes.titlesize':'15',
             'xtick.labelsize':'15',
             'ytick.labelsize':'15'}
    #'figure.figsize': (15, 5),
    pylab.rcParams.update(params)"""
    pyplot.title('Learning Curves (Accuracy)')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.plot(history.history['accuracy'], color='green', linewidth=2, label='Training accuracy')
    pyplot.plot(history.history['val_accuracy'], color='red', linewidth=2, label='Validation accuracy')
    pyplot.plot(history.history['top_k_categorical_accuracy'], color='blue', linewidth=2, label='Top-5 accuracy')
    pyplot.plot(history.history['val_top_k_categorical_accuracy'], color='orange', linewidth=2, label='Validation Top-5 accuracy')
    pyplot.legend()
    # pyplot.show()
    GRAFICO_TRAINING = "/datos/" + DATABASE + "/training_curve_HAR_accuracy_" + modelName + ".svg"
    pyplot.savefig(GRAFICO_TRAINING, format='svg', dpi=1200)
    
    # plot learning curves
    pyplot.clf()
    """params = {'legend.fontsize': '15',
             'axes.labelsize': '15',
             'axes.titlesize':'15',
             'xtick.labelsize':'15',
             'ytick.labelsize':'15'}
    #'figure.figsize': (15, 5),
    pylab.rcParams.update(params)"""
    pyplot.title('Learning Curves (Loss)')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.plot(history.history['loss'], color='green', linewidth=2, label='Training loss')
    pyplot.plot(history.history['val_loss'], color='red', linewidth=2, label='Validation loss')
    pyplot.legend()
    # pyplot.show()
    GRAFICO_LOSS = "/datos/" + DATABASE + "/training_curve_HAR_loss_" + modelName + ".svg"
    pyplot.savefig(GRAFICO_LOSS, format='svg', dpi=1200)
    
    # evaluate the model ( Verbose sirve para mostrar barra de progreso)
    # loss, acc = model.evaluate(test_generator, steps=test_steps_per_epoch, verbose=1)
    scores = model.evaluate(test_generator, steps=test_steps_per_epoch, verbose=1)
    #loss, acc = model.evaluate(my_test_batch_generator, verbose=1)
    print(str(scores))
    print('Test Loss: %.3f' % scores[0])
    print('Test Accuracy: %.3f' % scores[1])
    print('Test Top_5_categorical_accuracy: %.3f' % scores[2])

def main():
    if (len(sys.argv) == 2):
        modelo = str(sys.argv[1])
    else:
        print ("Usage: python3 entrenarAcciones_YOLO_OpenPose_Features modelo")
        print ("Example: python3 entrenarAcciones_YOLO_OpenPose_Features.py OpenPose")
        print ("Example: python3 entrenarAcciones_YOLO_OpenPose_Features.py YOLO")
        print ("Example: python3 entrenarAcciones_YOLO_OpenPose_Features.py Features")
        print ("Example: python3 entrenarAcciones_YOLO_OpenPose_Features.py OpenPoseYOLO")
        print ("Example: python3 entrenarAcciones_YOLO_OpenPose_Features.py Completo")
        exit (1)

    modelName = ""
    if modelo == "OpenPose":
        modelName = 'lstm_OpenPose'
    else:
        if modelo == "YOLO":
            modelName = 'lstm_YOLO'
        else:
            if modelo == "Features":
                modelName = 'lstm_Features'
            else:
                if modelo == "OpenPoseYOLO":
                    modelName = 'lstm_OpenPoseYOLO'
                else:                
                    if modelo == "Completo":
                        modelName = 'lstm_YOLO_OP_Features'
                    else:
                        print ("Usage: python3 entrenarAcciones_YOLO_OpenPose_Features modelo")
                        print ("Example: python3 entrenarAcciones_YOLO_OpenPose_Features.py OpenPose")
                        print ("Example: python3 entrenarAcciones_YOLO_OpenPose_Features.py YOLO")
                        print ("Example: python3 entrenarAcciones_YOLO_OpenPose_Features.py Features")
                        print ("Example: python3 entrenarAcciones_YOLO_OpenPose_Features.py OpenPoseYOLO")                    
                        print ("Example: python3 entrenarAcciones_YOLO_OpenPose_Features.py Completo")
                        exit (1)
    
    seq_length = NUMERO_FRAMES
    class_limit = NUMERO_ACCIONES

    checkpoints_dir = os.path.join('/datos','NTURGBD', 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    # modelName = 'lstm_Features'
    # modelName = 'lstm_YOLO_OP_Features'
    # batch_size = 32
    batch_size = 64
    nb_epoch = 200
    #nb_epoch = 50

    train(seq_length, modelName, class_limit=class_limit,
          batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
