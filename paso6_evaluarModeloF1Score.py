"""
Este programa procesa cada uno de los videos del juego de test y escribe en un fichero la salida esperada y el 
vector de salida de acciones de la red neuronal:   VIDEO; ACCION_ESPERADA (0-77); ACCION_ESPERADA_DESCRIPCION; SALIDA_0; SALIDA_1; ... SALIDA_77

Jaime Duque Domingo
"""
import csv
import sys
import numpy as np
import os.path
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from itertools import cycle

# Para el caso de NTU-RGB
from accionesPermitidasNTURGB import NUMERO_ACCIONES
# Para el caso de STAIRS
#   from accionesPermitidas import NUMERO_ACCIONES

DATABASE='NTURGBD'
#DATABASE='STAIR_Train_Test'

# Modelos
# OP+YOLO+Features
MODELO1 = "/home/roasis/contextAction/TF/TRAINING_200_EPOCAS/lstm_YOLO_OP_Features-final.hdf5"
# Features
MODELO2 = "/home/roasis/contextAction/TF/TRAINING_200_EPOCAS/lstm_Features-final.hdf5"
# OP+YOLO
MODELO3 = "/home/roasis/contextAction/TF/TRAINING_200_EPOCAS/YOP-lstm_YOLO_OP-final.hdf5"
# OP
MODELO4 = "/home/roasis/contextAction/TF/TRAINING_200_EPOCAS/YOP-lstm_OP-final.hdf5"
# YOLO
MODELO5 = "/home/roasis/contextAction/TF/TRAINING_200_EPOCAS/YOP-lstm_YOLO-final.hdf5"

MAX_FRAMES = 60  # En STAIRS son 90

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

# DATASETS
test = []
train = []
classes = []

with open(os.path.join('/datos',DATABASE,'data_file_filtrado_shuffle.csv'), 'r') as fin:
    reader = csv.reader(fin)
    data = list(reader)

    for item in data:
        if item[0] == 'test':
            test.append(item)
        if item[0] == 'train':
            train.append(item)

    # Obtenemos la relacion de clases de salida
    for item in data:
        if item[1] not in classes:
            classes.append(item[1])

    # Sort them.
    classes = sorted(classes)

def get_extracted_YOLO_OP_Feature_sequence(sample):
    """Get the saved extracted features."""
    filename = sample[2]
    path = os.path.join('/datos',DATABASE, 'sequences_features_OpenPose_YOLO', filename + '-' + str(MAX_FRAMES) + '-yolo_op_features.npz')
    if os.path.isfile(path):
        npzfile = np.load(path)
        matriz_YOLO = npzfile['YOLO']
        matriz_OpenPose = npzfile['OpenPose']
        matriz_Features = npzfile['FeaturesInception']
        return matriz_YOLO, matriz_OpenPose, matriz_Features
    else:
        return None, None, None

def get_class_one_hot(class_str):
    """Given a class as a string, return its number in the classes
    list. This lets us encode and one-hot it for training."""
    # Encode it first.
    label_encoded = classes.index(class_str)

    # Now one-hot it.
    label_hot = to_categorical(label_encoded, len(classes))

    assert len(label_hot) == len(classes)

    for p in range(len(label_hot)):
        if label_hot[p] == 1:
            return p
    return -1
    # return label_hot

# Realizamos las predicciones del test para realizar la evaluación del modelo
def evaluarModelo(grupo, m, modelo, param):
    res_X = []
    res_C = []
    test_Y = []
    for i in range(len(grupo)):
        # Reset to be safe.
        matriz_YOLO = None
        matriz_OpenPose = None
        matriz_Features = None
        sample = grupo[i]
    
        # Get the sequence from disk.
        matriz_YOLO, matriz_OpenPose, matriz_Features = get_extracted_YOLO_OP_Feature_sequence(sample)
    
        if matriz_YOLO is None or matriz_OpenPose is None or matriz_Features is None:
            print("Archivo: " + str(sample))
            raise ValueError(str(sample) + "Can't find sequence. Did you generate them?")
    
        # Para poderlo utilizar con Tensorflow facilmente, hacemos un flatten a las 3 matrices y las
        # fusionamos. De esta manera recibiremos un unico tensor y luego lo desacoplaremos dentro
        # de nuestro modelo
        m1 = matriz_YOLO.flatten()
        m2 = matriz_OpenPose.flatten()
        m3 = matriz_Features.flatten()
        
        # Para los dos primeros modelos, concatenamos las 3 matrices
        # Los otros modelos los hemos entrenado con la concatenación de YOLO - OpenPose        
        matriz = None
        if m == 1 or m == 2:
            matriz = np.concatenate((m1, m2, m3))
        else:
            matriz = np.concatenate((m1, m2))            
    
        #test_X.append(np.array([matriz]))
        v = modelo.predict(np.array([matriz]))
        res_X.append(v)
        # Como en los modelos Functional no existe predict_classes, lo emulamos
        #res_C.append(modelo.predict_classes(np.array([matriz])))
        o = np.argmax(v, axis=1)
        res_C.append(o)
        test_Y.append(get_class_one_hot(sample[1]))
    
    # REALIZAMOS LA EVALUACION DEL MODELO
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
    # predict probabilities for test set
    yhat_probs = np.array(res_X)
    
    # predict crisp classes for test set
    yhat_classes = np.array(res_C)

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]
	
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(test_Y, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(test_Y, yhat_classes, average='micro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(test_Y, yhat_classes, average='micro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(test_Y, yhat_classes, average='micro')
    print('F1 score: %f' % f1)
	
    # kappa
    kappa = cohen_kappa_score(test_Y, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    test_Y = np.array(test_Y)
    auc = roc_auc_score(test_Y, yhat_probs, multi_class="ovr",average='macro')
    print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(test_Y, yhat_classes)
    print(matrix)

    # DIBUJAMOS LA CURVA PRECISION-RECALL CALCULADA GENERICAMENTE
    # PLOT!!
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = NUMERO_ACCIONES
    Y_test = np.zeros((len(grupo), n_classes))
    for i in range(len(grupo)):
        Y_test[i][test_Y[i]] = 1
    y_score = yhat_probs
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
    
    # pyplot.show()
    plt.savefig("/datos/precision_recall_" + param + ".svg", format='svg', dpi=1200)

    # DIBUJAMOS LA CURVA PRECISION-RECALL PARA CADA CLASE
    plt.clf()    
    #plt.show()
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'teal',                    
            'dimgray',       'gray',             'darkgray',    'silver',       'lightgrey',  'whitesmoke',     'snow',                 'lightcoral', 
            'brown',         'maroon',           'red',         'salmon',       'darksalmon', 'orangered',      'sienna',               'chocolate', 
            'sandybrown',    'peru',             'bisque',      'burlywood',    'tan',        'blanchedalmond', 'moccasin',             'wheat', 
            'floralwhite',   'goldenrod',        'gold',        'khaki',        'darkkhaki',  'beige',          'lightgoldenrodyellow', 'yellow', 
            'yellowgreen',   'greenyellow',      'lawngreen',   'darkseagreen', 'lightgreen', 'limegreen',      'green',                'seagreen', 
            'springgreen',   'mediumspringgreen','aquamarine',  'lightseagreen','azure',      'paleturquoise',  'darkslategrey',        'darkcyan', 
            'cyan',          'cadetblue',        'lightblue',   'skyblue',      'steelblue',  'dodgerblue',     'lightslategrey',       'slategrey', 
            'cornflowerblue','ghostwhite',       'midnightblue','darkblue',     'blue',       'darkslateblue',  'mediumpurple',         'blueviolet', 
            'darkorchid',    'mediumorchid',     'plum',        'purple',       'fuchsia',    'orchid',         'deeppink',             'lavenderblush', 
            'crimson',       'lightpink'])
    
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        #plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(1.01, y[45] + 0.02), annotation_clip=False)
    
    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))
    
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall curve')
    #plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=5), ncol=2)
    plt.legend(lines, labels, loc="lower center", prop=dict(size=5.2), bbox_to_anchor=(0.5, -1.2), ncol=3)
    plt.savefig("/datos/precision_recall_by_class_" + param + ".svg", format='svg', dpi=1200)
    
def main():
    if (len(sys.argv) == 2):
        param = str(sys.argv[1])
    else:
        print ("Usage: python3 evaluarModeloF1Score modelo")
        print ("Example: python3 evaluarModeloF1Score.py OpenPose")
        print ("Example: python3 evaluarModeloF1Score.py YOLO")
        print ("Example: python3 evaluarModeloF1Score.py OpenPoseYOLO")
        print ("Example: python3 evaluarModeloF1Score.py OpenPoseFeaturesYOLO")
        print ("Example: python3 evaluarModeloF1Score.py Features")
        exit (1)

    m = 0
    if param == "OpenPose":
        m = 4
    else:
        if param == "YOLO":
            m = 5
        else:
            if param == "OpenPoseYOLO":
                m = 3
            else:
                if param == "Features":
                    m = 2
                else:
                    if param == "OpenPoseFeaturesYOLO":
                        m = 1
                    else:
                        print ("Usage: python3 evaluarModeloF1Score modelo")
                        print ("Example: python3 evaluarModeloF1Score.py OpenPose")
                        print ("Example: python3 evaluarModeloF1Score.py YOLO")
                        print ("Example: python3 evaluarModeloF1Score.py OpenPoseYOLO")
                        print ("Example: python3 evaluarModeloF1Score.py OpenPoseFeaturesYOLO")
                        print ("Example: python3 evaluarModeloF1Score.py Features")
                        exit (1)                

    #####################################################
    ## Cargamos el modelo
    #####################################################
    pmodelo = ""
    if m == 1:
        pmodelo = MODELO1
    if m == 2:
        pmodelo = MODELO2
    if m == 3:
        pmodelo = MODELO3
    if m == 4:
        pmodelo = MODELO4
    if m == 5:
        pmodelo = MODELO5
    
    modelo = tf.keras.models.load_model(pmodelo)
    
    # Show the model architecture
    modelo.summary()
    
    #####################################################
    ## Leer datos de test
    #####################################################
    
    evaluarModelo(test, m, modelo, param)
    #evaluarModelo(train, salida_train, m, modelo)
    
    print("Proceso terminado correctamente.")

if __name__ == '__main__':
    main()
