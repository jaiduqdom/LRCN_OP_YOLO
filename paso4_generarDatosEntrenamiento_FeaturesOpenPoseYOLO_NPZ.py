# Generar datos de entrenamiento para un sistema basado en features, los objetos detectados por YOLO y las poses de OpenPose
# Estos datos que generamos son los utilizados para entrenar nuestra red neuronal
# Autor: Jaime Duque Domingo
# Fecha: 21-Noviembre-2020

# Para cada video de STAIRS se ha generado previamente un fichero YOLO de datos xxxxx.dat que 
# incluye los objetos detectados con el siguiente formato:

# accion_STAIRS; clase_YOLO; nombre_fichero; numero_frame; ancho_frame; alto_frame; x1; y1; x2; y2; confianza
#        donde (x1,y1)-(x2,y2) representa la posicion del objeto YOLO

# Para cada video de STAIRS se ha generado previamente un fichero OpenPose de datos xxxxx.dat que 
# incluye los objetos detectados con el siguiente formato:

# accion_STAIRS; nombre_fichero; numero_frame; ancho_frame; alto_frame; id_persona; x0; y0; x1; y1; x2; y2; ... ; x25; y25; confianza
#        donde (xi,yi) corresponden a las coordenadas de la articulacion i

# Para cada video de STAIRS se ha generado previamente mediante Inception un fichero para cada video con una secuencia
# de 90 (MIN_FRAMES) frames de 2048 elementos representando las features de la imagen

# Ahora vamos a generar un fichero NPZ por video. Seran 3 matrices de elementos:
#      [Matriz YOLO][Matriz OpenPose][Matriz Features]

# Cada matriz tendra 90 (MIN_FRAMES) elementos correspondientes a cada frame:

# MATRIZ_YOLO = frame 0 [conf.obj1 x1y1 x2y2  conf.obj2 x1y1 x2y2 ...]
#               frame 1 [conf.obj1 x1y1 x2y2  conf.obj2 x1y1 x2y2 ...]
#               frame 90 (MIN_FRAMES) [conf.obj1 x1y1 x2y2  conf.obj2 x1y1 x2y2 ...]

#                            PERSONA1               PERSONA2
# MATRIZ_OPENPOSE = frame 0 [x0y0 x1y1 .... x17y17  x0y0 x1y1 .... x17y17]
#                   frame 1 [x0y0 x1y1 .... x17y17  x0y0 x1y1 .... x17y17]
#                   frame 90 (MIN_FRAMES) [x0y0 x1y1 .... x17y17  x0y0 x1y1 .... x17y17]

# MATRIZ_FEATURES = frame 0 [f0 ... f2047]
# MATRIZ_FEATURES = frame 1 [f0 ... f2047]
# MATRIZ_FEATURES = frame 90 (MIN_FRAMES) [f0 ... f2047]

# Para cada video registraremos 90 frames. En cada frame registraremos la confianza de cada objeto y sus coordenadas relativas
# En el caso de las personas cogemos 2 personas, las dos de mayor confianza ordenando de izquierda a derecha de la imagen
# A mayores mostraremos las articulaciones de las 2 personas mas grandes detectadas con OpenPose en otros 90 frames

# En vez de 90 frames que utilizamos en STAIR, para NTU-RGB utilizamos 60 frames
import os
import csv
import numpy as np
from sklearn.utils import shuffle

# Para el caso de NTU-RGB
from accionesPermitidasNTURGB import accionValida
from accionesPermitidasNTURGB import objetos_validos_YOLO
# Para el caso de STAIRS
#from accionesPermitidas import accionValida
#from accionesPermitidas import objetos_validos_YOLO

from sklearn.model_selection import train_test_split
import sys
import cv2

# DIRECTORIO SECUENCIAS INCEPTION
DIRECTORIO_SECUENCIAS = "/datos/NTURGBD/DATOS-FEATURES"
# DIRECTORIO DATOS YOLO
DIRECTORIO_DATOS_YOLO5 = "/datos/NTURGBD/DATOS-YOLO5"
# DIRECTORIO DATOS OPENPOSE
DIRECTORIO_DATOS_OPENPOSE = "/datos/NTURGBD/DATOS-OPENPOSE"
# DIRECTORIO CON LOS VIDEOS DE ENTRADA
DIRECTORIO_VIDEOS_STAIRS="/datos/NTURGBD/VIDEOS_50_120"
# DIRECTORIO CON LOS VIDEOS ORIGINALES PARA COMPROBAR FRAMES
DIRECTORIO_VIDEOS_ORIGINALES="/datos/NTURGBD/VIDEOS_ORIGEN/nturgb+d_rgb"
# DIRECTORIO DE SALIDA
DIRECTORIO_SALIDA = "/datos/NTURGBD/sequences_features_OpenPose_YOLO_50_120"
# Generamos un fichero de datos
FICHERO_DATOS = "/datos/NTURGBD/data_NTURGB120_file_filtrado_shuffle_50_90.csv"

NUM_OBJETOS_YOLO = len(objetos_validos_YOLO)
MIN_FRAMES = 50     # En el caso de STAIRS usamos 90 frames. En NTU-RGB utilizamos 60 ya que hay videos de menos de 2 segundos
MAX_FRAMES = 90     # Como nuestro método utiliza acciones de duración fija, tenemos que descartar vídeos muy largos
MAX_ARTICULACIONES_OPENPOSE = 18
MAX_PERSONAS = 20

SALIDA_ANTICIPADA = False  # Nos permite parar proceso al generar CSV

# Si el directorio de salida no existe, lo creamos
if not os.path.exists(DIRECTORIO_SALIDA):
    os.mkdir(DIRECTORIO_SALIDA)

def buscarYOLO(objeto):
    for i in range(0, NUM_OBJETOS_YOLO):
        if str(objetos_validos_YOLO[i]).strip() == str(objeto).strip():
            return i
    return -1 

"""
En el caso de STAIRS el fichero ya está previamente generado
# Contamos videos para hacer porcentaje. Solo tenemos en cuenta los que tienen tamaño > 0:
videos_totales = 0
with open(FICHERO_DATOS) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    for row in csv_reader:
        videos_totales = videos_totales + 1
videos_procesados = 0

# Este array nos servira para contabilizar objetos que aparecen alguna vez
suma_objetos_YOLO = np.zeros(NUM_OBJETOS_YOLO)

# La secuencia estara formada para cada video por 90 frames con 51 objetos (2 personas) con 5 componentes por objeto
# En total 90x51x5 = 22950 valores por video
with open(FICHERO_DATOS) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    for row in csv_reader:
        tipo = row[0]
        accion = row[1]
        fichero = row[2]
        frames = row[3]
        
En el caso de NTU-RGB es diferente
"""

def leerFrames(video):
    cap = cv2.VideoCapture(video)
    if (cap.isOpened() == False):
        print("Error abriendo fichero " + video)     
        exit(1)
   
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return nframes    

# Generamos una lista con todos los videos y sus clases
# Contamos videos para hacer porcentaje. Solo tenemos en cuenta los que tienen tamaño > 0:
videos_totales = 0
acciones = []
ficheros = []
tipo = []   # Train/Test

cont1 = 0
cont2 = 0

for base, dirs, files in sorted(os.walk(DIRECTORIO_VIDEOS_STAIRS)):
    accion_STAIRS = base.split(os.path.sep)[-1]
    if accion_STAIRS != "VIDEOS":
        for archivo_video in files:
            
            nframes = leerFrames(os.path.join(DIRECTORIO_VIDEOS_ORIGINALES, archivo_video))
            cont1 += 1
            #if (nframes >= MIN_FRAMES and nframes <= MAX_FRAMES):
            if (nframes >= MIN_FRAMES and nframes < 120):
                acciones.append(accion_STAIRS)
                archivo_sin_extension = os.path.splitext(archivo_video)[0]
                ficheros.append(archivo_sin_extension)
                videos_totales += 1
                cont2 += 1
videos_procesados = 0

print("Contador 1: " + str(cont1))
print("Contador 2: " + str(cont2))

# Hacemos un shuffle de los datos
acciones, ficheros = shuffle(np.array(acciones), np.array(ficheros))

# Establecemos si son train o test utilizando un reparto stratified
f_train, f_test, a_train, a_test = train_test_split(ficheros, acciones, test_size=0.10, random_state=42, stratify=acciones)

# Ahora creamos el tipo para indicar si son de train o test
f_sort = f_test.tolist()
f_sort.sort()

for posi in range(0, len(acciones)):
    if ficheros[posi] in f_sort:
        tipo.append("test")
    else:
        tipo.append("train")

"""aleatorio = np.random.randint(10, size=len(acciones))
for posi in range(0, len(acciones)):
    # Un 10% a test. Si el aleatorio es el 5, lo metemos en test, en caso contrario en train
    if aleatorio[posi] == 5:
        tipo.append("test")
    else:
        tipo.append("train")"""

# Generamos el fichero de los datos que se necesita para implementar el dataset del entrenamiento
with open(FICHERO_DATOS, 'w', newline='') as file:
    writer = csv.writer(file)
    for posi in range(0, len(acciones)):
        writer.writerow([tipo[posi], acciones[posi], ficheros[posi], MIN_FRAMES])

if (SALIDA_ANTICIPADA == True):
    exit(0)

# Este array nos servira para contabilizar objetos que aparecen alguna vez
suma_objetos_YOLO = np.zeros(NUM_OBJETOS_YOLO)

# La secuencia estara formada para cada video por 90 (MIN_FRAMES) frames con 51 objetos (2 personas) con 5 componentes por objeto
# si MIN_FRAMES = 90, en total 90x51x5 = 22950 valores por video
for posi in range(0, len(acciones)):
    accion = acciones[posi]
    fichero = ficheros[posi]
    frames = MIN_FRAMES
    
    confianza_objeto_encontrado =  np.zeros((MIN_FRAMES, NUM_OBJETOS_YOLO))
    x1 = np.zeros((MIN_FRAMES, NUM_OBJETOS_YOLO))
    y1 = np.zeros((MIN_FRAMES, NUM_OBJETOS_YOLO))
    x2 = np.zeros((MIN_FRAMES, NUM_OBJETOS_YOLO))
    y2 = np.zeros((MIN_FRAMES, NUM_OBJETOS_YOLO))
    
    confianza_persona_2 = float(0.0)
    x1_persona_2 = float(0.0)
    y1_persona_2 = float(0.0)
    x2_persona_2 = float(0.0)
    y2_persona_2 = float(0.0)
    
    # Omitimos las acciones no validas
    if accionValida(accion) == False:
        continue
    
    # Si el fichero de datos YOLO u OpenPose no existe, no continuamos con ese caso
    if not os.path.exists(DIRECTORIO_DATOS_YOLO5 + '/' + accion + '/' + fichero + '.dat'):
        continue
    if not os.path.exists(DIRECTORIO_DATOS_OPENPOSE + '/' + accion + '/' + fichero + '.dat'):
        continue
    # Si el fichero de features no existe, no continuamos con ese caso
    # fichero_secuencias = DIRECTORIO_SECUENCIAS + '/' + fichero + '-' + str(MIN_FRAMES) + '-features.npy'
    fichero_secuencias = DIRECTORIO_SECUENCIAS + '/' + accion + '/' + fichero + '.npy'
    if not os.path.exists(fichero_secuencias):
        continue

    # Primero leemos el fichero OpenPose en memoria (MAX_PERSONAS personas maximo detectadas)
    x_op = np.zeros((MIN_FRAMES, MAX_PERSONAS, MAX_ARTICULACIONES_OPENPOSE))
    y_op = np.zeros((MIN_FRAMES, MAX_PERSONAS, MAX_ARTICULACIONES_OPENPOSE))

    # Abrimos fichero del video
    with open(DIRECTORIO_DATOS_OPENPOSE + '/' + accion + '/' + fichero + ".dat") as csv_file_dat:
        csv_reader_dat = csv.reader(csv_file_dat, delimiter=';')

        for row1 in csv_reader_dat:            
            a_STAIRS = row1[0]
            nombre_fichero = row1[1]                
            numero_frame = int(row1[2])
            ancho_frame = float(row1[3])
            alto_frame = float(row1[4])
            idPersona = int(row1[5])

            if numero_frame > (MIN_FRAMES - 1):   # Procesamos 90 (MIN_FRAMES) frames por video
                break
            
            if idPersona >= 10:
                print("Video con mas de 10 personas: " + DIRECTORIO_DATOS_OPENPOSE + "/" + accion + "/" + fichero + ".dat")
                print("Frame: " + str(numero_frame))
            
            # Normalizamos las articulaciones entre el ancho y alto del escenario                
            for i in range(MAX_ARTICULACIONES_OPENPOSE):
                x_op[numero_frame][idPersona][i] = float(row1[6 + 2*i]) / ancho_frame
                y_op[numero_frame][idPersona][i] = float(row1[7 + 2*i]) / alto_frame

    # IMPORTANTE
    # Creamos un bounding box de las personas destacadas con OpenPose a nivel de video completo
    # No lo hacemos a nivel de frame ya que si desaparece o aparecen personas, los valores no
    # serian correctos ya que se podrian mover los bounding box mucho entre frames
    min_bb_x = float(1.0)
    min_bb_y = float(1.0)
    max_bb_x = float(0.0)
    max_bb_y = float(0.0)
    
    # Cogemos las personas mas grandes detectadas con OpenPose y las ponemos a izquierda y derecha
    # Solo cogemos las que tienen cuello
    for i in range (0, MIN_FRAMES):
        tam1 = float(0.0)
        tam2 = float(0.0)
        p1 = -1
        p2 = -1
        for p in range(MAX_PERSONAS):
            # Solo si tienen cuello
            if x_op[i][p][1] != 0 and y_op[i][p][1] != 0:
                # Ahora calculamos el tamaño de la persona                    
                min_x = float(1.0)
                min_y = float(1.0)
                max_x = float(0.0)
                max_y = float(0.0)
                for j in range(MAX_ARTICULACIONES_OPENPOSE):
                    if x_op[i][p][j] != 0 and y_op[i][p][j] != 0:
                        if x_op[i][p][j] > max_x:
                            max_x = x_op[i][p][j]
                        if y_op[i][p][j] > max_y:
                            max_y = y_op[i][p][j]
                        if x_op[i][p][j] < min_x:
                            min_x = x_op[i][p][j]
                        if y_op[i][p][j] < min_y:
                            min_y = y_op[i][p][j]
                tam = (max_x -min_x) * (max_y -min_y)
                if tam != 0:
                    if p1 == -1:
                        tam1 = tam
                        p1 = p
                    else:
                        if p2 == -1:
                            tam2 = tam
                            p2 = p
                        else:
                            if tam > tam1 and tam < tam2:
                                tam1 = tam
                                p1 = p     
                            else:
                                if tam > tam2 and tam < tam1:
                                    tam2 = tam
                                    p2 = p     
                                else:
                                    if tam > tam1 and tam > tam2:
                                        if tam1 < tam2:
                                            tam1 = tam
                                            p1 = p
                                        else:
                                            if tam2 < tam1:
                                                tam2 = tam
                                                p2 = p

        # Ahora buscamos minimos y maximos de los Bounding Box
        # Buscamos minimo y maximo para las dos personas seleccionadas
        for j in range(MAX_ARTICULACIONES_OPENPOSE):
            if p1 != -1:
                if x_op[i][p1][j] != 0 and y_op[i][p1][j] != 0:
                    if x_op[i][p1][j] > max_bb_x:
                        max_bb_x = x_op[i][p1][j]
                    if y_op[i][p1][j] > max_bb_y:
                        max_bb_y = y_op[i][p1][j]
                    if x_op[i][p1][j] < min_bb_x:
                        min_bb_x = x_op[i][p1][j]
                    if y_op[i][p1][j] < min_bb_y:
                        min_bb_y = y_op[i][p1][j]
            if p2 != -1:                            
                if x_op[i][p2][j] != 0 and y_op[i][p2][j] != 0:
                    if x_op[i][p2][j] > max_bb_x:
                        max_bb_x = x_op[i][p2][j]
                    if y_op[i][p2][j] > max_bb_y:
                        max_bb_y = y_op[i][p2][j]
                    if x_op[i][p2][j] < min_bb_x:
                        min_bb_x = x_op[i][p2][j]
                    if y_op[i][p2][j] < min_bb_y:
                        min_bb_y = y_op[i][p2][j]

    # Las coordenadas del bounding box tienen que ser diferentes de los limites de los
    # esqueletos. De lo contrario, al cambiar el sistema de referencia respecto al BB, las coordenadas
    # de la izquierda/arriba de la persona se podrian perder. Usamos un 1% del tamaño del ancho/alto
    # Lo usamos tanto para minimos como maximos
    min_bb_x = min_bb_x - 0.01
    min_bb_y = min_bb_y - 0.01
    max_bb_x = max_bb_x + 0.01
    max_bb_y = max_bb_y + 0.01
    if min_bb_x < 0.0:
        min_bb_x = float(0.0)
    if min_bb_y < 0.0:
        min_bb_y = float(0.0)        
    if max_bb_x > 1.0:
        max_bb_x = float(1.0)
    if max_bb_y > 1.0:
        max_bb_y = float(1.0)

    # Abrimos fichero del video
    with open(DIRECTORIO_DATOS_YOLO5 + '/' + accion + '/' + fichero + ".dat") as csv_file_dat:
        csv_reader_dat = csv.reader(csv_file_dat, delimiter=';')
        
        # solo admitimos un objeto por frame ya que podria darse el caso, por ejemplo, de un
        # frame donde aparecen 100 libros, y no tendria sentido contabilizar todos

        for row1 in csv_reader_dat:
            a_STAIRS = row1[0]
            clase_YOLO = str(row1[1])
            nombre_fichero = row1[2]
            numero_frame = int(row1[3]) - 1
            ancho_frame = float(row1[4])
            alto_frame = float(row1[5])
            rx1 = float(row1[6]) / ancho_frame
            ry1 = float(row1[7]) / alto_frame
            rx2 = float(row1[8]) / ancho_frame
            ry2 = float(row1[9]) / alto_frame
            confianza = float(row1[10])
            
            if numero_frame > (MIN_FRAMES - 1):   # Procesamos 90 (MIN_FRAMES) frames por video
                break
            
            oY = buscarYOLO(clase_YOLO)
            
            # La persona tiene el valor oY = 0
          
            # Para cada frame nos quedamos con el objeto detectado con mayor confianza  
            if oY != 0:                  
                if confianza > 0 and (confianza_objeto_encontrado[numero_frame][oY] == 0 or 
                    confianza_objeto_encontrado[numero_frame][oY] < confianza):
                    confianza_objeto_encontrado[numero_frame][oY] = confianza
                    x1[numero_frame][oY] = rx1                    
                    y1[numero_frame][oY] = ry1
                    x2[numero_frame][oY] = rx2
                    y2[numero_frame][oY] = ry2
            else:
                if confianza > 0:
                    if confianza_objeto_encontrado[numero_frame][oY] == 0:
                        confianza_objeto_encontrado[numero_frame][oY] = confianza
                        x1[numero_frame][oY] = rx1                    
                        y1[numero_frame][oY] = ry1
                        x2[numero_frame][oY] = rx2
                        y2[numero_frame][oY] = ry2
                    else:
                        if confianza_persona_2 == 0:
                            confianza_persona_2 = confianza
                            x1_persona_2 = rx1                    
                            y1_persona_2 = ry1
                            x2_persona_2 = rx2
                            y2_persona_2 = ry2
                        else:
                            if ((confianza > confianza_objeto_encontrado[numero_frame][oY] and
                               confianza > confianza_persona_2 and
                               confianza_objeto_encontrado[numero_frame][oY] >= confianza_persona_2) or
                               (confianza > confianza_persona_2 and 
                               confianza <= confianza_objeto_encontrado[numero_frame][oY])):
                                confianza_persona_2 = confianza
                                x1_persona_2 = rx1
                                y1_persona_2 = ry1
                                x2_persona_2 = rx2
                                y2_persona_2 = ry2
                            else:
                                if ((confianza > confianza_objeto_encontrado[numero_frame][oY] and
                                   confianza > confianza_persona_2 and
                                   confianza_objeto_encontrado[numero_frame][oY] < confianza_persona_2) or
                                   (confianza > confianza_objeto_encontrado[numero_frame][oY] and 
                                   confianza <= confianza_persona_2)):                                   
                                    confianza_objeto_encontrado[numero_frame][oY] = confianza
                                    x1[numero_frame][oY] = rx1                    
                                    y1[numero_frame][oY] = ry1
                                    x2[numero_frame][oY] = rx2
                                    y2[numero_frame][oY] = ry2

    # Ahora escribimos los datos en la matriz YOLO que grabaremos en el fichero NPZ
    matriz_yolo = np.zeros((MIN_FRAMES, NUM_OBJETOS_YOLO + 1, 5))

    for i in range (0, MIN_FRAMES):
        for j in range (0, NUM_OBJETOS_YOLO):                
            if confianza_objeto_encontrado[i][j] > 0:
                suma_objetos_YOLO[j] = suma_objetos_YOLO[j] + 1
            if j != 0:
                matriz_yolo[i][j+1][0] = confianza_objeto_encontrado[i][j]
                matriz_yolo[i][j+1][1] = x1[i][j]
                matriz_yolo[i][j+1][2] = y1[i][j]
                matriz_yolo[i][j+1][3] = x2[i][j]
                matriz_yolo[i][j+1][4] = y2[i][j]
            else:
                # En el caso de personas lo hacemos de una manera diferente
                if confianza_persona_2 > 0:
                    suma_objetos_YOLO[j] = suma_objetos_YOLO[j] + 1                            
                    if (((x1[i][j] + x2[i][j])/2) < ((x1_persona_2 + x2_persona_2)/2)):
                        # Primero persona 1
                        matriz_yolo[i][0][0] = confianza_objeto_encontrado[i][j]
                        matriz_yolo[i][0][1] = x1[i][j]
                        matriz_yolo[i][0][2] = y1[i][j]
                        matriz_yolo[i][0][3] = x2[i][j]
                        matriz_yolo[i][0][4] = y2[i][j]            
                        # Despues persona 2
                        matriz_yolo[i][1][0] = confianza_persona_2
                        matriz_yolo[i][1][1] = x1_persona_2
                        matriz_yolo[i][1][2] = y1_persona_2
                        matriz_yolo[i][1][3] = x2_persona_2
                        matriz_yolo[i][1][4] = y2_persona_2
                    else:
                        # Primero persona 2
                        matriz_yolo[i][0][0] = confianza_persona_2
                        matriz_yolo[i][0][1] = x1_persona_2
                        matriz_yolo[i][0][2] = y1_persona_2
                        matriz_yolo[i][0][3] = x2_persona_2
                        matriz_yolo[i][0][4] = y2_persona_2
                        # Despues persona 1
                        matriz_yolo[i][1][0] = confianza_objeto_encontrado[i][j]
                        matriz_yolo[i][1][1] = x1[i][j]
                        matriz_yolo[i][1][2] = y1[i][j]
                        matriz_yolo[i][1][3] = x2[i][j]
                        matriz_yolo[i][1][4] = y2[i][j]
                else:
                    # Primero persona 1
                    matriz_yolo[i][0][0] = confianza_objeto_encontrado[i][j]
                    matriz_yolo[i][0][1] = x1[i][j]
                    matriz_yolo[i][0][2] = y1[i][j]
                    matriz_yolo[i][0][3] = x2[i][j]
                    matriz_yolo[i][0][4] = y2[i][j]            
                    # Despues persona 2
                    matriz_yolo[i][1][0] = confianza_persona_2
                    matriz_yolo[i][1][1] = x1_persona_2
                    matriz_yolo[i][1][2] = y1_persona_2
                    matriz_yolo[i][1][3] = x2_persona_2
                    matriz_yolo[i][1][4] = y2_persona_2                        

    # Ahora escribimos los datos en la matriz OpenPose que grabaremos en el fichero NPZ (2 personas maximo y
    #  2 coordenadas por articulacion)
    matriz_op = np.zeros((MIN_FRAMES, 2, MAX_ARTICULACIONES_OPENPOSE, 2))
    # Cogemos las personas mas grandes detectadas con OpenPose y las ponemos a izquierda y derecha
    # Solo cogemos las que tienen cuello
    for i in range (0, MIN_FRAMES):
        tam1 = float(0.0)
        tam2 = float(0.0)
        p1 = -1
        p2 = -1
        for p in range(MAX_PERSONAS):
            # Solo si tienen cuello
            if x_op[i][p][1] != 0 and y_op[i][p][1] != 0:
                # Ahora calculamos el tamaño de la persona                    
                min_x = float(1.0)
                min_y = float(1.0)
                max_x = float(0.0)
                max_y = float(0.0)
                for j in range(MAX_ARTICULACIONES_OPENPOSE):
                    if x_op[i][p][j] != 0 and y_op[i][p][j] != 0:
                        if x_op[i][p][j] > max_x:
                            max_x = x_op[i][p][j]
                        if y_op[i][p][j] > max_y:
                            max_y = y_op[i][p][j]
                        if x_op[i][p][j] < min_x:
                            min_x = x_op[i][p][j]
                        if y_op[i][p][j] < min_y:
                            min_y = y_op[i][p][j]
                tam = (max_x -min_x) * (max_y -min_y)
                if tam != 0:
                    if p1 == -1:
                        tam1 = tam
                        p1 = p
                    else:
                        if p2 == -1:
                            tam2 = tam
                            p2 = p
                        else:
                            if tam > tam1 and tam < tam2:
                                tam1 = tam
                                p1 = p     
                            else:
                                if tam > tam2 and tam < tam1:
                                    tam2 = tam
                                    p2 = p     
                                else:
                                    if tam > tam1 and tam > tam2:
                                        if tam1 < tam2:
                                            tam1 = tam
                                            p1 = p
                                        else:
                                            if tam2 < tam1:
                                                tam2 = tam
                                                p2 = p
        # Ordenamos las personas segun la posicion de su cuello
        if p2 != -1:
            if x_op[i][p2][1] < x_op[i][p1][1]:
                tam_tmp = tam1
                p_tmp = p1
                tam1 = tam2
                p1 = p2
                tam2 = tam_tmp
                p2 = p_tmp

        # Ahora las normalizamos respecto al bounding box del video
        for j in range(MAX_ARTICULACIONES_OPENPOSE):
            if p1 != -1:
                # Solo normalizamos si existe la articulacion
                if x_op[i][p1][j] != 0 and y_op[i][p1][j] != 0:
                    x_op[i][p1][j] = (x_op[i][p1][j] - min_bb_x) / (max_bb_x - min_bb_x)
                    y_op[i][p1][j] = (y_op[i][p1][j] - min_bb_y) / (max_bb_y - min_bb_y)
                    valor_distinto_cero = 1
            if p2 != -1:
                # Solo normalizamos si existe la articulacion
                if x_op[i][p2][j] != 0 and y_op[i][p2][j] != 0:
                    x_op[i][p2][j] = (x_op[i][p2][j] - min_bb_x) / (max_bb_x - min_bb_x)                    
                    y_op[i][p2][j] = (y_op[i][p2][j] - min_bb_y) / (max_bb_y - min_bb_y)
                    valor_distinto_cero = 1                        

        # Construimos vector con las articulaciones normalizadas respecto al bounding box de
        # las dos personas
        if p1 != -1:
            for j in range(MAX_ARTICULACIONES_OPENPOSE):
                matriz_op[i][0][j][0] = x_op[i][p1][j]
                matriz_op[i][0][j][1] = y_op[i][p1][j]
        if p2 != -1:
            for j in range(MAX_ARTICULACIONES_OPENPOSE):
                matriz_op[i][1][j][0] = x_op[i][p2][j]
                matriz_op[i][1][j][1] = y_op[i][p2][j]

    # Cargamos matriz de features. Por defecto la ponemos a ceros
    matriz_features = np.zeros((MIN_FRAMES, 2048))
    if os.path.exists(fichero_secuencias):
        matriz_features = np.load(fichero_secuencias)
        matriz_features = matriz_features[:MIN_FRAMES]

    # Hacemos un flatten del array de OpenPose y YOLO
    matriz_yolo_flat = np.zeros((MIN_FRAMES, (NUM_OBJETOS_YOLO + 1) * 5))
    matriz_op_flat = np.zeros((MIN_FRAMES, (MAX_ARTICULACIONES_OPENPOSE) * 4))
    for n in range(MIN_FRAMES):
        matriz_yolo_flat[n] = matriz_yolo[n].flatten()
        matriz_op_flat[n] = matriz_op[n].flatten()
    
    # Grabamos el fichero con las 3 matrices completas
    fichero_salida = DIRECTORIO_SALIDA + '/' + fichero + '-' + str(MIN_FRAMES) + '-yolo_op_features.npz'
    if os.path.exists(fichero_salida):        
        os.remove(fichero_salida)
    np.savez(fichero_salida, YOLO=matriz_yolo_flat, OpenPose=matriz_op_flat, FeaturesInception=matriz_features)

    videos_procesados = videos_procesados +1
    avance = round(100 * float(videos_procesados) / float(videos_totales), 2)
    print("Porcentaje de avance = " + str(avance))
    
print("Proceso terminado correctamente.")
print("Resumen apariciones objetos YOLO:")
for p in range(0, NUM_OBJETOS_YOLO):
    print(objetos_validos_YOLO[p] + " = " + str(suma_objetos_YOLO[p]))
