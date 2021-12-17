# Extraccion de poses de las personas con OpenPose. Utilizamos la version de OpenCV.
# Autor: Jaime Duque Domingo

# Llamamos varias veces al script de la siguiente manera:
#   python3 paso2_procesarVideosOpenPoseParalelo.py accion1 accion2
# Procesara desde la accion1 hasta la accion2    
#   Ejemplo:    python3 paso2_procesarVideosOpenPoseParalelo.py smoking taking_photo

# Este programa lee todos los videos del dataset STAIRS y extrae fichero a fichero las poses 
# detectados en cada frame de cada video. El proceso crea el directorio de salida si no existe
# y detecta los videos que se han escrito. Si uno ya esta escrito, no se vuelve a escribir
# Sera necesario borrar la carpeta de salida para resetear el proceso

# Para cada directorio en la carpeta de entrada (accion) se crea un directorio en la carpeta de salida
# Para cada video xxxxx.mp4 se crea un fichero de datos xxxxx.dat que incluye los objetos detectados 
# con el siguiente formato:

# accion_STAIRS; nombre_fichero; numero_frame; ancho_frame; alto_frame; id_persona; x0; y0; x1; y1; x2; y2; ... ; x25; y25; confianza
#        donde (xi,yi) corresponden a las coordenadas de la articulacion i

# Import framework
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
accion1=sys.argv[1]
accion2=sys.argv[2]

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#sys.path.remove('/home/disa/catkin_ws/devel/lib/python2.7/dist-packages')
#sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2
print("Version OpenCV: " + cv2.__version__)

import numpy as np
import time
#import os

#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

DIRECTORIO_VIDEOS_STAIRS="/datos/NTURGBD/VIDEOS_50_120"
DIRECTORIO_SALIDA="/datos/NTURGBD/DATOS-OPENPOSE"
MAX_FRAMES = 50

protoFile = "/datos/NTURGBD/pose_deploy_linevec.prototxt"
weightsFile = "/datos/NTURGBD/pose_iter_440000.caffemodel"

nPoints = 18
# COCO Output Format

keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []

    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output, frameWidth, frameHeight, detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs



# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
print("Using GPU device")

def detect(video, accion_STAIRS):
    nombre_fichero = os.path.basename(video)
    cap = cv2.VideoCapture(video)

    n_frame = 0
    res = ""    

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:    
            # Fix the input Height and get the width according to the Aspect Ratio
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            
            inHeight = 368
            inWidth = int((inHeight/frameHeight)*frameWidth)
            
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                      (0, 0, 0), swapRB=False, crop=False)
            
            t = time.time()            
            net.setInput(inpBlob)
            output = net.forward()
            #print("Time Taken in forward pass = {}".format(time.time() - t))
            
            detected_keypoints = []
            keypoints_list = np.zeros((0,3))
            keypoint_id = 0
            threshold = 0.1
            
            for part in range(nPoints):
                probMap = output[0,part,:,:]
                probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
                keypoints = getKeypoints(probMap, threshold)
                #print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
                keypoints_with_id = []
                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1
            
                detected_keypoints.append(keypoints_with_id)

            """frameClone = image1.copy()
            for i in range(nPoints):
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
            cv2.imshow("Keypoints",frameClone)"""
            
            valid_pairs, invalid_pairs = getValidPairs(output, frameWidth, frameHeight, detected_keypoints)
            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)
            
            max_personas = 20
            # Almacenamos el resultado de las personas encontradas
            resultado_x = np.zeros((max_personas, nPoints), dtype=int)
            resultado_y = np.zeros((max_personas, nPoints), dtype=int)
            ok_persona = np.zeros(max_personas, dtype=int)
            
            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    #cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                    ok_persona[n] = 1
                    resultado_x[n][np.array(POSE_PAIRS[i])[0]] = int(B[0])
                    resultado_y[n][np.array(POSE_PAIRS[i])[0]] = int(A[0])
                    resultado_x[n][np.array(POSE_PAIRS[i])[1]] = int(B[1])
                    resultado_y[n][np.array(POSE_PAIRS[i])[1]] = int(A[1])

            # Write results
            for n in range(len(personwiseKeypoints)):                
                # accion_STAIRS; nombre_fichero; numero_frame; ancho_frame; alto_frame; id_persona; x0; y0; x1; y1; x2; y2; ... ; x25; y25; confianza
                #        donde (xi,yi) corresponden a las coordenadas de la articulacion i
                if ok_persona[n] == 1:
                    res = res + (accion_STAIRS + "; " + nombre_fichero + "; " + str(n_frame) + "; " +
                        str(frameWidth) + "; " + str(frameHeight) + "; " + str(n) )
                    
                    for i in range(nPoints):
                        res = res + "; " + str(resultado_x[n][i]) + "; " + str(resultado_y[n][i])
                    res = res + "\n"
  
            n_frame = n_frame + 1
            if n_frame == MAX_FRAMES:
                break            
        else:
            break

    cap.release()
    return res    

# Si el directorio de salida no existe, lo creamos
if not os.path.exists(DIRECTORIO_SALIDA):
    os.mkdir(DIRECTORIO_SALIDA)

# Control de la ejecucion en paralelo
empezar = accion1
finalizar = accion2
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
