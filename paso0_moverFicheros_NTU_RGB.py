import csv
import os
import shutil
import cv2
import numpy as np

"""
Reconocimiento de acciones.
Jaime Duque Domingo.
Realizamos un preproceso de los vídeos de NTU-RGB
https://github.com/STAIR-Lab-CIT/STAIR-actions

Movemos los ficheros del dataset a sus carpetas respectivas. Los vídeos, originalmente de 1920x1080, los reducimos a unas dimensiones de 640x360 para aumentar 
la velocidad de procesamiento. Además, como muchos son de 2 segundos de tamaño, a diferencia de STAIR que eran 3 segundos, cortamos en 60 frames el resultado
Los vídeos con menos de 60 frames son descartados.

Video samples have been captured by three Microsoft Kinect V2 cameras concurrently. The resolutions of RGB videos are 1920×1080

Each file/folder name in both datasets is in the format of SsssCcccPpppRrrrAaaa (e.g., S001C002P003R002A013), in which sss is the setup number, 
ccc is the camera ID, ppp is the performer (subject) ID, rrr is the replication number (1 or 2), and aaa is the action class label.

The "NTU RGB+D" dataset includes the files/folders with setup numbers between S001 and S017, while the "NTU RGB+D 120" dataset includes the 
files/folders with setup numbers between S001 and S032.

For more details about the setups, camera IDs, ..., please refer to the "NTU RGB+D" dataset paper and the "NTU RGB+D 120" dataset paper.

Las carpetas a utilizar irán de 1 a 60, ya que hay 60 carpetas.

Acciones:

A1. drink water.
A2. eat meal/snack.
A3. brushing teeth.
A4. brushing hair.
A5. drop.
A6. pickup.
A7. throw.
A8. sitting down.
A9. standing up (from sitting position).
A10. clapping.
A11. reading.
A12. writing.
A13. tear up paper.
A14. wear jacket.
A15. take off jacket.
A16. wear a shoe.
A17. take off a shoe.
A18. wear on glasses.
A19. take off glasses.
A20. put on a hat/cap.
A21. take off a hat/cap.
A22. cheer up.
A23. hand waving.
A24. kicking something.
A25. reach into pocket.
A26. hopping (one foot jumping).
A27. jump up.
A28. make a phone call/answer phone.
A29. playing with phone/tablet.
A30. typing on a keyboard.
A31. pointing to something with finger.
A32. taking a selfie.
A33. check time (from watch).
A34. rub two hands together.
A35. nod head/bow.
A36. shake head.
A37. wipe face.
A38. salute.
A39. put the palms together.
A40. cross hands in front (say stop).
A41. sneeze/cough.
A42. staggering.
A43. falling.
A44. touch head (headache).
A45. touch chest (stomachache/heart pain).
A46. touch back (backache).
A47. touch neck (neckache).
A48. nausea or vomiting condition.
A49. use a fan (with hand or paper)/feeling warm.
A50. punching/slapping other person.
A51. kicking other person.
A52. pushing other person.
A53. pat on back of other person.
A54. point finger at the other person.
A55. hugging other person.
A56. giving something to other person.
A57. touch other person's pocket.
A58. handshaking.
A59. walking towards each other.
A60. walking apart from each other.

"""

RUTA_VIDEOS="/datos/NTURGBD/VIDEOS_ORIGEN/nturgb+d_rgb"
RUTA_SALIDA="/datos/NTURGBD/VIDEOS"
MIN_FRAMES=60
MAX_FRAMES=240  # Descartamos acciones con más de 240 frames
WIDTH=640
HEIGHT=360

# Creamos directorio de salida si no existe
if not os.path.exists(RUTA_SALIDA):
    os.mkdir(RUTA_SALIDA)

def procesarVideo(carpeta, fichero):
    # Verificamos si existe la clase SsssCcccPpppRrrrAaaa
    # La carpeta será Aaaa
    clase = fichero[16:20]
    # print(clase)
    # Creamos directorio de clase si no existe
    psal = os.path.join(RUTA_SALIDA, clase)
    if not os.path.exists(psal):
        os.mkdir(psal)    
    
    # Si no existe el vídeo anteriormente, lo generamos
    if not os.path.exists(os.path.join(psal, fichero)):
        # Abrimos el vídeo y lo reescalamos
        # Debe tener más de 60 fps
        cap = cv2.VideoCapture(os.path.join(carpeta, fichero))
        if (cap.isOpened() == False):
            print("Error abriendo fichero " + fichero)     
            exit(1)
       
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames >= MIN_FRAMES and frames <= MAX_FRAMES:
            out = cv2.VideoWriter(os.path.join(psal, fichero), cv2.VideoWriter_fourcc('M','J','P','G'), 30, (WIDTH, HEIGHT))

            c = 0            
            while(True):
                ret, frame = cap.read()
            
                if ret == True:
                    out.write( cv2.resize(frame, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA) )                    
                    c += 1
                else:
                    print("Error procesando fichero " +  fichero)
                    exit(2)
                if c >= MIN_FRAMES:
                    break

            out.release()
        cap.release()

# Contamos ficheros de directorio
total = 0
contenido = os.listdir(RUTA_VIDEOS)
for fichero in contenido:
    if os.path.isfile(os.path.join(RUTA_VIDEOS, fichero)) and fichero.endswith(".avi"):
        total += 1

# Procesamos los ficheros
actual = 0

contenido = os.listdir(RUTA_VIDEOS)
for fichero in contenido:
    if os.path.isfile(os.path.join(RUTA_VIDEOS, fichero)):
        p = 100.0 * float(actual) / float(total)
        print("Porcentaje: " + str(round(p, 2)) + " - Fichero: " + fichero + ".")
        procesarVideo(RUTA_VIDEOS, fichero)
        actual += 1

