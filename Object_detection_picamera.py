######## Détection de 100 objets avec l'outil TensorFlow #########

# Date: 19/09/2020



# Importer les paquets suivants
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
# Le programme fonctionne uniquement avec la version 1 de TensorFlow
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

# Définition de la résolution de la caméra (pixels)
IM_WIDTH = 800
IM_HEIGHT = 608
#IM_WIDTH = 640    Pour optimiser le framerate, préférer une résolution moindre
#IM_HEIGHT = 480   

# Choix entre la Picamera et une webcam usb  

camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# 
sys.path.append('..')

# Importer les utilitaires depuis le dossier utils 
from utils import label_map_util
from utils import visualization_utils as vis_util

# Nom du répertoire contenant le modèle d'entrainement 
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'


CWD_PATH = os.getcwd()

# Chemin d'accès au fichier .pb du graphe de détection qui contient le modèle
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Dossier qui contient la liste des différents objets
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Nombre total des différents objet à détecter
NUM_CLASSES = 90


# Chaque catégorie d'objets est défini par un n° issue de la prédiction
# exemple : 44 = bouteille
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Charger le modèle en mémoire 
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Définir des tenseurs d'entrée et de sortie (c'est-à-dire les données)
# pour le classificateur de détection d'objets

# charger l'entrée 
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Les tenseurs de sortie sont les boîtes de détection, les scores et les classes d'objets
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Chaque score représente le niveau de confiance pour chacun des objets
# Le score est affiché sur l'image du résultat, avec le libellé de l'objet détecté
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Nombre d'objets détectés
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialisation du calcul de la fréquence d'images
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Cette partie du programme prend en compte le choix
# entre une Picamera et une Webcam usb 

### Picamera ###
if camera_type == 'picamera':
    # Initialisation de la Picamera
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        

        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Effectuer la détection réelle en exécutant le modèle avec l'image en entrée 
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Dessiner le contour de la détection d'objets(s)
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=0.40)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),6,cv2.LINE_AA)

        # Les résultats et les contours sont prêts à être affichés sur la vidéo
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Presser 'q' ou 'Ctrl + c' pour quitter le programme 
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

### Webcam usb ###
elif camera_type == 'usb':
    # Initialisation de la caméra usb
    camera = cv2.VideoCapture(0)
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)

    while(True):

        t1 = cv2.getTickCount()

        # Idem que la Picamera
        ret, frame = camera.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Idem que la Picamera
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Idem que la Picamera
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.85)

      

        # Idem que la Picamera
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()

cv2.destroyAllWindows()

