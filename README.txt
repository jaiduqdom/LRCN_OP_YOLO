Improving Human Activity Recognition Integrating LSTM with Different Data Sources: Features, Object Detection and Skeleton Tracking

Example of how generate features, 3D skeleton data and object detections and how to be trained with our integrated architecture

Before running the code:

1. Download STAIR dataset and verify that all folders of the code point to the right location.
2. Install OpenPose and YOLO v5.
3. Generate YOLO, OpenPose and Features data with:
	python3 paso1_procesarVideosYOLO.py assisting_in_getting_up end
	python3 paso2_procesarVideosOpenPoseParalelo.py assisting_in_getting_up end
	python3 paso3_getFeatures.py assisting_in_getting_up end
4. Join data:
	python3 paso4_generarDatosEntrenamiento_FeaturesOpenPoseYOLO_NPZ.py
5. Train integrated model:
	python3 paso5_entrenarAcciones_YOLO_OpenPose_Features.py
6. Evaluate model:
	python3 paso6_evaluarModeloF1Score.py

Optionally, for other datasets such as NTU-RGB-D, an initial distribution of data should be done:
	python3 paso0_moverFicheros_NTU_RGB.py

