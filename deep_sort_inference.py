# algoritmo de inferência com aplicação do deep-sort para a YOLOv8

# importação dos pacotes utilitários
import os
import cv2
import random
import numpy as np
from tracker import Tracker
from ultralytics import YOLO
from utils import bbox_corrections
from utils import segmentation_map_corrections

# array para definir a cor da segmentação no padrão RGB
SEGMENTATION_COLOR = [255, 0, 0]
BBOX_SELECTED_COLOR = [0, 255, 0]
BBOX_UNSELECTED_COLOR = [0, 0, 255]
BBOX_TEXT_COLOR = [0, 0, 0]
# definindo um limiar de confiabilidade na predição feita pelo classificador da YOLO
DETECTION_THRESHOLD = 0.5

# vídeo para realização de testes de inferência
video_path = os.path.join('.', 'data', 'people.mp4')
# local onde os vídeos com as inferências serão salvos
video_out_path = os.path.join('.', 'videos-teste', 'out.mp4')

# inicializando o módulo de captura de vídeo do opencv (0 para acessar a câmera)
cap = cv2.VideoCapture(0)
# captura do vídeo frame a frame
ret, frame = cap.read()

# inicializando o módulo de salvamento de vídeo do opencv
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

# carregando os parâmetros pré-treinados na arquitetura da YOLOv8
model = YOLO("modelos-salvos/yolov8n-seg.pt")

# inicializando o módulo de rastreamento do deep-sort
tracker = Tracker()

# criando um arranjo de escolha de corres para diferencias as caixas delimitadoras por IDs
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# variável auxiliar para adicionar o tempo de espera entre um frame e outro
time = 1
# variável auxiliar para adicionar o ID selecionado pelo usuário 
id_select = 0
# criando uma variável auxiliar para adicionar as coordenadas da caixa delimitadora selecionada
bbox_select = list()

# enquanto ainda houver um frame a ser detectado
while ret:

    if id_select != 0:

        # realizando a inferência com o modelo da YOLO no frame em análise
        results = model(frame, verbose = False)

        # adicionando as informações da caixa delimitadora em uma lista
        for result in results:
            detections = list()
            # pegando as informações do modelo da YOLO
            for index, r in enumerate(result.boxes.data.tolist()):
                seg_map = result.masks.data[index].cpu().numpy()
                seg_map = segmentation_map_corrections(seg_map = seg_map, new_shape = frame.shape,
                                                    color = SEGMENTATION_COLOR)
                x1, y1, x2, y2, score, class_id = r
                [x1, y1, x2, y2] = bbox_corrections(bbox = [x1, y1, x2, y2])
                # adicionando a lista caso a detecção tenha um determinado grau de confiabilidade
                if score > DETECTION_THRESHOLD:
                    detections.append([x1, y1, x2, y2, score])
                    if index == 0: segmentations = seg_map
                    else: segmentations += seg_map

            # passando as informações das caixas delimitadoras no frame atual para o módulo de rastreamento
            tracker.update(frame, detections)

            # a partir do módulo de rastreamento, recuperar as informações das detecções para desenhar no frame
            for track in tracker.tracks:
                id_view = 0
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                [x1, y1, x2, y2] = bbox_corrections(bbox = [x1, y1, x2, y2])

                
                if id_select == track_id:
                    # desenhando a caixa delimitadora no frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_SELECTED_COLOR, 3)
                    # identificando textualmente a detecção com rastreamento feita
                    cv2.putText(frame, f'Torre | ID: {track_id}', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, BBOX_TEXT_COLOR, 2)
                    id_view += 1
                else:
                    # desenhando a caixa delimitadora no frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_UNSELECTED_COLOR, 3)
                    # identificando textualmente a detecção com rastreamento feita
                    cv2.putText(frame, f'Torre | ID: {track_id}', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, BBOX_TEXT_COLOR, 2)
                    bbox_select = [x1, y1, x2, y2]

        if id_select == -1 or id_view == 0: 
            print_id_select, id_select = 'Nenhum', -1
            COLOR_ID_TEXT = [0, 0, 255]
        else: 
            print_id_select = id_select
            COLOR_ID_TEXT = [0, 255, 0]

        cv2.putText(frame, f'ID: {print_id_select}', (5, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ID_TEXT, 2)
        
        frame = cv2.addWeighted(frame, 0.7, segmentations.astype('uint8'), 0.3, 0)

    cv2.putText(frame, 'PROJETO ESFERAS - CEMIG', (frame.shape[1] - 360, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # gravando o frame para montar o vídeo a ser salvo
    cap_out.write(frame)
    # visualizando as detecções com rastreamento feitas
    cv2.imshow('ESFERAS - CEMIG', frame)

    # aguardando uma entrada do usuário
    key = cv2.waitKey(time)

    if key == ord('c'): time = 1
    elif key == ord('s'): time = 0
    elif key == ord('i'): id_select = int(input('selecione um ID: '))
    elif key == ord('r'): id_select = -1
    elif key == ord('q'): break
    elif key == -1: pass

    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()