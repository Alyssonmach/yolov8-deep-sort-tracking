# algoritmo de inferência com aplicação do deep-sort para a YOLOv8

# importação dos pacotes utilitários
import cv2
import numpy as np
import os, random
from tracker import Tracker
from ultralytics import YOLO

def recortar_imagem_pela_caixa(imagem, caixa):
    # Certifique-se de que a caixa esteja no formato [x1, y1, x2, y2]
    x1, y1, x2, y2 = caixa

    # Crie uma máscara de zeros com as mesmas dimensões da imagem
    mascara = np.zeros(imagem.shape, dtype=np.uint8)

    # Defina a região dentro da caixa delimitadora como a imagem na máscara
    mascara[y1:y2, x1:x2] = imagem[y1:y2, x1:x2]

    return mascara

time = 0
id_select = 0
bbox_select = list()

# vídeo para realização da inferência
video_path = os.path.join('.', 'data', 'people.mp4')
# local onde os vídeos com as inferências serão salvos
video_out_path = os.path.join('.', 'videos-teste', 'out.mp4')

# inicializando o módulo de captura de vídeo do opencv (0 para acessar a câmera)
cap = cv2.VideoCapture(video_path)
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

# definindo um limiar de confiabilidade na predição feita pelo classificador da YOLO
detection_threshold = 0.5

# enquanto ainda houver um frame a ser detectado
while ret:

    # realizando a inferência com o modelo da YOLO no frame em análise
    results = model(frame, verbose = False)

    # adicionando as informações da caixa delimitadora em uma lista
    for result in results:
        detections = list()
        # pegando as informações do modelo da YOLO
        for index, r in enumerate(result.boxes.data.tolist()):
            seg_map = result.masks.data[index].cpu().numpy()
            seg_map =  np.expand_dims(seg_map, 0).repeat(3, axis = 0)
            seg_map = np.moveaxis(seg_map, 0, -1)
            seg_map = cv2.resize(seg_map, [frame.shape[1], frame.shape[0]])
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            # adicionando a lista caso a detecção tenha um determinado grau de confiabilidade
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])
                if index == 0: segmentations = seg_map
                else: segmentations += seg_map

        # passando as informações das caixas delimitadoras no frame atual para o módulo de rastreamento
        tracker.update(frame, detections)

        # a partir do módulo de rastreamento, recuperar as informações das detecções para desenhar no frame
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

            if id_select == 0:
                # desenhando a caixa delimitadora no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (colors[track_id % len(colors)]), 3)
                # identificando textualmente a detecção com rastreamento feita
                cv2.putText(frame, f'Torre | ID: {track_id}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            elif id_select == track_id:
                bbox_select = [x1, y1, x2, y2]
                # desenhando a caixa delimitadora no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (colors[track_id % len(colors)]), 3)
                # identificando textualmente a detecção com rastreamento feita
                cv2.putText(frame, f'Torre | ID: {track_id}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    segmentations *= 255
    if id_select != 0: segmentations = recortar_imagem_pela_caixa(segmentations, bbox_select)
    frame = cv2.addWeighted(frame, 0.7, segmentations.astype('uint8'), 0.3, 0)
    # gravando o frame para montar o vídeo a ser salvo
    cap_out.write(frame)
    # visualizando as detecções com rastreamento feitas
    cv2.imshow('frame', frame)

    # aguardando uma entrada do usuário
    key = cv2.waitKey(time)

    if key == ord('c'): time = 1
    elif key == ord('s'): time = 0
    elif key == ord('i'): id_select = int(input('selecione um ID: '))
    elif key == ord('r'): id_select = 0
    elif key == ord('q'): break
    elif key == -1: pass

    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()