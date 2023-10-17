# importação dos pacotes utilitários
import os
import cv2
import random
from tracker import Tracker
from ultralytics import YOLO
from utils import bbox_corrections
from utils import segmentation_map_corrections

# arrays para definir a cor da segmentação no padrão RGB
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

# carregando os parâmetros pré-treinados nas arquiteturas da YOLOv8 (com e sem segmentação)
model_seg = YOLO("modelos-salvos/yolov8n-seg.pt")
model_without_seg = YOLO("modelos-salvos/yolov8n.pt")

# inicializando o módulo de rastreamento do deep-sort
tracker = Tracker()

# variável auxiliar para adicionar o tempo de espera entre um frame e outro
time = 1
# variável auxiliar para adicionar o ID selecionado pelo usuário 
id_select = 0
# criando uma variável auxiliar para adicionar as coordenadas da caixa delimitadora selecionada
bbox_select = list()
# variável auxiliar para armazenar o ID de rastreamento durante a seleção
riding_id = ''
# variável auxiliar de estado para quando o usuário estiver digitando um ID
select_mode = False
# variável auxiliar para definir quando ativar ou desativar o modelo de segmentação
verify_seg = 1
# variável auxiliar para mostrar o quadro de ajuda com as teclas
help_select = 1

# enquanto ainda houver um frame a ser detectado
while ret:

    # caso o programa não esteja inicializado, não realizar inferências no frame
    if id_select != 0:
        id_view = 0
        # realizando a inferência com o modelo da YOLO no frame em análise
        if verify_seg % 2:
            results = model_seg(frame, verbose = False)
        else: 
            results = model_without_seg(frame, verbose = False)

        # adicionando as informações da caixa delimitadora em uma lista
        for result in results:
            detections = list()
            # pegando as informações do modelo da YOLO
            for index, r in enumerate(result.boxes.data.tolist()):
                if verify_seg % 2:
                    seg_map = result.masks.data[index].cpu().numpy()
                    seg_map = segmentation_map_corrections(seg_map = seg_map, new_shape = frame.shape,
                                                           color = SEGMENTATION_COLOR)
                x1, y1, x2, y2, score, class_id = r
                [x1, y1, x2, y2] = bbox_corrections(bbox = [x1, y1, x2, y2])
                # adicionando a lista caso a detecção tenha um determinado grau de confiabilidade
                if score > DETECTION_THRESHOLD:
                    detections.append([x1, y1, x2, y2, score])
                    if verify_seg % 2:
                        if index == 0: segmentations = seg_map
                        else: segmentations += seg_map

            # passando as informações das caixas delimitadoras no frame atual para o módulo de rastreamento
            tracker.update(frame, detections)

            # a partir do módulo de rastreamento, recuperar as informações das detecções para desenhar no frame
            for track in tracker.tracks:
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

        # atualiza o estado na tela do ID selecionado
        if id_select == -1: 
            print_id_select, id_select = 'Nenhum', -1
            COLOR_ID_TEXT = [0, 0, 255]
        elif id_view == 0: 
            print_id_select = f'ID {id_select} fora de alcance'
            COLOR_ID_TEXT = [0, 0, 255]
        else: 
            print_id_select = id_select
            COLOR_ID_TEXT = [0, 255, 0]

        cv2.putText(frame, f'ID: {print_id_select}', (5, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ID_TEXT, 2)
        
        # coloca a máscara de segmentação caso o modelo de segmentação estiver selecionado
        if verify_seg % 2:
            frame = cv2.addWeighted(frame, 0.7, segmentations.astype('uint8'), 0.3, 0)

    # título do projeto informado no canto da tela
    cv2.putText(frame, 'PROJETO ESFERAS - CEMIG', (frame.shape[1] - 230, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # mostrar botão de ajuda na tela, com o manual de teclas
    if help_select % 2:
        cv2.putText(frame, '[h] -> AJUDA', (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        cv2.putText(frame, 'MENU DE TECLAS:', (5, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, '[s -> stop] Pausa o frame na tela', (5, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, '[c --> continue] Despausa o frame na tela', (5, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, '[i -> ID] Modelo de escolha do ID', (5, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, '[r -> reset] Reseta o ID selecionado', (5, 230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, '[m -> mask] Hab./Des. o segmento', (5, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, '[q -> quit] Fecha o programa', (5, 310), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, '[h] -> VOLTAR', (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # gravando o frame para montar o vídeo a ser salvo
    cap_out.write(frame)
    # visualizando as detecções com rastreamento feitas
    cv2.imshow('PROJETO ESFERAS - CEMIG', frame)

    # aguardando uma entrada do usuário
    key = cv2.waitKey(time)

    # caso o modelo de seleção esteja ativo, esperar o usuário digitar um ID
    if select_mode == True:
        if key != -1:
            riding_id += chr(key)
        if len(riding_id) == 2:
            try: id_select = int(riding_id)
            except: pass

            riding_id = ''
            select_mode = False

    # teclas de controlabilidade
    # despausa o frame atual na tela [c --> continue]
    if key == ord('c'): time = 1
    # pausa o frame atual na tela  [s --> stop]
    elif key == ord('s'): time = 0
    # ativa o modo de digitar o identificados [i --> ID]
    elif key == ord('i'): select_mode = True
    # reseta o ID selecionado [r --> reset]
    elif key == ord('r'): id_select = -1
    # habilita/desabilita a máscara de segmentação [m --> mask]
    elif key == ord('m'): verify_seg += 1
    # finaliza o programa [q --> quit]
    elif key == ord('q'): break
    # caso nenhuma tecla seja digitada, a chave sempre será -1
    elif key == ord('h'): help_select += 1
    elif key == -1: pass

    # captura o próxima frame do vídeo
    ret, frame = cap.read()

# fechando todo o program de forma adequada
cap.release()
cap_out.release()
cv2.destroyAllWindows()