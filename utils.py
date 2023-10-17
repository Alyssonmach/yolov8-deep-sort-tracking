import cv2
import numpy as np

def select_segmentation(mask_total, bbox):
    '''
    Seleciona apenas a segmentação da caixa delimitadora selecionada.

    Args:
        mask_total (array) --> máscara de segmentação com todos os objetos selecionados.
        bboc (list) --> coordenadas da caixa delimitadora no formato [x1, y1, x2, y2].
    
    Returns:
        mask (array) -->   máscara de segmentação com apenas o objeto de análise segmentado. 

    '''
    
    # extraindo as coordenadas da caixa delimitadora
    x1, y1, x2, y2 = bbox
    # criando uma máscara com valores zerados para adicionar apenas a área de interesse
    mask_selected = np.zeros(mask_total.shape, dtype = np.uint8)
    # realizando o recorte da máscara de segmentação na área selecionada
    mask_selected[y1:y2, x1:x2] = mask_total[y1:y2, x1:x2]

    return mask_selected

def segmentation_map_corrections(seg_map, new_shape, color):

    seg_map =  np.expand_dims(seg_map, 0).repeat(3, axis = 0)
    seg_map = np.moveaxis(seg_map, 0, -1)
    seg_map = cv2.resize(seg_map, [new_shape[1], new_shape[0]])

    seg_map[:, :, 0] *= color[0]
    seg_map[:, :, 1] *= color[1]
    seg_map[:, :, 2] *= color[2]

    return seg_map

def bbox_corrections(bbox):

    # extraindo as coordenadas da caixa delimitadora
    x1, y1, x2, y2 = bbox

    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)

    return [x1, y1, x2, y2]
