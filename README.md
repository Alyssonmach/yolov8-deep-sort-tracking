# YOLOv8 + Deep SORT: detecção e rastreamento

Aplicação em Python que combina modelos YOLOv8 para detectar pessoas em vídeo com o algoritmo Deep SORT para manter um identificador por objeto. O modo padrão usa segmentação; a tecla `m` alterna para detecção sem máscara.

## Requisitos

- Uma versão do Python compatível com as versões instaladas de TensorFlow e Ultralytics.
- OpenCV, NumPy e Ultralytics.
- TensorFlow compatível com o encoder Deep SORT usado pelo projeto.
- Código do Deep SORT disponível na pasta `deep-sort/`, com os diretórios `deep_sort/` e `tools/`.
- O encoder de aparência `model_data/mars-small128.pb`, incluído neste repositório.

Crie e ative um ambiente virtual e instale as dependências Python:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install opencv-python numpy ultralytics tensorflow
```

O código do Deep SORT não está incluído neste repositório. Obtenha uma cópia compatível com o encoder MARS e coloque-a em `deep-sort/`. Como o código importa o pacote `deep_sort` desse diretório, execute com ele no caminho de importação:

```bash
PYTHONPATH=deep-sort python deep_sort_inference.py
```

No Windows PowerShell, use:

```powershell
$env:PYTHONPATH = "deep-sort"
python deep_sort_inference.py
```

## Execução

O comando padrão processa `data/people.mp4` e grava o resultado em `output/tracked.mp4`:

```bash
PYTHONPATH=deep-sort python deep_sort_inference.py
```

Para escolher outro vídeo, arquivo de saída ou pesos YOLO:

```bash
PYTHONPATH=deep-sort python deep_sort_inference.py \
  --source caminho/entrada.mp4 \
  --output output/resultado.mp4 \
  --seg-model yolov8n-seg.pt \
  --detect-model yolov8n.pt
```

Também é possível usar uma câmera pelo índice, por exemplo `--source 0`. Se os pesos YOLO não estiverem no caminho informado, o Ultralytics pode baixá-los automaticamente na primeira execução.

## Controles

- `i`: informa o ID de rastreamento desejado (digite dois caracteres numéricos).
- `r`: limpa a seleção e reinicia o rastreador.
- `m`: alterna entre o modelo com segmentação e o modelo sem segmentação.
- `s` / `c`: pausa / continua a reprodução.
- `h`: mostra ou esconde a ajuda na tela.
- `q`: encerra a aplicação.

## Estrutura relevante

```text
.
├── data/people.mp4                 # vídeo de exemplo
├── model_data/mars-small128.pb     # encoder de aparência do Deep SORT
├── deep_sort_inference.py          # aplicação de inferência
├── tracker.py                      # integração com o Deep SORT
└── utils.py                        # utilitários de caixas e máscaras
```

Os arquivos gerados em `output/` não são versionados. O projeto está sob a licença MIT; veja [LICENSE](LICENSE).
