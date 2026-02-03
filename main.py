"""
================================================================================
FACE MESH - DETECCAO DE LANDMARKS FACIAIS 3D EM TEMPO REAL
================================================================================

Aplicacao que utiliza MediaPipe Face Mesh para detectar e rastrear 478 pontos
de referencia (landmarks) no rosto humano em tempo real.

FUNCIONALIDADES:
- Deteccao de 468 landmarks faciais + 10 landmarks de iris (478 total)
- Rastreamento em tempo real via webcam
- Visualizacao de contornos faciais (olhos, boca, nariz, sobrancelhas)
- Bounding box ao redor do rosto detectado

REGIOES MAPEADAS:
- Olhos (esquerdo e direito)
- Sobrancelhas (esquerda e direita)
- Labios (contorno externo e interno)
- Nariz
- Contorno do rosto (face oval)
- Iris (esquerda e direita)

CONTROLES:
- ESC: Encerra a aplicacao

DEPENDENCIAS:
- opencv-python (cv2)
- mediapipe
- numpy

================================================================================
"""

import cv2          # OpenCV - biblioteca para processamento de imagens e video
import mediapipe as mp  # MediaPipe - framework de ML para deteccao facial
import numpy as np  # NumPy - operacoes numericas (importado para uso futuro)

# ============================================================================
# INICIALIZACAO DO MEDIAPIPE FACE MESH
# ============================================================================

# Carrega o modulo Face Mesh do MediaPipe
# Este modulo contem o modelo de ML treinado para detectar landmarks faciais
mp_face_mesh = mp.solutions.face_mesh

# ============================================================================
# CONFIGURACAO DA CAMERA
# ============================================================================

# Inicializa a captura de video da webcam
# Parametro 0 = camera padrao do sistema (FaceTime HD Camera no MacBook)
cap = cv2.VideoCapture(0)

# ============================================================================
# DEFINICAO DOS INDICES DOS LANDMARKS PARA CADA REGIAO FACIAL
# ============================================================================
# O MediaPipe Face Mesh retorna 478 pontos (468 base + 10 iris)
# Cada ponto tem um indice especifico que corresponde a uma posicao no rosto
# Estes indices foram definidos pelo Google e sao fixos para o modelo

# OLHO ESQUERDO - 16 pontos que formam o contorno do olho esquerdo
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# OLHO DIREITO - 16 pontos que formam o contorno do olho direito
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# SOBRANCELHA ESQUERDA - 10 pontos ao longo da sobrancelha esquerda
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

# SOBRANCELHA DIREITA - 10 pontos ao longo da sobrancelha direita
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# LABIOS EXTERNOS - 20 pontos que formam o contorno externo da boca
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# LABIOS INTERNOS - 20 pontos que formam o contorno interno da boca
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# CONTORNO DO ROSTO - 36 pontos que formam o oval do rosto
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# NARIZ - 10 pontos ao longo do nariz (ponte e ponta)
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]

# IRIS ESQUERDA - 5 pontos que formam o circulo da iris esquerda
# Indices 468-472 (requer refine_landmarks=True)
LEFT_IRIS = [468, 469, 470, 471, 472]

# IRIS DIREITA - 5 pontos que formam o circulo da iris direita
# Indices 473-477 (requer refine_landmarks=True)
RIGHT_IRIS = [473, 474, 475, 476, 477]

# ============================================================================
# FUNCAO AUXILIAR: DESENHAR CONTORNO
# ============================================================================

def desenhar_contorno(frame, points, indices, fechar=True):
    """
    Desenha linhas verdes conectando os pontos de uma regiao facial.

    Parametros:
        frame: Imagem onde desenhar (numpy array BGR)
        points: Lista de todas as coordenadas (x, y) dos 478 landmarks
        indices: Lista de indices dos pontos a conectar
        fechar: Se True, conecta o ultimo ponto ao primeiro (forma fechada)

    Retorno:
        None (modifica o frame diretamente)
    """
    # Conecta cada ponto ao proximo na sequencia
    for i in range(len(indices) - 1):
        ponto_atual = points[indices[i]]
        ponto_proximo = points[indices[i + 1]]
        cv2.line(frame, ponto_atual, ponto_proximo, (0, 255, 0), 1)

    # Se fechar=True, conecta o ultimo ponto ao primeiro
    if fechar and len(indices) > 2:
        cv2.line(frame, points[indices[-1]], points[indices[0]], (0, 255, 0), 1)

# ============================================================================
# LOOP PRINCIPAL - PROCESSAMENTO DE VIDEO EM TEMPO REAL
# ============================================================================

# Inicializa o Face Mesh com configuracoes otimizadas
# Usando 'with' para garantir que os recursos sejam liberados corretamente
with mp_face_mesh.FaceMesh(
    max_num_faces=1,              # Detecta apenas 1 rosto (mais rapido)
    refine_landmarks=True,        # Ativa deteccao de iris (478 pontos ao inves de 468)
    min_detection_confidence=0.5, # Confianca minima para detectar um rosto (0.0 a 1.0)
    min_tracking_confidence=0.5   # Confianca minima para rastrear o rosto entre frames
) as face_mesh:

    # Loop infinito ate o usuario pressionar ESC
    while cap.isOpened():

        # ====================================================================
        # CAPTURA DO FRAME
        # ====================================================================

        # Le um frame da webcam
        # ret: booleano indicando se a leitura foi bem sucedida
        # frame: imagem capturada (numpy array no formato BGR)
        ret, frame = cap.read()

        # Se nao conseguiu ler o frame, tenta novamente
        if not ret:
            continue

        # ====================================================================
        # PRE-PROCESSAMENTO
        # ====================================================================

        # Espelha o frame horizontalmente para efeito de espelho
        # Isso faz com que os movimentos na tela correspondam aos do usuario
        frame = cv2.flip(frame, 1)

        # Obtem as dimensoes do frame
        # h = altura, w = largura, _ = canais de cor (ignorado)
        h, w, _ = frame.shape

        # Converte de BGR (padrao OpenCV) para RGB (padrao MediaPipe)
        # O MediaPipe foi treinado com imagens RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ====================================================================
        # DETECCAO DE LANDMARKS
        # ====================================================================

        # Processa o frame e detecta os landmarks faciais
        # Retorna um objeto com multi_face_landmarks contendo os pontos detectados
        results = face_mesh.process(rgb)

        # ====================================================================
        # DESENHO DOS LANDMARKS (se um rosto foi detectado)
        # ====================================================================

        # Verifica se algum rosto foi detectado
        if results.multi_face_landmarks:

            # Itera sobre cada rosto detectado (neste caso, apenas 1)
            for landmarks in results.multi_face_landmarks:

                # Lista para armazenar as coordenadas (x, y) de cada ponto
                points = []

                # ============================================================
                # DESENHA OS 478 PONTOS VERMELHOS
                # ============================================================

                # Itera sobre cada um dos 478 landmarks
                for lm in landmarks.landmark:
                    # Converte coordenadas normalizadas (0-1) para pixels
                    # lm.x e lm.y estao normalizados entre 0 e 1
                    x = int(lm.x * w)  # Multiplica pela largura do frame
                    y = int(lm.y * h)  # Multiplica pela altura do frame

                    # Adiciona o ponto a lista
                    points.append((x, y))

                    # Desenha um circulo vermelho no ponto
                    # Parametros: imagem, centro, raio, cor BGR, espessura (-1 = preenchido)
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                # ============================================================
                # DESENHA A BOUNDING BOX VERMELHA
                # ============================================================

                # Extrai todas as coordenadas X e Y
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                # Calcula os limites da caixa com margem de 20 pixels
                x1, y1 = min(xs) - 20, min(ys) - 20  # Canto superior esquerdo
                x2, y2 = max(xs) + 20, max(ys) + 20  # Canto inferior direito

                # Desenha o retangulo vermelho
                # Parametros: imagem, ponto1, ponto2, cor BGR, espessura
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # ============================================================
                # DESENHA OS CONTORNOS VERDES
                # ============================================================

                # Olhos - contornos fechados
                desenhar_contorno(frame, points, LEFT_EYE, fechar=True)
                desenhar_contorno(frame, points, RIGHT_EYE, fechar=True)

                # Sobrancelhas - contornos abertos (nao fecha o loop)
                desenhar_contorno(frame, points, LEFT_EYEBROW, fechar=False)
                desenhar_contorno(frame, points, RIGHT_EYEBROW, fechar=False)

                # Labios - contornos fechados
                desenhar_contorno(frame, points, LIPS_OUTER, fechar=True)
                desenhar_contorno(frame, points, LIPS_INNER, fechar=True)

                # Contorno do rosto - contorno fechado
                desenhar_contorno(frame, points, FACE_OVAL, fechar=True)

                # Nariz - contorno aberto
                desenhar_contorno(frame, points, NOSE, fechar=False)

                # Iris - contornos fechados
                desenhar_contorno(frame, points, LEFT_IRIS, fechar=True)
                desenhar_contorno(frame, points, RIGHT_IRIS, fechar=True)

        # ====================================================================
        # EXIBICAO DO RESULTADO
        # ====================================================================

        # Mostra o frame processado em uma janela
        # 'Face Mesh' e o titulo da janela
        cv2.imshow('Face Mesh', frame)

        # ====================================================================
        # CONTROLE DE SAIDA
        # ====================================================================

        # Aguarda 1 milissegundo por uma tecla
        # waitKey retorna o codigo ASCII da tecla pressionada
        # & 0xFF garante compatibilidade entre sistemas
        # 27 = codigo ASCII da tecla ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

# ============================================================================
# LIMPEZA DE RECURSOS
# ============================================================================

# Libera a camera
cap.release()

# Fecha todas as janelas do OpenCV
cv2.destroyAllWindows()
