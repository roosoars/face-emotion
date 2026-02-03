# Face Emotion

Sistema de deteccao de landmarks faciais 3D em tempo real utilizando MediaPipe Face Mesh.

---

## Sumario

1. [Visao Geral](#visao-geral)
2. [Arquitetura do Sistema](#arquitetura-do-sistema)
3. [Requisitos](#requisitos)
4. [Instalacao](#instalacao)
5. [Uso](#uso)
6. [Etapas do Projeto](#etapas-do-projeto)
7. [Estrutura de Arquivos](#estrutura-de-arquivos)
8. [Detalhes Tecnicos](#detalhes-tecnicos)
9. [Referencias](#referencias)
10. [Licenca](#licenca)

---

## Visao Geral

O **Face Emotion** e uma aplicacao de visao computacional que detecta e rastreia 478 pontos de referencia (landmarks) no rosto humano em tempo real. Utiliza a biblioteca MediaPipe Face Mesh do Google para processamento de machine learning e OpenCV para captura e exibicao de video.

### Funcionalidades Principais

- Deteccao de 468 landmarks faciais base
- Deteccao de 10 landmarks adicionais para iris (478 total)
- Rastreamento em tempo real via webcam
- Visualizacao de contornos faciais com cores distintas
- Bounding box dinamica ao redor do rosto detectado
- Suporte a deteccao com oculos
- Processamento otimizado para baixa latencia

---

## Arquitetura do Sistema

```
+------------------+     +-------------------+     +------------------+
|                  |     |                   |     |                  |
|  Webcam/Camera   +---->+  MediaPipe Face   +---->+  Renderizacao    |
|  (OpenCV)        |     |  Mesh (ML Model)  |     |  (OpenCV)        |
|                  |     |                   |     |                  |
+------------------+     +-------------------+     +------------------+
        |                         |                        |
        v                         v                        v
   Captura de              Deteccao de 478          Desenho de pontos,
   frames BGR              landmarks 3D             contornos e bbox
```

### Fluxo de Processamento

1. Captura do frame da webcam (BGR)
2. Conversao de BGR para RGB
3. Processamento pelo modelo Face Mesh
4. Extracao das coordenadas dos 478 landmarks
5. Conversao de coordenadas normalizadas para pixels
6. Desenho dos pontos e contornos no frame
7. Exibicao do resultado

---

## Requisitos

### Hardware

- Computador com webcam (testado em MacBook com FaceTime HD Camera)
- Processador com suporte a instrucoes SSE4.1 ou superior
- Minimo 4GB de RAM

### Software

- Python 3.8 ou superior
- Sistema operacional: macOS, Linux ou Windows

### Dependencias Python

| Pacote | Versao | Descricao |
|--------|--------|-----------|
| mediapipe | 0.10.9 | Framework de ML para deteccao facial |
| opencv-python | 4.8.0+ | Processamento de imagens e video |
| numpy | 1.24.0+ | Operacoes numericas |

---

## Instalacao

### 1. Clonar o Repositorio

```bash
git clone https://github.com/roosoars/face-emotion.git
cd face-emotion
```

### 2. Criar Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate     # Windows
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

---

## Uso

### Executar a Aplicacao

```bash
python main.py
```

### Controles

| Tecla | Acao |
|-------|------|
| ESC | Encerra a aplicacao |

### Saida Esperada

A aplicacao abre uma janela mostrando:

- Video da webcam em tempo real
- 478 pontos vermelhos nos landmarks faciais
- Linhas verdes conectando os contornos (olhos, boca, nariz, sobrancelhas)
- Retangulo vermelho ao redor do rosto (bounding box)

---

## Etapas do Projeto

### Etapa 1: Deteccao Basica de Landmarks (Atual)

**Status:** Concluida

**Objetivo:** Implementar a deteccao basica dos 478 landmarks faciais em tempo real.

**Funcionalidades:**
- Captura de video da webcam
- Deteccao de rosto usando MediaPipe Face Mesh
- Extracao de 468 landmarks faciais + 10 landmarks de iris
- Visualizacao dos pontos e contornos
- Bounding box ao redor do rosto

**Regioes Mapeadas:**
- Olho esquerdo (16 pontos)
- Olho direito (16 pontos)
- Sobrancelha esquerda (10 pontos)
- Sobrancelha direita (10 pontos)
- Labios externos (20 pontos)
- Labios internos (20 pontos)
- Contorno do rosto (36 pontos)
- Nariz (10 pontos)
- Iris esquerda (5 pontos)
- Iris direita (5 pontos)

### Etapa 2: Analise de Expressoes Faciais (Planejada)

**Status:** Planejada

**Objetivo:** Implementar algoritmos para detectar expressoes faciais baseadas nos landmarks.

**Funcionalidades Previstas:**
- Calculo de distancias entre landmarks chave
- Deteccao de abertura dos olhos (Eye Aspect Ratio)
- Deteccao de abertura da boca (Mouth Aspect Ratio)
- Classificacao de expressoes basicas (feliz, triste, surpreso, neutro)

### Etapa 3: Deteccao de Emocoes (Planejada)

**Status:** Planejada

**Objetivo:** Classificar emocoes em tempo real usando os dados dos landmarks.

**Funcionalidades Previstas:**
- Modelo de classificacao de emocoes
- Interface com indicadores de emocao
- Historico de emocoes detectadas
- Exportacao de dados

---

## Estrutura de Arquivos

```
face-emotion/
|
|-- main.py              # Aplicacao principal
|-- requirements.txt     # Dependencias do projeto
|-- README.md            # Documentacao do projeto
|-- venv/                # Ambiente virtual Python (nao versionado)
|
|-- Face Emotion/        # Projeto Xcode (opcional, para integracao iOS)
    |-- ContentView.swift
    |-- FaceEmotionApp.swift
    |-- Assets.xcassets/
```

---

## Detalhes Tecnicos

### MediaPipe Face Mesh

O MediaPipe Face Mesh e uma solucao de machine learning que estima 468 landmarks faciais 3D em tempo real. Com a opcao `refine_landmarks=True`, adiciona 10 landmarks para as iris, totalizando 478 pontos.

**Caracteristicas:**
- Inferencia em tempo real (30+ FPS em hardware moderno)
- Coordenadas 3D normalizadas (x, y, z)
- Robusto a diferentes angulos e iluminacao
- Suporte a multiplos rostos

### Indices dos Landmarks

Os landmarks sao identificados por indices fixos definidos pelo modelo. Abaixo estao os principais grupos:

**Olhos:**
- Olho esquerdo: indices 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
- Olho direito: indices 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398

**Sobrancelhas:**
- Esquerda: indices 70, 63, 105, 66, 107, 55, 65, 52, 53, 46
- Direita: indices 300, 293, 334, 296, 336, 285, 295, 282, 283, 276

**Labios:**
- Externos: indices 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185
- Internos: indices 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191

**Nariz:**
- Indices: 168, 6, 197, 195, 5, 4, 1, 19, 94, 2

**Iris:**
- Esquerda: indices 468, 469, 470, 471, 472
- Direita: indices 473, 474, 475, 476, 477

### Parametros de Configuracao

| Parametro | Valor | Descricao |
|-----------|-------|-----------|
| max_num_faces | 1 | Numero maximo de rostos a detectar |
| refine_landmarks | True | Ativa deteccao de iris |
| min_detection_confidence | 0.5 | Confianca minima para deteccao |
| min_tracking_confidence | 0.5 | Confianca minima para rastreamento |

---

## Referencias

### Documentacao Oficial

- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [MediaPipe GitHub](https://github.com/google/mediapipe)

### Artigos Academicos

- Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019). Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs. arXiv preprint arXiv:1907.06724.

### Recursos Adicionais

- [MediaPipe Face Mesh - Mapa de Landmarks](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)

---

## Licenca

Este projeto esta licenciado sob a licenca MIT.

```
MIT License

Copyright (c) 2026 Rodrigo Soares

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Autor

**Rodrigo Soares**

- GitHub: [@roosoars](https://github.com/roosoars)

---

## Historico de Versoes

| Versao | Data | Descricao |
|--------|------|-----------|
| 1.0.0 | 2026-02-03 | Versao inicial com deteccao basica de landmarks |
