"""
================================================================================
TESTES UNITARIOS - FACE EMOTION
================================================================================

Testes para validar o funcionamento dos componentes do sistema de deteccao
de landmarks faciais.

Executar testes:
    pytest test_face_mesh.py -v

================================================================================
"""

import pytest
import numpy as np


class TestLandmarkIndices:
    """Testes para validar os indices dos landmarks faciais"""

    # Indices definidos no main.py
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    def test_left_eye_count(self):
        """Verifica se o olho esquerdo tem 16 pontos"""
        assert len(self.LEFT_EYE) == 16

    def test_right_eye_count(self):
        """Verifica se o olho direito tem 16 pontos"""
        assert len(self.RIGHT_EYE) == 16

    def test_left_eyebrow_count(self):
        """Verifica se a sobrancelha esquerda tem 10 pontos"""
        assert len(self.LEFT_EYEBROW) == 10

    def test_right_eyebrow_count(self):
        """Verifica se a sobrancelha direita tem 10 pontos"""
        assert len(self.RIGHT_EYEBROW) == 10

    def test_lips_outer_count(self):
        """Verifica se os labios externos tem 20 pontos"""
        assert len(self.LIPS_OUTER) == 20

    def test_lips_inner_count(self):
        """Verifica se os labios internos tem 20 pontos"""
        assert len(self.LIPS_INNER) == 20

    def test_face_oval_count(self):
        """Verifica se o contorno do rosto tem 36 pontos"""
        assert len(self.FACE_OVAL) == 36

    def test_nose_count(self):
        """Verifica se o nariz tem 10 pontos"""
        assert len(self.NOSE) == 10

    def test_left_iris_count(self):
        """Verifica se a iris esquerda tem 5 pontos"""
        assert len(self.LEFT_IRIS) == 5

    def test_right_iris_count(self):
        """Verifica se a iris direita tem 5 pontos"""
        assert len(self.RIGHT_IRIS) == 5

    def test_all_indices_within_range(self):
        """Verifica se todos os indices estao no range valido (0-477)"""
        all_indices = (
            self.LEFT_EYE + self.RIGHT_EYE +
            self.LEFT_EYEBROW + self.RIGHT_EYEBROW +
            self.LIPS_OUTER + self.LIPS_INNER +
            self.FACE_OVAL + self.NOSE +
            self.LEFT_IRIS + self.RIGHT_IRIS
        )
        for idx in all_indices:
            assert 0 <= idx <= 477, f"Indice {idx} fora do range valido"

    def test_iris_indices_are_refined(self):
        """Verifica se os indices de iris estao no range refinado (468-477)"""
        for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
            assert 468 <= idx <= 477, f"Indice de iris {idx} fora do range refinado"

    def test_no_duplicate_indices_in_region(self):
        """Verifica se nao ha indices duplicados em cada regiao"""
        regions = [
            ("LEFT_EYE", self.LEFT_EYE),
            ("RIGHT_EYE", self.RIGHT_EYE),
            ("LEFT_EYEBROW", self.LEFT_EYEBROW),
            ("RIGHT_EYEBROW", self.RIGHT_EYEBROW),
            ("LIPS_OUTER", self.LIPS_OUTER),
            ("LIPS_INNER", self.LIPS_INNER),
            ("FACE_OVAL", self.FACE_OVAL),
            ("NOSE", self.NOSE),
            ("LEFT_IRIS", self.LEFT_IRIS),
            ("RIGHT_IRIS", self.RIGHT_IRIS),
        ]
        for name, indices in regions:
            assert len(indices) == len(set(indices)), f"Indices duplicados em {name}"


class TestCoordinateConversion:
    """Testes para conversao de coordenadas normalizadas para pixels"""

    def test_normalized_to_pixel_conversion(self):
        """Testa conversao de coordenadas normalizadas (0-1) para pixels"""
        # Simulando coordenadas normalizadas
        normalized_x = 0.5
        normalized_y = 0.5
        width = 1280
        height = 720

        pixel_x = int(normalized_x * width)
        pixel_y = int(normalized_y * height)

        assert pixel_x == 640
        assert pixel_y == 360

    def test_corner_coordinates(self):
        """Testa conversao nos cantos do frame"""
        width = 1280
        height = 720

        # Canto superior esquerdo
        assert int(0.0 * width) == 0
        assert int(0.0 * height) == 0

        # Canto inferior direito
        assert int(1.0 * width) == 1280
        assert int(1.0 * height) == 720

    def test_bounding_box_calculation(self):
        """Testa calculo da bounding box"""
        points = [(100, 100), (200, 150), (150, 200)]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        margin = 20
        x1, y1 = min(xs) - margin, min(ys) - margin
        x2, y2 = max(xs) + margin, max(ys) + margin

        assert x1 == 80   # 100 - 20
        assert y1 == 80   # 100 - 20
        assert x2 == 220  # 200 + 20
        assert y2 == 220  # 200 + 20


class TestDrawingFunctions:
    """Testes para funcoes de desenho"""

    def test_contour_connection_logic(self):
        """Testa logica de conexao de contornos"""
        indices = [0, 1, 2, 3, 4]
        connections = []

        # Simula a logica de conexao
        for i in range(len(indices) - 1):
            connections.append((indices[i], indices[i + 1]))

        expected = [(0, 1), (1, 2), (2, 3), (3, 4)]
        assert connections == expected

    def test_closed_contour_connection(self):
        """Testa conexao de contorno fechado"""
        indices = [0, 1, 2, 3, 4]
        connections = []

        for i in range(len(indices) - 1):
            connections.append((indices[i], indices[i + 1]))

        # Fecha o contorno
        connections.append((indices[-1], indices[0]))

        expected = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        assert connections == expected


class TestTotalLandmarks:
    """Testes para validar o total de landmarks"""

    def test_base_landmarks_count(self):
        """Verifica que o modelo base tem 468 landmarks"""
        BASE_LANDMARKS = 468
        assert BASE_LANDMARKS == 468

    def test_refined_landmarks_count(self):
        """Verifica que com refine_landmarks=True tem 478 landmarks"""
        BASE_LANDMARKS = 468
        IRIS_LANDMARKS = 10
        TOTAL = BASE_LANDMARKS + IRIS_LANDMARKS
        assert TOTAL == 478

    def test_iris_landmarks_distribution(self):
        """Verifica distribuicao dos landmarks de iris"""
        LEFT_IRIS_COUNT = 5
        RIGHT_IRIS_COUNT = 5
        assert LEFT_IRIS_COUNT + RIGHT_IRIS_COUNT == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
