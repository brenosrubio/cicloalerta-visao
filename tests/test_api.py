from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

from via_atenta.api import TAMANHO_MAXIMO, criar_app
from via_atenta.detector import Deteccao


class DetectorFalso:
    nomes = {0: "buraco", 1: "rachadura"}

    def __init__(self):
        self.quadros = []

    def detectar(self, quadro):
        self.quadros.append(quadro)
        return [Deteccao(0, "buraco", 0.8123, (10.04, 20.0, 30.0, 40.0))]


@pytest.fixture
def detector():
    return DetectorFalso()


@pytest.fixture
def cliente(detector):
    return criar_app(detector).test_client()


def jpeg():
    ok, dados = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    assert ok
    return dados.tobytes()


def test_saude(cliente):
    resposta = cliente.get("/saude")
    assert resposta.status_code == 200
    assert resposta.get_json() == {"status": "ok", "classes": ["buraco", "rachadura"]}


def test_detecta_em_imagem_valida(cliente, detector):
    resposta = cliente.post("/v1/deteccoes", data={"imagem": (io.BytesIO(jpeg()), "rua.jpg")})
    assert resposta.status_code == 200
    corpo = resposta.get_json()
    assert (corpo["largura"], corpo["altura"]) == (64, 48)
    assert corpo["status"] == "pendente_revisao"
    assert corpo["deteccoes"] == [{"classe": "buraco", "confianca": 0.812, "caixa": [10.0, 20.0, 30.0, 40.0]}]
    assert detector.quadros[0].shape == (48, 64, 3)


def test_sem_arquivo(cliente):
    resposta = cliente.post("/v1/deteccoes")
    assert resposta.status_code == 400


def test_arquivo_que_nao_e_imagem(cliente):
    resposta = cliente.post("/v1/deteccoes", data={"imagem": (io.BytesIO(b"nao sou imagem"), "x.jpg")})
    assert resposta.status_code == 415


def test_arquivo_vazio(cliente):
    resposta = cliente.post("/v1/deteccoes", data={"imagem": (io.BytesIO(b""), "x.jpg")})
    assert resposta.status_code == 415


def test_imagem_grande_demais(cliente):
    resposta = cliente.post("/v1/deteccoes", data={"imagem": (io.BytesIO(b"0" * (TAMANHO_MAXIMO + 1)), "x.jpg")})
    assert resposta.status_code == 413
    assert "8 MB" in resposta.get_json()["erro"]
