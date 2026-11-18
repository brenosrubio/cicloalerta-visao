from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

from cicloalerta.api import TAMANHO_MAXIMO, analisar_imagem, criar_app, severidade_sugerida
from cicloalerta.detector import Deteccao


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
    assert corpo["tem_defeito"] is True
    assert corpo["deteccoes"] == [
        {"classe": "buraco", "confianca": 0.812, "caixa": [10.0, 20.0, 30.0, 40.0], "severidade_sugerida": "alta"}
    ]
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


def test_rota_do_contrato_do_mvp_aceita_campo_foto(cliente):
    resposta = cliente.post("/api/detectar", data={"foto": (io.BytesIO(jpeg()), "buraco.jpg")})
    assert resposta.status_code == 200
    assert resposta.get_json()["deteccoes"][0]["classe"] == "buraco"


def test_rota_do_contrato_sem_arquivo(cliente):
    assert cliente.post("/api/detectar").status_code == 400


def test_analisar_imagem_sem_flask(detector):
    corpo, codigo = analisar_imagem(jpeg(), detector)
    assert codigo == 200 and corpo["status"] == "pendente_revisao"
    corpo, codigo = analisar_imagem(b"nao sou imagem", detector)
    assert codigo == 415


def test_foto_sem_defeito():
    class Vazio(DetectorFalso):
        def detectar(self, quadro):
            return []

    corpo, codigo = analisar_imagem(jpeg(), Vazio())
    assert codigo == 200 and corpo["tem_defeito"] is False and corpo["deteccoes"] == []


def test_severidade_sugerida_pelo_tamanho_da_caixa():
    assert severidade_sugerida("buraco", 0.10) == "alta"
    assert severidade_sugerida("buraco", 0.01) == "media"
    assert severidade_sugerida("rachadura", 0.20) == "media"
    assert severidade_sugerida("rachadura", 0.02) == "baixa"
