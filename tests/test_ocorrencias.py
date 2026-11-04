from __future__ import annotations

import json

import cv2
import numpy as np

from via_atenta.ocorrencias import RegistroOcorrencias, recortar
from via_atenta.rastreador import Rastro


def rastro(caixa=(100, 50, 300, 250)):
    return Rastro(
        id=1,
        classe_id=0,
        classe="buraco",
        caixa=caixa,
        confianca=0.5,
        confianca_maxima=0.87654,
        primeiro_quadro=0,
        ultimo_quadro=4,
        acertos=5,
    )


QUADRO = np.zeros((400, 800, 3), dtype=np.uint8)


def test_registra_no_formato_do_contrato(tmp_path):
    registro = RegistroOcorrencias(tmp_path, "camera-0", "via_atenta", latitude=-23.55, longitude=-46.63)
    registro.registrar(rastro(), QUADRO)
    linha = json.loads((tmp_path / "ocorrencias.jsonl").read_text(encoding="utf-8"))
    assert linha["classe"] == "buraco"
    assert linha["confianca"] == 0.877
    assert linha["quadros_observados"] == 5
    assert linha["caixa_normalizada"] == [0.125, 0.125, 0.375, 0.625]
    assert linha["status"] == "pendente_revisao"
    assert (linha["latitude"], linha["longitude"]) == (-23.55, -46.63)
    assert linha["versao_contrato"] == "1.0"
    assert linha["registrada_em"].endswith("+00:00")


def test_localizacao_e_opcional(tmp_path):
    ocorrencia = RegistroOcorrencias(tmp_path, "f", "m").registrar(rastro(), QUADRO)
    assert ocorrencia.latitude is None and ocorrencia.longitude is None


def test_salva_so_o_recorte_e_nao_o_quadro(tmp_path):
    ocorrencia = RegistroOcorrencias(tmp_path, "f", "m").registrar(rastro(), QUADRO)
    salvo = cv2.imread(str(tmp_path / ocorrencia.recorte))
    assert salvo.shape[:2] == (240, 240)
    assert [p.name for p in tmp_path.rglob("*.jpg")] == [f"{ocorrencia.id}.jpg"]


def test_recortes_podem_ser_desligados(tmp_path):
    ocorrencia = RegistroOcorrencias(tmp_path, "f", "m", salvar_recortes=False).registrar(rastro(), QUADRO)
    assert ocorrencia.recorte is None
    assert not list(tmp_path.rglob("*.jpg"))


def test_conta_por_classe(tmp_path):
    registro = RegistroOcorrencias(tmp_path, "f", "m")
    registro.registrar(rastro(), QUADRO)
    registro.registrar(rastro(), QUADRO)
    assert registro.total == {"buraco": 2}
    assert len((tmp_path / "ocorrencias.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_recorte_na_borda_nao_estoura():
    assert recortar(QUADRO, (780, 380, 800, 400)).shape[:2] == (22, 22)
