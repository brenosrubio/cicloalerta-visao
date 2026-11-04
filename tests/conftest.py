from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ATIVOS = Path(os.environ.get("VIA_ATIVOS", "tests/ativos"))


class SessaoFalsa:
    """Imita uma sessão do ONNX Runtime devolvendo uma saída pronta."""

    def __init__(self, saida: np.ndarray | None = None, metadados: dict | None = None) -> None:
        self.saida = saida
        self.metadados = metadados or {}

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.metadados)

    def run(self, _nomes, _entradas):
        return [self.saida]


@pytest.fixture
def sessao_falsa():
    return SessaoFalsa


@pytest.fixture
def modelo_real() -> Path:
    caminho = ATIVOS / "yolo11n.onnx"
    if not caminho.exists():
        pytest.skip("yolo11n.onnx ausente; defina VIA_ATIVOS para rodar este teste")
    return caminho


@pytest.fixture
def coco128() -> Path:
    caminho = ATIVOS / "coco128"
    if not caminho.exists():
        pytest.skip("coco128 ausente; defina VIA_ATIVOS para rodar este teste")
    return caminho


def escrever_xml(caminho: Path, objetos: list[tuple], largura: int = 600, altura: int = 600) -> None:
    corpo = "".join(
        f"<object><name>{rotulo}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>"
        f"<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
        for rotulo, x1, y1, x2, y2 in objetos
    )
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        f"<annotation><folder>images</folder><filename>{caminho.stem}.jpg</filename>"
        f"<size><width>{largura}</width><height>{altura}</height><depth>3</depth></size>{corpo}</annotation>",
        encoding="utf-8",
    )


def criar_amostra(raiz: Path, pais: str, numero: int, objetos: list[tuple]) -> None:
    nome = f"{pais}_{numero:06d}"
    pasta = raiz / pais / "train"
    escrever_xml(pasta / "annotations" / "xmls" / f"{nome}.xml", objetos)
    (pasta / "images").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pasta / "images" / f"{nome}.jpg"), np.full((60, 60, 3), numero % 255, dtype=np.uint8))


@pytest.fixture
def rdd_falso(tmp_path: Path) -> Path:
    """Árvore no layout do RDD2022, com volume suficiente para as três partições."""
    raiz = tmp_path / "rdd"
    for pais, deslocamento in (("India", 0), ("Czech", 7)):
        for i in range(600):
            numero = i + deslocamento
            if i % 5 == 0:
                objetos = [("D40", 100.0, 100.0, 300.0, 250.0)]
            elif i % 5 == 1:
                objetos = [("D00", 10, 20, 200, 80), ("D20", 300, 300, 500, 450)]
            elif i % 5 == 2:
                objetos = [("D10", 50, 50, 400, 90), ("D44", 1, 1, 50, 50)]
            else:
                objetos = []
            criar_amostra(raiz, pais, numero, objetos)
    # O conjunto de teste do RDD2022 não tem anotação e não pode ser lido.
    teste = raiz / "India" / "test" / "images"
    teste.mkdir(parents=True)
    cv2.imwrite(str(teste / "India_900000.jpg"), np.zeros((60, 60, 3), dtype=np.uint8))
    return raiz
