"""Registro das ocorrências que os outros componentes do MVP consomem.

Cada defeito confirmado vira uma linha em ocorrencias.jsonl. Tudo nasce com status
"pendente_revisao": o modelo sugere, e uma pessoa confirma antes de virar chamado.
Por padrão só o recorte do defeito é salvo, nunca o quadro inteiro, para reduzir a
chance de guardar rostos e placas.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .rastreador import Rastro

VERSAO_CONTRATO = "1.0"
STATUS_INICIAL = "pendente_revisao"


@dataclass
class Ocorrencia:
    id: str
    versao_contrato: str
    registrada_em: str
    classe: str
    confianca: float
    quadros_observados: int
    caixa_normalizada: tuple[float, float, float, float]
    fonte: str
    modelo: str
    status: str
    latitude: Optional[float]
    longitude: Optional[float]
    recorte: Optional[str]


def recortar(quadro: np.ndarray, caixa: tuple[float, float, float, float], margem: float = 0.1) -> np.ndarray:
    altura, largura = quadro.shape[:2]
    x1, y1, x2, y2 = caixa
    folga_x, folga_y = (x2 - x1) * margem, (y2 - y1) * margem
    x1 = max(0, int(x1 - folga_x))
    y1 = max(0, int(y1 - folga_y))
    x2 = min(largura, int(x2 + folga_x))
    y2 = min(altura, int(y2 + folga_y))
    return quadro[y1:y2, x1:x2]


class RegistroOcorrencias:
    def __init__(
        self,
        pasta: str | Path,
        fonte: str,
        modelo: str,
        latitude: float | None = None,
        longitude: float | None = None,
        salvar_recortes: bool = True,
    ) -> None:
        self.pasta = Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.arquivo = self.pasta / "ocorrencias.jsonl"
        self.fonte = fonte
        self.modelo = modelo
        self.latitude = latitude
        self.longitude = longitude
        self.salvar_recortes = salvar_recortes
        self.total: dict[str, int] = {}

    def registrar(self, rastro: Rastro, quadro: np.ndarray) -> Ocorrencia:
        identificador = uuid.uuid4().hex[:12]
        altura, largura = quadro.shape[:2]
        x1, y1, x2, y2 = rastro.caixa
        caminho_recorte = None
        if self.salvar_recortes:
            imagem = recortar(quadro, rastro.caixa)
            if imagem.size:
                relativo = Path("recortes") / f"{identificador}.jpg"
                (self.pasta / relativo).parent.mkdir(exist_ok=True)
                cv2.imwrite(str(self.pasta / relativo), imagem)
                caminho_recorte = relativo.as_posix()

        ocorrencia = Ocorrencia(
            id=identificador,
            versao_contrato=VERSAO_CONTRATO,
            registrada_em=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            classe=rastro.classe,
            confianca=round(rastro.confianca_maxima, 3),
            quadros_observados=rastro.acertos,
            caixa_normalizada=(
                round(x1 / largura, 4),
                round(y1 / altura, 4),
                round(x2 / largura, 4),
                round(y2 / altura, 4),
            ),
            fonte=self.fonte,
            modelo=self.modelo,
            status=STATUS_INICIAL,
            latitude=self.latitude,
            longitude=self.longitude,
            recorte=caminho_recorte,
        )
        with self.arquivo.open("a", encoding="utf-8") as saida:
            saida.write(json.dumps(asdict(ocorrencia), ensure_ascii=False) + "\n")
        self.total[rastro.classe] = self.total.get(rastro.classe, 0) + 1
        return ocorrencia
