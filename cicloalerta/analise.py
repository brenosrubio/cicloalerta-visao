"""Análise de uma foto enviada pelo site, sem depender de servidor web.

O Flask deste repositório e o FastAPI do site chamam a mesma função.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .detector import DetectorONNX


def severidade_sugerida(classe: str, fracao_da_area: float) -> str:
    """Regra provisória pelo tamanho da caixa na foto. Não foi validada.

    Uma foto tirada de perto deixa qualquer defeito grande, então a revisão humana
    confirma a severidade antes de o relato virar alerta.
    """
    if classe == "buraco":
        return "alta" if fracao_da_area >= 0.04 else "media"
    return "media" if fracao_da_area >= 0.10 else "baixa"


def analisar_imagem(dados: bytes, detector: DetectorONNX) -> tuple[dict, int]:
    """Recebe os bytes de uma foto e devolve (corpo da resposta, código HTTP).

    Não depende do Flask, então o servidor FastAPI do site pode chamar esta função direto.
    """
    bruto = np.frombuffer(dados, dtype=np.uint8)
    quadro = cv2.imdecode(bruto, cv2.IMREAD_COLOR) if bruto.size else None
    if quadro is None:
        return {"erro": "Arquivo não é uma imagem válida."}, 415

    inicio = time.perf_counter()
    resultado = detector.detectar(quadro)
    altura, largura = quadro.shape[:2]
    area_total = float(largura * altura)
    deteccoes = []
    for d in resultado:
        x1, y1, x2, y2 = d.caixa
        fracao = max(0.0, x2 - x1) * max(0.0, y2 - y1) / area_total
        deteccoes.append(
            {
                "classe": d.classe,
                "confianca": round(d.confianca, 3),
                "caixa": [round(v, 1) for v in d.caixa],
                "severidade_sugerida": severidade_sugerida(d.classe, fracao),
            }
        )
    corpo = {
        "largura": largura,
        "altura": altura,
        "tempo_ms": round((time.perf_counter() - inicio) * 1000, 1),
        "status": "pendente_revisao",
        "tem_defeito": bool(deteccoes),
        "deteccoes": deteccoes,
    }
    return corpo, 200
