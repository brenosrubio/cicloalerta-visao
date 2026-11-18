"""Rastreamento simples por IoU.

Um mesmo buraco aparece em muitos quadros. O rastreador junta essas aparições em
uma só e só confirma o defeito depois de alguns quadros, o que corta falsos
positivos de um quadro. Com a câmera em movimento rápido a IoU entre quadros cai,
e o mesmo defeito pode virar dois rastros: a contagem é uma aproximação.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .detector import Deteccao


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersecao = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersecao
    return intersecao / uniao if uniao > 0 else 0.0


@dataclass
class Rastro:
    id: int
    classe_id: int
    classe: str
    caixa: tuple[float, float, float, float]
    confianca: float
    confianca_maxima: float
    primeiro_quadro: int
    ultimo_quadro: int
    acertos: int = 1
    perdidos: int = 0
    confirmado: bool = False


class RastreadorIoU:
    def __init__(self, min_quadros: int = 3, max_perdidos: int = 8, iou_minimo: float = 0.2) -> None:
        self.min_quadros = min_quadros
        self.max_perdidos = max_perdidos
        self.iou_minimo = iou_minimo
        self.rastros: list[Rastro] = []
        self._proximo_id = 1

    def _casar(self, deteccoes: Sequence[Deteccao]) -> dict[int, int]:
        """Casa cada detecção com um rastro da mesma classe, da maior IoU para a menor."""
        pares = [
            (iou(rastro.caixa, deteccao.caixa), r, d)
            for r, rastro in enumerate(self.rastros)
            for d, deteccao in enumerate(deteccoes)
            if rastro.classe_id == deteccao.classe_id
        ]
        pares = [par for par in pares if par[0] >= self.iou_minimo]
        pares.sort(key=lambda par: par[0], reverse=True)
        casados: dict[int, int] = {}
        usados: set[int] = set()
        for _, r, d in pares:
            if r not in usados and d not in casados:
                casados[d] = r
                usados.add(r)
        return casados

    def atualizar(self, deteccoes: Sequence[Deteccao], quadro: int) -> tuple[list[Rastro], list[Rastro]]:
        """Devolve os rastros confirmados visíveis e os que acabaram de ser confirmados."""
        casados = self._casar(deteccoes)
        rastros_vistos = set(casados.values())
        novos_confirmados: list[Rastro] = []

        for indice, rastro in enumerate(self.rastros):
            if indice not in rastros_vistos:
                rastro.perdidos += 1

        for d, deteccao in enumerate(deteccoes):
            if d in casados:
                rastro = self.rastros[casados[d]]
                rastro.caixa = deteccao.caixa
                rastro.confianca = deteccao.confianca
                rastro.confianca_maxima = max(rastro.confianca_maxima, deteccao.confianca)
                rastro.ultimo_quadro = quadro
                rastro.acertos += 1
                rastro.perdidos = 0
            else:
                rastro = Rastro(
                    id=self._proximo_id,
                    classe_id=deteccao.classe_id,
                    classe=deteccao.classe,
                    caixa=deteccao.caixa,
                    confianca=deteccao.confianca,
                    confianca_maxima=deteccao.confianca,
                    primeiro_quadro=quadro,
                    ultimo_quadro=quadro,
                )
                self._proximo_id += 1
                self.rastros.append(rastro)
            if not rastro.confirmado and rastro.acertos >= self.min_quadros:
                rastro.confirmado = True
                novos_confirmados.append(rastro)

        self.rastros = [r for r in self.rastros if r.perdidos <= self.max_perdidos]
        visiveis = [r for r in self.rastros if r.confirmado and r.perdidos == 0]
        return visiveis, novos_confirmados
