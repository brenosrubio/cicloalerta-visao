"""Inferência do modelo exportado para ONNX, sem depender de PyTorch.

Espera a saída padrão do YOLO11 exportado: (1, 4 + classes, N), com caixas em
(cx, cy, l, a), em pixels da entrada, e pontuações já passadas pela sigmoide.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

NOMES_PADRAO = {0: "buraco", 1: "rachadura"}


@dataclass(frozen=True)
class Deteccao:
    classe_id: int
    classe: str
    confianca: float
    caixa: tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels da imagem original


@dataclass(frozen=True)
class Ajuste:
    """Como a imagem original foi encaixada na entrada do modelo."""

    escala: float
    margem_x: int
    margem_y: int


def letterbox(imagem: np.ndarray, tamanho: tuple[int, int]) -> tuple[np.ndarray, Ajuste]:
    """Redimensiona sem distorcer e completa com cinza, como no treino."""
    altura, largura = imagem.shape[:2]
    alvo_a, alvo_l = tamanho
    escala = min(alvo_a / altura, alvo_l / largura)
    nova_l, nova_a = round(largura * escala), round(altura * escala)
    if (nova_l, nova_a) != (largura, altura):
        imagem = cv2.resize(imagem, (nova_l, nova_a), interpolation=cv2.INTER_LINEAR)
    margem_y = round((alvo_a - nova_a) / 2 - 0.1)
    margem_x = round((alvo_l - nova_l) / 2 - 0.1)
    saida = np.full((alvo_a, alvo_l, 3), 114, dtype=np.uint8)
    saida[margem_y : margem_y + nova_a, margem_x : margem_x + nova_l] = imagem
    return saida, Ajuste(escala, margem_x, margem_y)


def nms(caixas: np.ndarray, pontuacoes: np.ndarray, limiar_iou: float) -> list[int]:
    """Supressão não máxima. Devolve os índices mantidos, do mais confiante ao menos."""
    ordem = pontuacoes.argsort()[::-1]
    areas = (caixas[:, 2] - caixas[:, 0]).clip(0) * (caixas[:, 3] - caixas[:, 1]).clip(0)
    mantidos: list[int] = []
    while ordem.size:
        atual = int(ordem[0])
        mantidos.append(atual)
        resto = ordem[1:]
        x1 = np.maximum(caixas[atual, 0], caixas[resto, 0])
        y1 = np.maximum(caixas[atual, 1], caixas[resto, 1])
        x2 = np.minimum(caixas[atual, 2], caixas[resto, 2])
        y2 = np.minimum(caixas[atual, 3], caixas[resto, 3])
        intersecao = (x2 - x1).clip(0) * (y2 - y1).clip(0)
        uniao = areas[atual] + areas[resto] - intersecao
        iou = np.divide(intersecao, uniao, out=np.zeros_like(intersecao, dtype=float), where=uniao > 0)
        ordem = resto[iou <= limiar_iou]
    return mantidos


def decodificar(
    saida: np.ndarray,
    ajuste: Ajuste,
    tamanho_original: tuple[int, int],
    nomes: Mapping[int, str],
    limiares: np.ndarray,
    limiar_iou: float = 0.5,
    maximo: int = 100,
) -> list[Deteccao]:
    """Transforma a saída bruta da rede em detecções na escala da imagem original."""
    if saida.ndim != 3 or saida.shape[0] != 1 or saida.shape[1] != 4 + len(nomes):
        raise ValueError(
            f"Saída {saida.shape} não é (1, {4 + len(nomes)}, N). "
            "Exporte um modelo YOLO11 para ONNX sem NMS embutido."
        )
    predicoes = saida[0].T
    pontuacoes_classes = predicoes[:, 4:]
    classes = pontuacoes_classes.argmax(1)
    confiancas = pontuacoes_classes[np.arange(len(classes)), classes]
    mascara = confiancas >= limiares[classes]
    if not mascara.any():
        return []

    cx, cy, largura, altura = predicoes[mascara, :4].T
    caixas = np.stack((cx - largura / 2, cy - altura / 2, cx + largura / 2, cy + altura / 2), axis=1)
    classes, confiancas = classes[mascara], confiancas[mascara]

    # A soma por classe separa as caixas no espaço, então o NMS atua dentro de cada classe.
    deslocamento = classes[:, None].astype(caixas.dtype) * 10_000.0
    mantidos = nms(caixas + deslocamento, confiancas, limiar_iou)[:maximo]

    altura_img, largura_img = tamanho_original
    caixas[:, [0, 2]] = ((caixas[:, [0, 2]] - ajuste.margem_x) / ajuste.escala).clip(0, largura_img)
    caixas[:, [1, 3]] = ((caixas[:, [1, 3]] - ajuste.margem_y) / ajuste.escala).clip(0, altura_img)

    return [
        Deteccao(
            classe_id=int(classes[i]),
            classe=nomes[int(classes[i])],
            confianca=float(confiancas[i]),
            caixa=tuple(float(v) for v in caixas[i]),
        )
        for i in mantidos
    ]


def _ler_metadados(sessao) -> tuple[dict[int, str] | None, tuple[int, int] | None]:
    metadados = sessao.get_modelmeta().custom_metadata_map
    nomes = imgsz = None
    if "names" in metadados:
        nomes = {int(k): str(v) for k, v in ast.literal_eval(metadados["names"]).items()}
    if "imgsz" in metadados:
        valor = ast.literal_eval(metadados["imgsz"])
        imgsz = (int(valor[0]), int(valor[1])) if isinstance(valor, (list, tuple)) else (int(valor), int(valor))
    return nomes, imgsz


class DetectorONNX:
    """Carrega o modelo e detecta em imagens BGR, o formato usado pelo OpenCV."""

    def __init__(
        self,
        modelo: str | Path,
        limiares: Mapping[str, float] | None = None,
        limiar_padrao: float = 0.25,
        limiar_iou: float = 0.5,
        provedores: Sequence[str] = ("CPUExecutionProvider",),
        sessao=None,
    ) -> None:
        if sessao is None:
            import onnxruntime as ort

            caminho = Path(modelo)
            if not caminho.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {caminho}. Rode o notebook de treino primeiro.")
            sessao = ort.InferenceSession(str(caminho), providers=list(provedores))
            if limiares is None:
                arquivo = caminho.with_name("limiares.json")
                if arquivo.exists():
                    limiares = json.loads(arquivo.read_text(encoding="utf-8"))

        self.sessao = sessao
        self.entrada = sessao.get_inputs()[0].name
        nomes, imgsz = _ler_metadados(sessao)
        self.nomes = nomes or NOMES_PADRAO
        self.imgsz = imgsz or (640, 640)
        self.limiar_iou = limiar_iou
        self.limiares = np.full(len(self.nomes), limiar_padrao, dtype=np.float32)
        for classe_id, nome in self.nomes.items():
            if limiares and nome in limiares:
                self.limiares[classe_id] = limiares[nome]

    def preprocessar(self, quadro: np.ndarray) -> tuple[np.ndarray, Ajuste]:
        encaixada, ajuste = letterbox(quadro, self.imgsz)
        tensor = encaixada[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return tensor, ajuste

    def detectar(self, quadro: np.ndarray) -> list[Deteccao]:
        tensor, ajuste = self.preprocessar(quadro)
        saida = self.sessao.run(None, {self.entrada: tensor})[0]
        return decodificar(saida, ajuste, quadro.shape[:2], self.nomes, self.limiares, self.limiar_iou)
