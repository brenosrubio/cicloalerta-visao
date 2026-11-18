"""Mede o modelo em recortes aproximados das amostras do teste.

A tela Reportar do site recebe fotos tiradas de perto, por alguém em pé ao lado do
buraco. O modelo foi treinado com câmera de carro, em que o defeito aparece pequeno.
Este teste recorta um quadrado em volta de cada marcação humana e verifica se o
modelo ainda encontra o defeito.

O recorte é ampliado para a entrada do modelo, então fica mais borrado do que uma
foto real tirada de perto. O resultado é uma estimativa pessimista.

    python ferramentas/testar_fotos_aproximadas.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from cicloalerta.detector import DetectorONNX  # noqa: E402
from cicloalerta.rastreador import iou  # noqa: E402

NOMES = {0: "buraco", 1: "rachadura"}
IOU_MINIMO = 0.3
LADO_MINIMO = 160
# Quantas vezes o lado do recorte é maior que o lado maior da caixa.
AMPLIACOES = {"foto_inteira": None, "recorte_medio": 4.0, "recorte_perto": 2.5}


def ler_marcacoes(rotulo: Path, largura: int, altura: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    marcacoes = []
    for linha in rotulo.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        classe, cx, cy, l, a = linha.split()
        cx, cy, l, a = float(cx) * largura, float(cy) * altura, float(l) * largura, float(a) * altura
        marcacoes.append((int(classe), (cx - l / 2, cy - a / 2, cx + l / 2, cy + a / 2)))
    return marcacoes


def recortar(imagem, caixa, fator):
    altura, largura = imagem.shape[:2]
    lado = max(max(caixa[2] - caixa[0], caixa[3] - caixa[1]) * fator, LADO_MINIMO)
    cx, cy = (caixa[0] + caixa[2]) / 2, (caixa[1] + caixa[3]) / 2
    x1, y1 = int(max(0, cx - lado / 2)), int(max(0, cy - lado / 2))
    x2, y2 = int(min(largura, cx + lado / 2)), int(min(altura, cy + lado / 2))
    deslocada = (caixa[0] - x1, caixa[1] - y1, caixa[2] - x1, caixa[3] - y1)
    return imagem[y1:y2, x1:x2], deslocada


def medir(detector: DetectorONNX, pasta: Path) -> dict:
    contagem = {nome: {classe: [0, 0] for classe in NOMES.values()} for nome in AMPLIACOES}
    for caminho in sorted((pasta / "images").glob("*.jpg")):
        imagem = cv2.imread(str(caminho))
        rotulo = pasta / "labels" / f"{caminho.stem}.txt"
        if imagem is None or not rotulo.exists():
            continue
        altura, largura = imagem.shape[:2]
        marcacoes = ler_marcacoes(rotulo, largura, altura)
        deteccoes_inteira = detector.detectar(imagem)
        for classe_id, caixa in marcacoes:
            classe = NOMES[classe_id]
            for nome, fator in AMPLIACOES.items():
                if fator is None:
                    deteccoes, alvo = deteccoes_inteira, caixa
                else:
                    recorte, alvo = recortar(imagem, caixa, fator)
                    deteccoes = detector.detectar(recorte)
                achou = any(d.classe == classe and iou(d.caixa, alvo) >= IOU_MINIMO for d in deteccoes)
                contagem[nome][classe][0] += int(achou)
                contagem[nome][classe][1] += 1
    return {
        nome: {
            classe: {"encontrados": a, "total": t, "taxa": round(a / t, 3) if t else None}
            for classe, (a, t) in por_classe.items()
        }
        for nome, por_classe in contagem.items()
    }


def main() -> int:
    detector = DetectorONNX(RAIZ / "modelos" / "cicloalerta.onnx")
    resultado = {
        "descricao": "Defeitos marcados encontrados pelo modelo (mesma classe, IoU >= 0,3) nas 60 amostras do teste, "
        "na foto inteira e em recortes quadrados em volta de cada marcação",
        "lado_do_recorte": {"recorte_medio": "4 vezes o lado maior da caixa", "recorte_perto": "2,5 vezes o lado maior da caixa"},
        "resultados": medir(detector, RAIZ / "resultados" / "amostras_teste"),
    }
    destino = RAIZ / "resultados" / "teste_aproximado.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado["resultados"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
