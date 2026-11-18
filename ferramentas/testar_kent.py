"""Mede o modelo em fotos de celular de buracos, muitos cheios de água (base de Kent).

A base tem 713 fotos tiradas com celular no Reino Unido, com 1.153 buracos marcados
(Data in Brief, 2023, DOI 10.17632/tp95cdvgm8.1, licença CC BY 4.0). O modelo nunca viu
essas fotos, então todas servem de teste. As fotos da partição de teste também foram
separadas à mão em "com água" e "seco" (arquivo resultados/kent_agua.json).

    python ferramentas/testar_kent.py caminho/para/kent_pequeno
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from cicloalerta.detector import DetectorONNX  # noqa: E402
from cicloalerta.rastreador import iou  # noqa: E402


def ler_caixas(rotulo: Path, largura: int, altura: int) -> list[tuple[float, float, float, float]]:
    caixas = []
    for linha in rotulo.read_text(encoding="utf-8").splitlines():
        partes = linha.split()
        if len(partes) != 5:
            continue
        cx, cy, l, a = (float(v) for v in partes[1:])
        caixas.append(((cx - l / 2) * largura, (cy - a / 2) * altura, (cx + l / 2) * largura, (cy + a / 2) * altura))
    return caixas


def analisar(detector: DetectorONNX, pasta: Path) -> dict[str, dict]:
    """Para cada foto: buracos marcados e todas as detecções de buraco com a confiança."""
    resultado = {}
    for imagem in sorted((pasta / "images").rglob("*.jpg")):
        quadro = cv2.imread(str(imagem))
        altura, largura = quadro.shape[:2]
        reais = ler_caixas(pasta / "labels" / imagem.parent.name / f"{imagem.stem}.txt", largura, altura)
        deteccoes = detector.detectar(quadro)
        resultado[imagem.stem] = {
            "particao": imagem.parent.name,
            "reais": reais,
            "buracos": [(d.confianca, d.caixa) for d in deteccoes if d.classe == "buraco"],
            "rachaduras": [d.confianca for d in deteccoes if d.classe == "rachadura"],
        }
    return resultado


def medir(analises: list[dict], limiares: dict, iou_minimo: float = 0.5) -> dict:
    """AP50 da classe buraco, precisão e revocação no limiar do aplicativo e confirmação por foto."""
    limiar = limiares["buraco"]
    total_reais = sum(len(a["reais"]) for a in analises)
    acertos = []  # (confiança, é acerto)
    for a in analises:
        usados = set()
        for confianca, caixa in sorted(a["buracos"], key=lambda d: -d[0]):
            melhor, indice = 0.0, -1
            for i, real in enumerate(a["reais"]):
                if i in usados:
                    continue
                valor = iou(caixa, real)
                if valor > melhor:
                    melhor, indice = valor, i
            if melhor >= iou_minimo:
                usados.add(indice)
                acertos.append((confianca, True))
            else:
                acertos.append((confianca, False))
    acertos.sort(key=lambda x: -x[0])
    vp = np.cumsum([1 if ok else 0 for _, ok in acertos])
    fp = np.cumsum([0 if ok else 1 for _, ok in acertos])
    revocacao = vp / max(total_reais, 1)
    precisao = vp / np.maximum(vp + fp, 1)
    # AP com interpolação em todos os pontos
    r = np.concatenate([[0.0], revocacao, [1.0]])
    p = np.concatenate([[1.0], precisao, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    passos = np.where(r[1:] != r[:-1])[0]
    ap = float(np.sum((r[passos + 1] - r[passos]) * p[passos + 1]))
    no_limiar = [ok for c, ok in acertos if c >= limiar]
    vp_l = sum(no_limiar)
    com_buraco = [a for a in analises if a["reais"]]
    confirmou_buraco = sum(1 for a in com_buraco if any(c >= limiar for c, _ in a["buracos"]))
    confirmou_defeito = sum(
        1 for a in com_buraco
        if any(c >= limiar for c, _ in a["buracos"]) or any(c >= limiares["rachadura"] for c in a["rachaduras"])
    )
    return {
        "fotos": len(analises),
        "buracos_marcados": total_reais,
        "ap50": round(ap, 3),
        "limiar": limiar,
        "precisao_no_limiar": round(vp_l / len(no_limiar), 3) if no_limiar else None,
        "revocacao_no_limiar": round(vp_l / max(total_reais, 1), 3),
        "fotos_em_que_aponta_buraco": round(confirmou_buraco / max(len(com_buraco), 1), 3),
        "fotos_em_que_aponta_algum_defeito": round(confirmou_defeito / max(len(com_buraco), 1), 3),
    }


def main() -> int:
    analisador = argparse.ArgumentParser()
    analisador.add_argument("pasta", type=Path, help="pasta com images/ e labels/ da base de Kent")
    analisador.add_argument("--modelo", type=Path, default=RAIZ / "modelos" / "cicloalerta.onnx")
    args = analisador.parse_args()

    limiares = json.loads((args.modelo.parent / "limiares.json").read_text(encoding="utf-8"))
    detector = DetectorONNX(args.modelo, limiares={}, limiar_padrao=0.01)
    analises = analisar(detector, args.pasta)

    agua_arquivo = RAIZ / "resultados" / "kent_agua.json"
    agua = json.loads(agua_arquivo.read_text(encoding="utf-8")) if agua_arquivo.exists() else {}

    grupos = {
        "todas_as_fotos": list(analises.values()),
        "particao_teste": [a for a in analises.values() if a["particao"] == "test"],
    }
    if agua:
        grupos["teste_com_agua"] = [analises[n] for n, v in agua.items() if v == "agua" and n in analises]
        grupos["teste_seco"] = [analises[n] for n, v in agua.items() if v == "seco" and n in analises]

    resultado = {
        "base": "Kent (Data in Brief, 2023), 713 fotos de celular, licença CC BY 4.0",
        "modelo": args.modelo.name,
        "criterio": "classe buraco, IoU >= 0,5",
        "grupos": {nome: medir(lista, limiares) for nome, lista in grupos.items()},
        "grupos_iou_30": {nome: medir(lista, limiares, 0.3) for nome, lista in grupos.items()},
    }
    (RAIZ / "resultados" / "teste_kent.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
