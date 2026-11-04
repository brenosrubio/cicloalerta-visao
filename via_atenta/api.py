"""API HTTP que entrega o componente de visão aos outros componentes do MVP.

A imagem é processada em memória e descartada: nada é gravado em disco.

    python -m via_atenta.api --modelo modelos/via_atenta.onnx --porta 8000
    curl -F imagem=@rua.jpg http://127.0.0.1:8000/v1/deteccoes
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

from .detector import DetectorONNX

TAMANHO_MAXIMO = 8 * 1024 * 1024


def criar_app(detector: DetectorONNX) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = TAMANHO_MAXIMO

    @app.get("/saude")
    def saude():
        return jsonify(status="ok", classes=list(detector.nomes.values()))

    @app.post("/v1/deteccoes")
    def deteccoes():
        arquivo = request.files.get("imagem")
        if arquivo is None:
            return jsonify(erro="Envie o arquivo no campo 'imagem'."), 400
        bruto = np.frombuffer(arquivo.read(), dtype=np.uint8)
        quadro = cv2.imdecode(bruto, cv2.IMREAD_COLOR) if bruto.size else None
        if quadro is None:
            return jsonify(erro="Arquivo não é uma imagem válida."), 415

        inicio = time.perf_counter()
        resultado = detector.detectar(quadro)
        altura, largura = quadro.shape[:2]
        return jsonify(
            largura=largura,
            altura=altura,
            tempo_ms=round((time.perf_counter() - inicio) * 1000, 1),
            status="pendente_revisao",
            deteccoes=[
                {
                    "classe": d.classe,
                    "confianca": round(d.confianca, 3),
                    "caixa": [round(v, 1) for v in d.caixa],
                }
                for d in resultado
            ],
        )

    @app.errorhandler(413)
    def grande_demais(_erro):
        return jsonify(erro="Imagem maior que 8 MB."), 413

    return app


def main() -> None:
    analisador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analisador.add_argument("--modelo", type=Path, default=Path("modelos/via_atenta.onnx"))
    analisador.add_argument("--host", default="127.0.0.1")
    analisador.add_argument("--porta", type=int, default=8000)
    args = analisador.parse_args()
    try:
        detector = DetectorONNX(args.modelo)
    except FileNotFoundError as erro:
        raise SystemExit(str(erro)) from None
    criar_app(detector).run(host=args.host, port=args.porta)


if __name__ == "__main__":
    main()
