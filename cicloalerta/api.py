"""API HTTP que entrega o componente de visão aos outros componentes do MVP.

A imagem é processada em memória e descartada: nada é gravado em disco.

    python -m cicloalerta.api --modelo modelos/cicloalerta.onnx --porta 8000
    curl -F imagem=@foto.jpg http://127.0.0.1:8000/api/detectar

`/api/detectar` é a rota combinada no contrato do MVP. `/v1/deteccoes` é a mesma
coisa e continua existindo para quem já usava.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, request

from .analise import analisar_imagem, severidade_sugerida  # noqa: F401  (reexportadas)
from .detector import DetectorONNX

TAMANHO_MAXIMO = 8 * 1024 * 1024
CAMPOS_ACEITOS = ("imagem", "foto")


def criar_app(detector: DetectorONNX) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = TAMANHO_MAXIMO

    @app.get("/saude")
    def saude():
        return jsonify(status="ok", classes=list(detector.nomes.values()))

    def detectar():
        arquivo = next((request.files[c] for c in CAMPOS_ACEITOS if c in request.files), None)
        if arquivo is None:
            return jsonify(erro="Envie a foto no campo 'imagem'."), 400
        corpo, codigo = analisar_imagem(arquivo.read(), detector)
        return jsonify(corpo), codigo

    app.add_url_rule("/api/detectar", "detectar", detectar, methods=["POST"])
    app.add_url_rule("/v1/deteccoes", "deteccoes", detectar, methods=["POST"])

    @app.errorhandler(413)
    def grande_demais(_erro):
        return jsonify(erro="Imagem maior que 8 MB."), 413

    return app


def main() -> None:
    analisador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analisador.add_argument("--modelo", type=Path, default=Path("modelos/cicloalerta.onnx"))
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
