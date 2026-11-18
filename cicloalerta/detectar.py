"""Detecção em tempo real por webcam, celular, vídeo, imagem ou pasta de imagens.

Exemplos:
    python -m cicloalerta.detectar --fonte 0
    python -m cicloalerta.detectar --fonte http://192.168.0.10:8080/video
    python -m cicloalerta.detectar --fonte rua.mp4 --salvar-video
    python -m cicloalerta.detectar --fonte fotos/
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

import cv2
import numpy as np

from .detector import Deteccao, DetectorONNX
from .ocorrencias import RegistroOcorrencias
from .rastreador import RastreadorIoU

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TITULO_JANELA = "CicloAlerta (Q para sair)"
TECLAS_DE_SAIDA = {ord("q"), ord("Q"), 27}  # 27 é o Esc
CORES = {"buraco": (40, 40, 235), "rachadura": (0, 170, 255)}
COR_PADRAO = (0, 220, 120)


class LeitorAoVivo:
    """Lê a câmera em paralelo e guarda só o quadro mais recente.

    Sem isso, os quadros acumulam no buffer enquanto o modelo trabalha, e a imagem
    na tela vai ficando cada vez mais atrasada em relação ao mundo real.
    """

    def __init__(self, captura: cv2.VideoCapture) -> None:
        self.captura = captura
        self.quadro: np.ndarray | None = None
        self.sequencia = 0
        self.ativo = True
        self._condicao = threading.Condition()
        self._linha = threading.Thread(target=self._rodar, daemon=True)
        self._linha.start()

    def _rodar(self) -> None:
        while self.ativo:
            ok, quadro = self.captura.read()
            with self._condicao:
                if not ok:
                    self.ativo = False
                else:
                    self.quadro = quadro
                    self.sequencia += 1
                self._condicao.notify_all()

    def __iter__(self) -> Iterator[np.ndarray]:
        ultimo = 0
        while True:
            with self._condicao:
                while self.ativo and self.sequencia == ultimo:
                    self._condicao.wait(timeout=1.0)
                if self.sequencia == ultimo:
                    return
                ultimo = self.sequencia
                quadro = self.quadro
            yield quadro

    def parar(self) -> None:
        self.ativo = False
        self._linha.join(timeout=2)


def quadros_de_video(captura: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ok, quadro = captura.read()
        if not ok:
            return
        yield quadro


def listar_imagens(fonte: Path) -> list[Path]:
    if fonte.is_file():
        return [fonte]
    return sorted(p for p in fonte.iterdir() if p.suffix.lower() in EXTENSOES_IMAGEM)


def rotular_fonte(fonte: str) -> str:
    """Nome que vai para as ocorrências, sem caminho local nem senha de URL."""
    if fonte.isdigit():
        return f"camera-{fonte}"
    if "://" in fonte:
        return urlsplit(fonte).hostname or "camera-ip"
    return Path(fonte).name


def eh_fonte_de_imagens(fonte: str) -> bool:
    caminho = Path(fonte)
    return caminho.is_dir() or caminho.suffix.lower() in EXTENSOES_IMAGEM


def filtrar_regiao(
    deteccoes: list[Deteccao], altura: int, ignorar_topo: float, ignorar_base: float = 0.0
) -> list[Deteccao]:
    """Descarta o topo do quadro (céu e fachadas) e a base (guidão da bicicleta ou capô)."""
    if ignorar_topo <= 0 and ignorar_base <= 0:
        return deteccoes
    topo = altura * max(0.0, ignorar_topo)
    base = altura * (1 - max(0.0, ignorar_base))
    return [d for d in deteccoes if topo <= (d.caixa[1] + d.caixa[3]) / 2 <= base]


def deve_sair(tecla: int, janela_visivel: float) -> bool:
    """Sai com Q, com Caps Lock ligado ou não, com Esc ou ao fechar a janela no X."""
    return (tecla & 0xFF) in TECLAS_DE_SAIDA or janela_visivel < 1


def desenhar(quadro: np.ndarray, itens: list, fps: float | None = None) -> np.ndarray:
    """Desenha caixas. Aceita detecções e rastros, que têm classe, caixa e confiança."""
    saida = quadro.copy()
    escala = max(0.5, min(saida.shape[:2]) / 900)
    espessura = max(1, round(escala * 2))
    for item in itens:
        x1, y1, x2, y2 = (int(v) for v in item.caixa)
        cor = CORES.get(item.classe, COR_PADRAO)
        cv2.rectangle(saida, (x1, y1), (x2, y2), cor, espessura + 1)
        texto = f"{item.classe} {item.confianca:.2f}"
        (largura, altura), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, espessura)
        topo = max(0, y1 - altura - base - 6)
        cv2.rectangle(saida, (x1, topo), (x1 + largura + 8, topo + altura + base + 6), cor, -1)
        cv2.putText(
            saida, texto, (x1 + 4, topo + altura + 3), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), espessura,
            cv2.LINE_AA,
        )
    if fps is not None:
        cv2.putText(
            saida, f"{fps:.1f} FPS", (12, int(36 * escala + 8)), cv2.FONT_HERSHEY_SIMPLEX, escala * 1.2,
            (0, 255, 0), espessura + 1, cv2.LINE_AA,
        )
    return saida


def resumir(latencias_ms: list[float], segundos: float, registro: RegistroOcorrencias) -> dict:
    valores = np.array(latencias_ms) if latencias_ms else np.zeros(1)
    return {
        "quadros": len(latencias_ms),
        "duracao_s": round(segundos, 2),
        "fps_medio": round(len(latencias_ms) / segundos, 2) if segundos > 0 else 0.0,
        "latencia_media_ms": round(float(valores.mean()), 1),
        "latencia_p95_ms": round(float(np.percentile(valores, 95)), 1),
        "ocorrencias": registro.total,
    }


def processar_imagens(detector: DetectorONNX, args: argparse.Namespace, registro: RegistroOcorrencias) -> dict:
    latencias: list[float] = []
    inicio = time.perf_counter()
    pasta_anotadas = args.saida / "anotadas"
    pasta_anotadas.mkdir(parents=True, exist_ok=True)
    for caminho in listar_imagens(Path(args.fonte)):
        quadro = cv2.imread(str(caminho))
        if quadro is None:
            print(f"Ignorando arquivo ilegível: {caminho}", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        deteccoes = filtrar_regiao(detector.detectar(quadro), quadro.shape[0], args.ignorar_topo, args.ignorar_base)
        latencias.append((time.perf_counter() - t0) * 1000)
        rastreador = RastreadorIoU(min_quadros=1)
        _, confirmados = rastreador.atualizar(deteccoes, 0)
        for rastro in confirmados:
            registro.registrar(rastro, quadro)
        cv2.imwrite(str(pasta_anotadas / f"{caminho.stem}.jpg"), desenhar(quadro, deteccoes))
    return resumir(latencias, time.perf_counter() - inicio, registro)


def processar_fluxo(detector: DetectorONNX, args: argparse.Namespace, registro: RegistroOcorrencias) -> dict:
    ao_vivo = args.fonte.isdigit() or "://" in args.fonte
    if args.fonte.isdigit():
        captura = cv2.VideoCapture(int(args.fonte))
    else:
        captura = cv2.VideoCapture(args.fonte)
    if not captura.isOpened():
        raise SystemExit(f"Não consegui abrir a fonte: {args.fonte}")

    leitor = LeitorAoVivo(captura) if ao_vivo else None
    quadros = iter(leitor) if leitor else quadros_de_video(captura)
    rastreador = RastreadorIoU(args.min_quadros, args.max_perdidos)
    mostrar = not args.sem_janela
    escritor = None
    latencias: list[float] = []
    inicio = time.perf_counter()
    fps_suave = 0.0

    try:
        for indice, quadro in enumerate(quadros):
            t0 = time.perf_counter()
            deteccoes = filtrar_regiao(detector.detectar(quadro), quadro.shape[0], args.ignorar_topo, args.ignorar_base)
            visiveis, confirmados = rastreador.atualizar(deteccoes, indice)
            for rastro in confirmados:
                registro.registrar(rastro, quadro)
            decorrido = time.perf_counter() - t0
            latencias.append(decorrido * 1000)
            fps_suave = 1 / decorrido if fps_suave == 0 else 0.9 * fps_suave + 0.1 / decorrido
            anotado = desenhar(quadro, visiveis, fps_suave)

            if args.salvar_video:
                if escritor is None:
                    args.saida.mkdir(parents=True, exist_ok=True)
                    altura, largura = anotado.shape[:2]
                    taxa = captura.get(cv2.CAP_PROP_FPS) or 15.0
                    escritor = cv2.VideoWriter(
                        str(args.saida / "video_anotado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), taxa, (largura, altura)
                    )
                escritor.write(anotado)

            if mostrar:
                try:
                    cv2.imshow(TITULO_JANELA, anotado)
                    tecla = cv2.waitKey(1)
                    if deve_sair(tecla, cv2.getWindowProperty(TITULO_JANELA, cv2.WND_PROP_VISIBLE)):
                        break
                except cv2.error:
                    print("Sem suporte a janela neste ambiente. Seguindo sem exibir.", file=sys.stderr)
                    mostrar = False
    finally:
        if leitor:
            leitor.parar()
        captura.release()
        if escritor:
            escritor.release()
        if args.sem_janela is False and mostrar:
            cv2.destroyAllWindows()
    return resumir(latencias, time.perf_counter() - inicio, registro)


def criar_analisador() -> argparse.ArgumentParser:
    analisador = argparse.ArgumentParser(description="Detecta buracos e rachaduras em tempo real.")
    analisador.add_argument("--fonte", default="0", help="câmera (0), URL, vídeo, imagem ou pasta")
    analisador.add_argument("--modelo", type=Path, default=Path("modelos/cicloalerta.onnx"))
    analisador.add_argument("--saida", type=Path, default=Path("saida"))
    analisador.add_argument("--conf", type=float, default=None, help="troca os limiares calibrados por um valor único")
    analisador.add_argument("--min-quadros", type=int, default=3, help="quadros para confirmar um defeito")
    analisador.add_argument("--max-perdidos", type=int, default=8, help="quadros até esquecer um defeito")
    analisador.add_argument("--ignorar-topo", type=float, default=0.0, help="fração superior do quadro a ignorar")
    analisador.add_argument(
        "--ignorar-base", type=float, default=0.0, help="fração inferior do quadro a ignorar (guidão ou capô)"
    )
    analisador.add_argument("--latitude", type=float, default=None, help="ponto informado pelo usuário")
    analisador.add_argument("--longitude", type=float, default=None, help="ponto informado pelo usuário")
    analisador.add_argument("--salvar-video", action="store_true", help="grava o vídeo anotado, com o quadro inteiro")
    analisador.add_argument("--sem-recortes", action="store_true", help="não salva os recortes dos defeitos")
    analisador.add_argument("--sem-janela", action="store_true", help="não abre a janela de exibição")
    return analisador


def main(argv: list[str] | None = None) -> int:
    args = criar_analisador().parse_args(argv)
    limiares = None
    if args.conf is not None:
        limiares = {}
    try:
        detector = DetectorONNX(args.modelo, limiares=limiares, limiar_padrao=args.conf or 0.25)
    except FileNotFoundError as erro:
        raise SystemExit(str(erro)) from None
    registro = RegistroOcorrencias(
        args.saida, rotular_fonte(args.fonte), args.modelo.stem, args.latitude, args.longitude, not args.sem_recortes
    )
    if args.salvar_video:
        print("Aviso: o vídeo anotado guarda o quadro inteiro, que pode conter rostos e placas.", file=sys.stderr)

    processar = processar_imagens if eh_fonte_de_imagens(args.fonte) else processar_fluxo
    resumo = processar(detector, args, registro)
    args.saida.mkdir(parents=True, exist_ok=True)
    (args.saida / "desempenho.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
