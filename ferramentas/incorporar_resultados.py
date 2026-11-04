#!/usr/bin/env python3
"""Coloca o resultados.zip do Colab no repositório e gera o que depende dele.

Passos: copia modelo e métricas, roda o aplicativo nas amostras do teste com o ONNX
treinado, monta as grades de acertos e erros, gera um vídeo de demonstração, atualiza
o README e refaz o relatório.

Uso: python ferramentas/incorporar_resultados.py resultados.zip
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from via_atenta import detectar as app  # noqa: E402
from via_atenta.detector import DetectorONNX  # noqa: E402
from via_atenta.rastreador import iou  # noqa: E402

IOU_ACERTO = 0.4
INICIO_README = "<!-- RESULTADOS:INICIO -->"
FIM_README = "<!-- RESULTADOS:FIM -->"


def ler_rotulos(caminho: Path, largura: int, altura: int) -> list[tuple[int, list[float]]]:
    caixas = []
    if not caminho.exists():
        return caixas
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        classe, cx, cy, l, a = map(float, linha.split())
        caixas.append(
            (int(classe), [(cx - l / 2) * largura, (cy - a / 2) * altura, (cx + l / 2) * largura, (cy + a / 2) * altura])
        )
    return caixas


def analisar_amostras(detector: DetectorONNX, pasta: Path) -> list[dict]:
    analises = []
    for imagem in sorted((pasta / "images").glob("*.jpg")):
        quadro = cv2.imread(str(imagem))
        if quadro is None:
            continue
        altura, largura = quadro.shape[:2]
        reais = ler_rotulos(pasta / "labels" / f"{imagem.stem}.txt", largura, altura)
        previstas = detector.detectar(quadro)
        usadas: set[int] = set()
        acertos = 0
        for classe, caixa in reais:
            candidatas = [
                (iou(p.caixa, caixa), i)
                for i, p in enumerate(previstas)
                if p.classe_id == classe and i not in usadas
            ]
            melhor = max(candidatas, default=(0.0, -1))
            if melhor[0] >= IOU_ACERTO:
                usadas.add(melhor[1])
                acertos += 1
        analises.append(
            {
                "quadro": quadro,
                "reais": reais,
                "previstas": previstas,
                "acertos": acertos,
                "falsos_negativos": len(reais) - acertos,
                "falsos_positivos": len(previstas) - acertos,
                "tem_buraco": any(classe == 0 for classe, _ in reais),
            }
        )
    return analises


def ilustrar(analise: dict, lado: int = 480) -> np.ndarray:
    quadro = analise["quadro"].copy()
    espessura = max(2, round(min(quadro.shape[:2]) / 300))
    for _, caixa in analise["reais"]:
        x1, y1, x2, y2 = (int(v) for v in caixa)
        cv2.rectangle(quadro, (x1, y1), (x2, y2), (60, 200, 60), espessura)
    quadro = app.desenhar(quadro, analise["previstas"])
    escala = lado / max(quadro.shape[:2])
    quadro = cv2.resize(quadro, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    tela = np.full((lado, lado, 3), 255, dtype=np.uint8)
    altura, largura = quadro.shape[:2]
    tela[(lado - altura) // 2 : (lado - altura) // 2 + altura, (lado - largura) // 2 : (lado - largura) // 2 + largura] = quadro
    return tela


def grade(imagens: list[np.ndarray], colunas: int = 3, respiro: int = 8) -> np.ndarray:
    lado = imagens[0].shape[0]
    linhas = (len(imagens) + colunas - 1) // colunas
    tela = np.full(
        (linhas * lado + (linhas - 1) * respiro, colunas * lado + (colunas - 1) * respiro, 3), 255, dtype=np.uint8
    )
    for indice, imagem in enumerate(imagens):
        linha, coluna = divmod(indice, colunas)
        y, x = linha * (lado + respiro), coluna * (lado + respiro)
        tela[y : y + lado, x : x + lado] = imagem
    return tela


def escolher_acertos(analises: list[dict], quantidade: int = 6) -> list[dict]:
    limpas = [a for a in analises if a["reais"] and not a["falsos_negativos"] and not a["falsos_positivos"]]
    limpas.sort(key=lambda a: (not a["tem_buraco"], -a["acertos"]))
    com_buraco = [a for a in limpas if a["tem_buraco"]][: quantidade // 2]
    resto = [a for a in limpas if not a["tem_buraco"]][: quantidade - len(com_buraco)]
    return (com_buraco + resto + limpas)[:quantidade] if len(com_buraco + resto) < quantidade else com_buraco + resto


def escolher_erros(analises: list[dict], quantidade: int = 6) -> list[dict]:
    com_erro = [a for a in analises if a["falsos_negativos"] or a["falsos_positivos"]]
    return sorted(com_erro, key=lambda a: a["falsos_negativos"] + a["falsos_positivos"], reverse=True)[:quantidade]


def gerar_video_demo(pasta_imagens: Path, saida: Path, taxa: int = 15, segundos: float = 2.0) -> None:
    """Aproxima devagar cada imagem, imitando um veículo que anda. Serve de fonte para o aplicativo."""
    imagens = sorted(pasta_imagens.glob("*.jpg"))
    if not imagens:
        return
    imagens = imagens[:: max(1, len(imagens) // 20)][:20]  # espalha para pegar os três países
    saida.parent.mkdir(parents=True, exist_ok=True)
    tamanho = (720, 720)
    escritor = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"mp4v"), taxa, tamanho)
    for caminho in imagens:
        quadro = cv2.resize(cv2.imread(str(caminho)), tamanho)
        total = int(taxa * segundos)
        for passo in range(total):
            corte = 1 - 0.12 * passo / total
            lado = int(720 * corte)
            inicio_y = 720 - lado  # mantém a parte de baixo, onde fica o asfalto
            inicio_x = (720 - lado) // 2
            escritor.write(cv2.resize(quadro[inicio_y:720, inicio_x : inicio_x + lado], tamanho))
    escritor.release()


def tabela_readme(metricas: dict) -> str:
    teste = metricas["teste"]
    linhas = [
        f"mAP50 no teste: **{teste['geral']['map50']:.3f}**. mAP50-95: **{teste['geral']['map50_95']:.3f}**.",
        "",
        "| Classe | Instâncias | AP50 | Limiar | Precisão | Revocação |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for classe, c in teste["por_classe"].items():
        linhas.append(
            f"| {classe} | {c['instancias']} | {c['ap50']:.3f} | {c['limiar']:.2f} | "
            f"{c['precisao_no_limiar']:.3f} | {c['revocacao_no_limiar']:.3f} |"
        )
    ms = metricas["onnx"]["latencia_cpu_colab_ms"]
    linhas += ["", f"Velocidade do aplicativo em CPU do Colab: {ms:.0f} ms por quadro ({1000 / ms:.1f} FPS)."]
    return "\n".join(linhas)


def atualizar_readme(raiz: Path, metricas: dict) -> None:
    arquivo = raiz / "README.md"
    if not arquivo.exists():
        return
    texto = arquivo.read_text(encoding="utf-8")
    inicio, fim = texto.index(INICIO_README), texto.index(FIM_README)
    novo = f"{texto[:inicio]}{INICIO_README}\n{tabela_readme(metricas)}\n{texto[fim:]}"
    arquivo.write_text(novo, encoding="utf-8")


def main() -> int:
    analisador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analisador.add_argument("zip", type=Path)
    analisador.add_argument("--raiz", type=Path, default=RAIZ)
    args = analisador.parse_args()
    raiz = args.raiz

    with tempfile.TemporaryDirectory() as temporario:
        pacote = Path(temporario)
        with zipfile.ZipFile(args.zip) as arquivo:
            arquivo.extractall(pacote)
        (raiz / "modelos").mkdir(exist_ok=True)
        for arquivo in (pacote / "modelos").iterdir():
            shutil.copy2(arquivo, raiz / "modelos" / arquivo.name)
        destino = raiz / "resultados"
        if destino.exists():
            shutil.rmtree(destino)
        destino.mkdir()
        for item in pacote.iterdir():
            if item.name == "modelos":
                continue
            if item.is_dir():
                shutil.copytree(item, destino / item.name)
            else:
                shutil.copy2(item, destino / item.name)

    metricas = json.loads((destino / "metricas.json").read_text(encoding="utf-8"))
    modelo = raiz / "modelos" / "via_atenta.onnx"
    amostras = destino / "amostras_teste"

    analises = analisar_amostras(DetectorONNX(modelo), amostras)
    exemplos = destino / "exemplos"
    exemplos.mkdir()
    acertos, erros = escolher_acertos(analises), escolher_erros(analises)
    if acertos:
        cv2.imwrite(str(exemplos / "acertos.jpg"), grade([ilustrar(a) for a in acertos]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if erros:
        cv2.imwrite(str(exemplos / "erros.jpg"), grade([ilustrar(a) for a in erros]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Roda o aplicativo de verdade: as ocorrências e a velocidade do relatório vêm daqui.
    app.main(
        ["--fonte", str(amostras / "images"), "--modelo", str(modelo), "--saida", str(destino / "exemplo_saida")]
    )
    gerar_video_demo(amostras / "images", raiz / "docs" / "demo" / "demo_rdd2022.mp4")
    atualizar_readme(raiz, metricas)

    from gerar_relatorio import montar

    print(json.dumps(montar(raiz, raiz / "docs" / "relatorio_via_atenta.pdf"), ensure_ascii=False))
    print(f"Pronto. {len(analises)} amostras, {len(acertos)} acertos e {len(erros)} erros ilustrados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
