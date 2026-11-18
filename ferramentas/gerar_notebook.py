#!/usr/bin/env python3
"""Gera o notebook do Colab embutindo o mesmo código testado em cicloalerta/.

Assim existe uma só versão do preparo de dados e do detector. Rode de novo sempre
que um desses arquivos mudar.
"""
from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "notebooks" / "treino_cicloalerta_colab.ipynb"
VERSAO_ULTRALYTICS = "8.4.200"


def markdown(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").splitlines(keepends=True)}


def codigo(texto: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texto.strip("\n").splitlines(keepends=True),
    }


def embutir(arquivo: str) -> dict:
    conteudo = (RAIZ / "cicloalerta" / arquivo).read_text(encoding="utf-8")
    return codigo(f"%%writefile {arquivo}\n{conteudo}")


CELULAS = [
    markdown(
        """
# CicloAlerta: treino do detector de buracos e rachaduras

Este notebook treina o YOLO11n com a base pública RDD2022, avalia o modelo em imagens
que ele nunca viu e gera o arquivo `resultados.zip`.

**Como usar:** menu **Ambiente de execução > Executar tudo**. Mantenha a aba aberta até o
fim. No final, o `resultados.zip` é baixado sozinho. Se o download não começar, abra o
ícone de pasta na lateral, procure `cicloalerta/resultados.zip` e baixe por lá.

O notebook já pede GPU T4. Se a primeira célula acusar falta de GPU, vá em
**Ambiente de execução > Alterar o tipo de ambiente de execução > GPU T4** e rode de novo.
"""
    ),
    markdown("## 1. Ambiente"),
    codigo(
        f"""
import os
import shutil
import subprocess
import sys

if shutil.which("nvidia-smi") is None or subprocess.run(["nvidia-smi"], capture_output=True).returncode != 0:
    raise RuntimeError(
        "Sem GPU. Vá em Ambiente de execução > Alterar o tipo de ambiente de execução > GPU T4 e rode de novo."
    )

os.environ["WANDB_MODE"] = "disabled"
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ultralytics=={VERSAO_ULTRALYTICS}", "onnx", "onnxslim", "onnxruntime"],
    check=True,
)
print("Ambiente pronto.")
"""
    ),
    markdown("## 2. Configuração"),
    codigo(
        """
from pathlib import Path

PAISES = ["India", "Japan", "Czech"]  # captura com celular em carro, como no uso previsto
LIMITE_POR_PAIS = 2500                # imagens com defeito por país, para o treino caber em cerca de 1 hora
FRACAO_FUNDO = 0.10                   # imagens sem defeito, que ensinam o modelo a ficar quieto
EPOCAS = 50
PACIENCIA = 12                        # para antes se a validação não melhorar por 12 épocas
TAMANHO = 640
LOTE = 32
MODELO_BASE = "yolo11n.pt"            # pré-treinado no COCO, ponto de partida do fine-tuning
SEMENTE = 42

RAIZ = Path.cwd() / "cicloalerta"
ORIGEM = RAIZ / "rdd2022"
DESTINO = RAIZ / "via_yolo"
EXECUCOES = RAIZ / "execucoes"
ORIGEM.mkdir(parents=True, exist_ok=True)
EXECUCOES.mkdir(parents=True, exist_ok=True)
"""
    ),
    markdown("## 3. Código do projeto\n\nOs dois arquivos abaixo são cópias de `cicloalerta/` no repositório."),
    embutir("preparar_dados.py"),
    embutir("detector.py"),
    markdown(
        """
## 4. Baixar a base RDD2022

Os links oficiais responderam 403 em 04/11/2026, então usamos dois espelhos no Hugging Face.
As imagens e as trincas (D00, D10 e D20) vêm de `dronefreak/RDD2022`. A classe 3 desse espelho
marca faixas e sinalização, e não buraco, por isso os buracos (D40) vêm de `edwrdc/RDD2022`.
Só entram imagens presentes nos dois. Licença da base: CC BY-SA 4.0.
"""
    ),
    codigo(
        """
import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict

import cv2
import numpy as np

import preparar_dados

REPO_IMAGENS = "dronefreak/RDD2022"
REVISAO_IMAGENS = "72b01f94c2e5bfe430861afd73995f2d412b8855"
REPO_BURACOS = "edwrdc/RDD2022"
REVISAO_BURACOS = "d89672227027116a551aa3bfcc69efdf658b9579"
BASE_IMAGENS = f"https://huggingface.co/datasets/{REPO_IMAGENS}/resolve/{REVISAO_IMAGENS}"
TRINCAS = {0: "D00", 1: "D10", 2: "D20"}  # a categoria 3 do espelho não é buraco e fica de fora
CLASSE_BURACO = "2"                       # em edwrdc/RDD2022: 0 line_crack, 1 group_crack, 2 pothole


def ler(url: str, tentativas: int = 10) -> bytes:
    \"\"\"Baixa um arquivo e espera quando o Hugging Face pede para ir mais devagar.\"\"\"
    for tentativa in range(1, tentativas + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resposta:
                return resposta.read()
        except urllib.error.HTTPError as erro:
            if erro.code not in (429, 500, 502, 503, 504) or tentativa == tentativas:
                raise
            time.sleep(int(erro.headers.get("Retry-After") or 5 * tentativa))
        except (urllib.error.URLError, TimeoutError):
            if tentativa == tentativas:
                raise
            time.sleep(5 * tentativa)


pasta_buracos = RAIZ / "rotulos_buracos"
if not pasta_buracos.exists():
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"https://huggingface.co/datasets/{REPO_BURACOS}", str(pasta_buracos)],
        check=True,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
    )
versao = subprocess.run(["git", "-C", str(pasta_buracos), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
if versao and versao != REVISAO_BURACOS:
    print(f"Aviso: {REPO_BURACOS} mudou. Esperado {REVISAO_BURACOS}, obtido {versao}.")

buracos = {}
for arquivo in (pasta_buracos / "labels").glob("*/*/*.txt"):
    if arquivo.parts[-3] != "test" and arquivo.parent.name in PAISES:
        linhas = [linha.split() for linha in arquivo.read_text().splitlines()]
        buracos[arquivo.stem] = [tuple(map(float, l[1:5])) for l in linhas if l[:1] == [CLASSE_BURACO]]

info = json.loads(ler(f"https://huggingface.co/api/datasets/{REPO_IMAGENS}/revision/{REVISAO_IMAGENS}"))
arquivos = [s["rfilename"] for s in info["siblings"]]
caminhos = {Path(a).stem: a for a in arquivos if a.endswith(".jpg") and "/images/" in a}
trincas = {}
for arquivo in (a for a in arquivos if a.endswith("metadata.jsonl")):
    for linha in ler(f"{BASE_IMAGENS}/{arquivo}").decode("utf-8").splitlines():
        item = json.loads(linha)
        objetos = item["objects"]
        trincas[Path(item["file_name"]).stem] = [
            (TRINCAS[c], *caixa) for caixa, c in zip(objetos["bbox"], objetos["categories"]) if c in TRINCAS
        ]

por_pais = defaultdict(list)
for nome in sorted(buracos.keys() & trincas.keys() & caminhos.keys()):
    pais = re.sub(r"_\\d+$", "", nome)
    caixas = [(preparar_dados.ID_RACHADURA, 0.0, 0.0, 0.0, 0.0)] * len(trincas[nome])
    caixas += [(preparar_dados.ID_BURACO, 0.0, 0.0, 0.0, 0.0)] * len(buracos[nome])
    por_pais[pais].append(preparar_dados.Amostra(nome, pais, Path(nome), caixas))
selecionadas = [a for pais in PAISES for a in preparar_dados.selecionar(por_pais[pais], LIMITE_POR_PAIS, FRACAO_FUNDO)]
print({pais: len(lista) for pais, lista in por_pais.items()}, "selecionadas:", len(selecionadas))


def baixar_amostra(amostra) -> None:
    \"\"\"Grava imagem e XML no layout do RDD2022, para o preparo seguir igual ao da base oficial.\"\"\"
    pasta = ORIGEM / amostra.pais / "train"
    xml = pasta / "annotations" / "xmls" / f"{amostra.nome}.xml"
    if xml.exists():
        return
    bruto = ler(f"{BASE_IMAGENS}/{caminhos[amostra.nome]}")
    altura, largura = cv2.imdecode(np.frombuffer(bruto, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    objetos = [(r, x, y, x + l, y + a) for r, x, y, l, a in trincas[amostra.nome]]
    objetos += [
        ("D40", (cx - l / 2) * largura, (cy - a / 2) * altura, (cx + l / 2) * largura, (cy + a / 2) * altura)
        for cx, cy, l, a in buracos[amostra.nome]
    ]
    corpo = "".join(
        f"<object><name>{r}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
        for r, x1, y1, x2, y2 in objetos
    )
    (pasta / "images").mkdir(parents=True, exist_ok=True)
    xml.parent.mkdir(parents=True, exist_ok=True)
    (pasta / "images" / f"{amostra.nome}.jpg").write_bytes(bruto)
    xml.write_text(
        f"<annotation><size><width>{largura}</width><height>{altura}</height></size>{corpo}</annotation>",
        encoding="utf-8",
    )


inicio = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    for feitas, _ in enumerate(executor.map(baixar_amostra, selecionadas), 1):
        if feitas % 500 == 0:
            print(f"{feitas} de {len(selecionadas)} imagens em {time.time() - inicio:.0f} s", flush=True)
print(f"Base pronta em {time.time() - inicio:.0f} s")
"""
    ),
    markdown("## 5. Converter para YOLO e dividir em treino, validação e teste"),
    codigo(
        """
import json

import preparar_dados

estatisticas = preparar_dados.preparar(ORIGEM, DESTINO, LIMITE_POR_PAIS, FRACAO_FUNDO)
print(f"{'partição':<10}{'imagens':>9}{'buraco':>9}{'rachadura':>11}{'fundo':>7}")
for nome, linha in estatisticas["particoes"].items():
    print(f"{nome:<10}{linha['imagens']:>9}{linha['buraco']:>9}{linha['rachadura']:>11}{linha['fundo']:>7}")
print("Descartes:", estatisticas["descartes"] or "nenhum")
print("Rótulos ignorados:", estatisticas["rotulos_ignorados"] or "nenhum")

for classe in preparar_dados.CLASSES:
    if estatisticas["particoes"]["test"][classe] < 30:
        raise RuntimeError(f"Poucos exemplos de {classe} no teste. A avaliação não seria confiável.")
"""
    ),
    markdown("Confira se as caixas caem em cima dos defeitos. Vermelho é buraco, laranja é rachadura."),
    codigo(
        """
import random

import cv2
import matplotlib.pyplot as plt

imagens = sorted((DESTINO / "images" / "train").glob("*.jpg"))
random.Random(SEMENTE).shuffle(imagens)
com_rotulo = [p for p in imagens if (DESTINO / "labels" / "train" / f"{p.stem}.txt").stat().st_size > 0][:6]

figura, eixos = plt.subplots(2, 3, figsize=(14, 9))
for eixo, caminho in zip(eixos.flat, com_rotulo):
    quadro = cv2.cvtColor(cv2.imread(str(caminho)), cv2.COLOR_BGR2RGB)
    altura, largura = quadro.shape[:2]
    for linha in (DESTINO / "labels" / "train" / f"{caminho.stem}.txt").read_text().splitlines():
        classe, cx, cy, l, a = map(float, linha.split())
        p1 = (int((cx - l / 2) * largura), int((cy - a / 2) * altura))
        p2 = (int((cx + l / 2) * largura), int((cy + a / 2) * altura))
        cv2.rectangle(quadro, p1, p2, (235, 40, 40) if classe == 0 else (255, 170, 0), 3)
    eixo.imshow(quadro)
    eixo.set_title(caminho.stem)
    eixo.axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 6. Treinar\n\nÉ a etapa mais longa. O mAP aparece no fim de cada época."),
    codigo(
        """
from ultralytics import YOLO

inicio_treino = time.time()
modelo = YOLO(MODELO_BASE)
modelo.train(
    data=str(DESTINO / "dados.yaml"),
    epochs=EPOCAS,
    patience=PACIENCIA,
    imgsz=TAMANHO,
    batch=LOTE,
    workers=2,
    seed=SEMENTE,
    cos_lr=True,
    close_mosaic=10,
    flipud=0.0,   # a rua nunca aparece de cabeça para baixo
    fliplr=0.5,
    degrees=0.0,
    project=str(EXECUCOES),
    name="treino",
    exist_ok=True,
)
tempo_treino_min = (time.time() - inicio_treino) / 60
melhor_pesos = Path(modelo.trainer.best)
print(f"Treino concluído em {tempo_treino_min:.1f} min. Pesos: {melhor_pesos}")
"""
    ),
    markdown(
        """
## 7. Avaliar

Os limiares de confiança saem da **validação**, no ponto de maior F1 de cada classe. O **teste**
só é usado depois, para não contaminar a medida. A avaliação por país mostra se o modelo
funciona pior em algum lugar.
"""
    ),
    codigo(
        """
import numpy as np

avaliador = YOLO(str(melhor_pesos))
OPCOES_VAL = dict(imgsz=TAMANHO, batch=LOTE, workers=2, project=str(EXECUCOES), exist_ok=True, verbose=False)


def resumir_geral(metricas) -> dict:
    caixa = metricas.box
    return {
        "precisao": float(caixa.mp),
        "revocacao": float(caixa.mr),
        "map50": float(caixa.map50),
        "map50_95": float(caixa.map),
    }


def resumir_classes(metricas, nomes: dict, limiares: dict | None = None) -> dict:
    caixa = metricas.box
    saida = {}
    for indice, classe_id in enumerate(caixa.ap_class_index):
        nome = nomes[int(classe_id)]
        precisao, revocacao, ap50, ap = caixa.class_result(indice)
        linha = {
            "instancias": int(metricas.nt_per_class[int(classe_id)]),
            "precisao": float(precisao),
            "revocacao": float(revocacao),
            "ap50": float(ap50),
            "ap50_95": float(ap),
        }
        if limiares:
            ponto = int(np.searchsorted(caixa.px, limiares[nome]))
            ponto = min(ponto, len(caixa.px) - 1)
            linha["limiar"] = limiares[nome]
            linha["precisao_no_limiar"] = float(caixa.p_curve[indice][ponto])
            linha["revocacao_no_limiar"] = float(caixa.r_curve[indice][ponto])
        saida[nome] = linha
    return saida


validacao = avaliador.val(data=str(DESTINO / "dados.yaml"), split="val", name="validacao", plots=False, **OPCOES_VAL)
NOMES = {int(k): v for k, v in validacao.names.items()}
limiares = {
    NOMES[int(classe_id)]: round(float(validacao.box.px[validacao.box.f1_curve[indice].argmax()]), 3)
    for indice, classe_id in enumerate(validacao.box.ap_class_index)
}
print("Limiares calibrados na validação:", limiares)

teste = avaliador.val(data=str(DESTINO / "dados.yaml"), split="test", name="teste", plots=True, **OPCOES_VAL)
por_pais = {}
for pasta in sorted((DESTINO / "paises").iterdir()):
    resultado = avaliador.val(
        data=str(pasta / "dados.yaml"), split="test", name=f"teste_{pasta.name}", plots=False, **OPCOES_VAL
    )
    por_pais[pasta.name] = {"geral": resumir_geral(resultado), "por_classe": resumir_classes(resultado, NOMES)}

print("Teste:", resumir_geral(teste))
for pais, dados in por_pais.items():
    print(f"  {pais}: mAP50 = {dados['geral']['map50']:.3f}")
"""
    ),
    markdown("## 8. Exportar para ONNX e conferir\n\nO aplicativo roda o ONNX, sem PyTorch. Por isso comparo as duas versões."),
    codigo(
        """
import detector as detector_onnx

caminho_onnx = Path(avaliador.export(format="onnx", imgsz=TAMANHO, opset=17, simplify=True, dynamic=False))
onnx_final = RAIZ / "cicloalerta.onnx"
shutil.copy2(caminho_onnx, onnx_final)

detector_app = detector_onnx.DetectorONNX(onnx_final, limiares=limiares)
imagens_teste = sorted((DESTINO / "images" / "test").glob("*.jpg"))


def iou_simples(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersecao = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersecao
    return intersecao / uniao if uniao > 0 else 0.0


# Mesma imagem nas duas versões: cada caixa do PyTorch precisa existir no aplicativo.
iguais = total_caixas = 0
for caminho in imagens_teste[:100]:
    quadro = cv2.imread(str(caminho))
    resultado = avaliador.predict(quadro, conf=min(limiares.values()), iou=0.5, imgsz=TAMANHO, verbose=False)[0]
    caixas = resultado.boxes
    do_app = detector_app.detectar(quadro)
    for xyxy, confianca, classe in zip(caixas.xyxy.tolist(), caixas.conf.tolist(), caixas.cls.tolist()):
        if confianca < limiares.get(NOMES[int(classe)], 0):
            continue
        total_caixas += 1
        iguais += any(d.classe_id == int(classe) and iou_simples(d.caixa, xyxy) > 0.8 for d in do_app)
concordancia = iguais / total_caixas if total_caixas else None
print(f"Caixas iguais entre PyTorch e aplicativo: {iguais} de {total_caixas}")

latencias = []
for caminho in imagens_teste[:100]:
    quadro = cv2.imread(str(caminho))
    inicio = time.perf_counter()
    detector_app.detectar(quadro)
    latencias.append((time.perf_counter() - inicio) * 1000)
latencia_ms = float(np.median(latencias[5:])) if len(latencias) > 5 else float(np.median(latencias))
print(f"Latência em CPU do Colab: {latencia_ms:.0f} ms por quadro ({1000 / latencia_ms:.1f} FPS)")
"""
    ),
    markdown("## 9. Empacotar e baixar os resultados"),
    codigo(
        """
import csv
import platform

import torch
import ultralytics

pacote = RAIZ / "pacote"
if pacote.exists():
    shutil.rmtree(pacote)
(pacote / "modelos").mkdir(parents=True)
shutil.copy2(onnx_final, pacote / "modelos" / "cicloalerta.onnx")
shutil.copy2(melhor_pesos, pacote / "modelos" / "best.pt")
(pacote / "modelos" / "limiares.json").write_text(json.dumps(limiares, indent=2), encoding="utf-8")
shutil.copy2(DESTINO / "estatisticas.json", pacote / "estatisticas_dados.json")

for nome in ("treino", "teste"):
    pasta = pacote / nome
    pasta.mkdir()
    for arquivo in (EXECUCOES / nome).iterdir():
        if arquivo.suffix.lower() in {".png", ".jpg", ".csv", ".yaml"}:
            shutil.copy2(arquivo, pasta / arquivo.name)

# Amostras do teste para gerar os exemplos do relatório sem precisar baixar a base de novo.
com_buraco, com_rachadura, sem_defeito = [], [], []
for caminho in imagens_teste:
    rotulo = (DESTINO / "labels" / "test" / f"{caminho.stem}.txt").read_text()
    classes = {linha.split()[0] for linha in rotulo.splitlines()}
    (com_buraco if "0" in classes else com_rachadura if "1" in classes else sem_defeito).append(caminho)
for caminho in com_buraco[:30] + com_rachadura[:20] + sem_defeito[:10]:
    for subpasta, origem in (("images", caminho), ("labels", DESTINO / "labels" / "test" / f"{caminho.stem}.txt")):
        (pacote / "amostras_teste" / subpasta).mkdir(parents=True, exist_ok=True)
        shutil.copy2(origem, pacote / "amostras_teste" / subpasta / origem.name)

with (EXECUCOES / "treino" / "results.csv").open(encoding="utf-8") as arquivo:
    historico = [{k.strip(): v for k, v in linha.items()} for linha in csv.DictReader(arquivo)]
melhor_linha = max(historico, key=lambda linha: float(linha["metrics/mAP50-95(B)"]))

metricas = {
    "ambiente": {
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
    "configuracao": {
        "paises": PAISES,
        "limite_por_pais": LIMITE_POR_PAIS,
        "fracao_fundo": FRACAO_FUNDO,
        "epocas_maximas": EPOCAS,
        "paciencia": PACIENCIA,
        "tamanho": TAMANHO,
        "lote": LOTE,
        "modelo_base": MODELO_BASE,
        "semente": SEMENTE,
    },
    "treino": {
        "tempo_min": round(tempo_treino_min, 1),
        "epocas_executadas": len(historico),
        "melhor_epoca": int(float(melhor_linha["epoch"])),
    },
    "limiares": limiares,
    "validacao": {"geral": resumir_geral(validacao), "por_classe": resumir_classes(validacao, NOMES)},
    "teste": {
        "geral": resumir_geral(teste),
        "por_classe": resumir_classes(teste, NOMES, limiares),
        "velocidade_ms": {k: float(v) for k, v in teste.speed.items()},
    },
    "por_pais": por_pais,
    "onnx": {
        "caixas_comparadas": total_caixas,
        "concordancia_com_pytorch": concordancia,
        "latencia_cpu_colab_ms": round(latencia_ms, 1),
    },
}
(pacote / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")

arquivo_final = Path(shutil.make_archive(str(RAIZ / "resultados"), "zip", pacote))
print(f"Pacote pronto: {arquivo_final} ({arquivo_final.stat().st_size / 1e6:.0f} MB)")
print(json.dumps(metricas["teste"]["geral"], indent=2))

try:
    from google.colab import files

    files.download(str(arquivo_final))
except Exception as erro:
    print(f"Baixe pelo ícone de pasta: {arquivo_final} ({erro})")
"""
    ),
    markdown("## Pronto\n\nEnvie o `resultados.zip` no chat. O relatório e os exemplos saem dele."),
]


def main() -> None:
    caderno = {
        "cells": CELULAS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(json.dumps(caderno, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{SAIDA.relative_to(RAIZ)}: {len(CELULAS)} células")


if __name__ == "__main__":
    main()
