#!/usr/bin/env python3
"""Ensaia o notebook de ponta a ponta sem GPU e sem PyTorch.

Executa as células na ordem, com arquivos no formato dos espelhos do RDD2022 e um YOLO
de mentira que imita a API real. Só a rede, o treino e o torch são simulados: conversão,
seleção, cálculo de limiares, ONNX Runtime e empacotamento rodam de verdade. Serve
para pegar erro de nome, de caminho e de ordem antes de gastar uma hora de GPU.

Uso: python ferramentas/ensaiar_notebook.py caminho/para/yolo11n.onnx
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

RAIZ_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_REPO))
from tests.conftest import criar_amostra  # noqa: E402


class CaixaMetricas:
    def __init__(self, classes=(0, 1)):
        self.ap_class_index = np.array(classes)
        self.mp, self.mr, self.map50, self.map = 0.6, 0.5, 0.55, 0.3
        self.px = np.linspace(0, 1, 1000)
        pico = np.array([0.31, 0.27])[: len(classes)]
        self.f1_curve = np.stack([np.exp(-((self.px - p) ** 2) / 0.02) for p in pico])
        self.p_curve = np.stack([self.px for _ in pico])
        self.r_curve = np.stack([1 - self.px for _ in pico])

    def class_result(self, indice):
        return 0.6 + indice / 100, 0.5, 0.55, 0.3


class MetricasFalsas:
    names = {0: "buraco", 1: "rachadura"}
    speed = {"preprocess": 0.2, "inference": 3.1, "loss": 0.0, "postprocess": 1.4}
    nt_per_class = np.array([40, 90])

    def __init__(self):
        self.box = CaixaMetricas()


class YOLOFalso:
    def __init__(self, pesos):
        self.pesos = pesos
        self.trainer = None

    def train(self, **opcoes):
        assert Path(opcoes["data"]).exists()
        pasta = Path(opcoes["project"]) / opcoes["name"]
        (pasta / "weights").mkdir(parents=True, exist_ok=True)
        (pasta / "weights" / "best.pt").write_bytes(b"pesos")
        (pasta / "results.png").write_bytes(b"png")
        (pasta / "args.yaml").write_text("epochs: 2\n", encoding="utf-8")
        # O CSV real tem espaços nos cabeçalhos em algumas versões.
        (pasta / "results.csv").write_text(
            "epoch, metrics/mAP50-95(B)\n1, 0.2\n2, 0.3\n3, 0.25\n", encoding="utf-8"
        )
        self.trainer = types.SimpleNamespace(best=pasta / "weights" / "best.pt", save_dir=pasta)

    def val(self, **opcoes):
        assert opcoes["split"] in {"val", "test"}
        assert Path(opcoes["data"]).exists()
        pasta = Path(opcoes["project"]) / opcoes["name"]
        pasta.mkdir(parents=True, exist_ok=True)
        (pasta / "BoxPR_curve.png").write_bytes(b"png")
        return MetricasFalsas()

    def export(self, **opcoes):
        assert opcoes["format"] == "onnx"
        return os.environ["ENSAIO_ONNX"]

    def predict(self, quadro, **opcoes):
        caixas = types.SimpleNamespace(
            xyxy=np.array([[0.0, 0.0, 10.0, 10.0]]), conf=np.array([0.9]), cls=np.array([0.0])
        )
        return [types.SimpleNamespace(boxes=caixas)]


def instalar_stubs() -> None:
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = YOLOFalso
    ultralytics.__version__ = "ensaio"
    sys.modules["ultralytics"] = ultralytics

    torch = types.ModuleType("torch")
    torch.__version__ = "ensaio"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda _: "ensaio")
    sys.modules["torch"] = torch

    baixados = []
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=baixados.append)
    google = types.ModuleType("google")
    google.colab = colab
    sys.modules.update({"google": google, "google.colab": colab})
    instalar_stubs.baixados = baixados


REVISAO_BURACOS = "d89672227027116a551aa3bfcc69efdf658b9579"


def criar_espelhos(pasta: Path) -> dict:
    """Monta o que os dois espelhos do Hugging Face serviriam, no formato real deles."""
    rotulos = pasta / "rotulos"
    linhas_metadados, jpg = [], cv2.imencode(".jpg", np.full((600, 600, 3), 90, dtype=np.uint8))[1].tobytes()
    caminhos = []
    for pais, deslocamento in (("India", 0), ("Japan", 3), ("Czech", 11)):
        for i in range(500):
            nome = f"{pais}_{i + deslocamento:06d}"
            caminhos.append(f"data/images/train/shard_000/{nome}.jpg")
            trincas = {"bbox": [[10, 20, 190, 60], [5, 5, 40, 40]], "categories": [0, 3]} if i % 3 == 1 else {"bbox": [[5, 5, 40, 40]], "categories": [3]}
            linhas_metadados.append(json.dumps({"file_name": f"{nome}.jpg", "objects": trincas}))
            if i % 50 == 49:
                continue  # imagem sem arquivo de rótulo: não pode entrar, pois o buraco dela é desconhecido
            texto = "2 0.3 0.3 0.3 0.25\n" if i % 3 == 0 else "0 0.5 0.5 0.2 0.2\n"
            arquivo = rotulos / "labels" / ("val" if i % 5 == 0 else "train") / pais / f"{nome}.txt"
            arquivo.parent.mkdir(parents=True, exist_ok=True)
            arquivo.write_text(texto, encoding="utf-8")
        teste = rotulos / "labels" / "test" / pais / f"{pais}_900000.txt"
        teste.parent.mkdir(parents=True, exist_ok=True)
        teste.write_text("", encoding="utf-8")
    metadados = "\n".join(linhas_metadados).encode("utf-8")
    siblings = [{"rfilename": c} for c in [".gitattributes", "data/images/train/shard_000/metadata.jsonl", *caminhos]]
    return {"rotulos": rotulos, "metadados": metadados, "jpg": jpg, "api": json.dumps({"siblings": siblings}).encode(), "lento": [True]}


class RespostaFalsa(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def urlopen_falso(espelhos: dict):
    def abrir(url, timeout=None):
        assert url.startswith("https://huggingface.co/"), url
        if "/api/datasets/dronefreak/RDD2022/revision/" in url:
            return RespostaFalsa(espelhos["api"])
        if url.endswith("metadata.jsonl"):
            return RespostaFalsa(espelhos["metadados"])
        if url.endswith(".jpg"):
            if espelhos["lento"][0]:  # o primeiro pedido leva 429, como no limite real
                espelhos["lento"][0] = False
                raise urllib.error.HTTPError(url, 429, "Too Many Requests", {"Retry-After": "0"}, None)
            return RespostaFalsa(espelhos["jpg"])
        raise AssertionError(url)

    return abrir


def main() -> int:
    onnx = Path(sys.argv[1]).resolve()
    os.environ["ENSAIO_ONNX"] = str(onnx)
    caderno = json.loads((RAIZ_REPO / "notebooks" / "treino_cicloalerta_colab.ipynb").read_text(encoding="utf-8"))
    instalar_stubs()

    with tempfile.TemporaryDirectory() as temporario:
        trabalho = Path(temporario)
        espelhos = criar_espelhos(trabalho / "espelhos")
        area = trabalho / "colab"
        area.mkdir()
        os.chdir(area)
        sys.path.insert(0, str(area))

        run_real = subprocess.run

        def run_falso(comando, **opcoes):
            if comando[0] == "nvidia-smi" or comando[1:3] == ["-m", "pip"]:
                return subprocess.CompletedProcess(comando, 0)
            if comando[:2] == ["git", "clone"]:
                shutil.copytree(espelhos["rotulos"], comando[-1])
                return subprocess.CompletedProcess(comando, 0)
            if comando[0] == "git":
                return subprocess.CompletedProcess(comando, 0, stdout=REVISAO_BURACOS + "\n", stderr="")
            return run_real(comando, **opcoes)

        subprocess.run = run_falso
        urllib.request.urlopen = urlopen_falso(espelhos)
        which_real = shutil.which
        shutil.which = lambda nome: "/usr/bin/nvidia-smi" if nome == "nvidia-smi" else which_real(nome)

        ambiente: dict = {"__name__": "__main__"}
        codigos = [c for c in caderno["cells"] if c["cell_type"] == "code"]
        for numero, celula in enumerate(codigos, 1):
            fonte = "".join(celula["source"])
            if fonte.startswith("%%writefile"):
                nome, corpo = fonte.split("\n", 1)
                Path(nome.split()[1]).write_text(corpo, encoding="utf-8")
            else:
                exec(compile(fonte, f"celula_{numero}", "exec"), ambiente)
            print(f"célula {numero}/{len(codigos)} ok")

        final = area / "cicloalerta" / "resultados.zip"
        with zipfile.ZipFile(final) as pacote:
            nomes = pacote.namelist()
            metricas = json.loads(pacote.read("metricas.json"))
        for esperado in (
            "metricas.json",
            "estatisticas_dados.json",
            "modelos/cicloalerta.onnx",
            "modelos/best.pt",
            "modelos/limiares.json",
            "treino/results.csv",
            "teste/BoxPR_curve.png",
        ):
            assert esperado in nomes, esperado
        assert any(n.startswith("amostras_teste/images/") for n in nomes)
        assert any(n.startswith("amostras_teste/labels/") for n in nomes)
        assert instalar_stubs.baixados == [str(final)]
        assert metricas["limiares"] == {"buraco": 0.31, "rachadura": 0.27}, metricas["limiares"]
        assert set(metricas["por_pais"]) == {"India", "Japan", "Czech"}
        with zipfile.ZipFile(final) as pacote:
            dados = json.loads(pacote.read("estatisticas_dados.json"))
        assert set(dados["rotulos_originais"]) == {"D00", "D40"}, dados["rotulos_originais"]
        assert dados["rotulos_ignorados"] == {}, "a categoria 3 do espelho não pode chegar ao preparo"
        assert all(linha["fundo"] > 0 for linha in dados["particoes"].values())
        assert not espelhos["lento"][0], "o caso de 429 precisa ter sido exercitado"
        assert metricas["treino"] == {"tempo_min": metricas["treino"]["tempo_min"], "epocas_executadas": 3, "melhor_epoca": 2}
        ponto = metricas["teste"]["por_classe"]["buraco"]
        assert abs(ponto["precisao_no_limiar"] - 0.31) < 0.01 and abs(ponto["revocacao_no_limiar"] - 0.69) < 0.01
        if len(sys.argv) > 2:
            shutil.copy2(final, sys.argv[2])
        print(f"\nresultados.zip com {len(nomes)} arquivos")
        print(json.dumps(metricas["onnx"], indent=2))
        print(json.dumps(ambiente["estatisticas"]["particoes"], indent=None))
        os.chdir(RAIZ_REPO)
    print("Ensaio concluído.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
