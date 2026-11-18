#!/usr/bin/env python3
"""Converte o RDD2022 (Pascal VOC) para YOLO, com duas classes: buraco e rachadura.

As imagens do RDD2022 vêm de veículos em movimento, então quadros próximos podem
ser quase iguais. Para isso não vazar entre treino e teste, a divisão é feita por
blocos de nomes consecutivos, e não imagem a imagem. A divisão usa hash, logo é a
mesma em qualquer máquina.

Uso:
    python preparar_dados.py --origem dados/rdd2022 --destino dados/via_yolo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CLASSES = ("buraco", "rachadura")
ID_BURACO = 0
ID_RACHADURA = 1

# D00, D10 e D20 descrevem o formato da trinca. Na triagem, o que pesa é separar
# trinca de buraco, e juntar as três categorias dá mais exemplos ao modelo.
MAPA_ROTULOS = {"D40": ID_BURACO, "D00": ID_RACHADURA, "D10": ID_RACHADURA, "D20": ID_RACHADURA}

TAMANHO_BLOCO = 50
PARTICOES = ("train", "val", "test")
_PADRAO_NOME = re.compile(r"^(?P<prefixo>.*?)_?(?P<numero>\d+)$")


@dataclass
class Amostra:
    """Uma imagem anotada, com as caixas já no formato YOLO."""

    nome: str
    pais: str
    imagem: Path
    caixas: list[tuple[int, float, float, float, float]] = field(default_factory=list)

    def tem_classe(self, classe_id: int) -> bool:
        return any(caixa[0] == classe_id for caixa in self.caixas)


def ler_anotacao(caminho_xml: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """Lê largura, altura e os objetos (rótulo, xmin, ymin, xmax, ymax) de um XML VOC."""
    raiz = ET.parse(caminho_xml).getroot()
    tamanho = raiz.find("size")
    largura = int(float(tamanho.findtext("width", "0"))) if tamanho is not None else 0
    altura = int(float(tamanho.findtext("height", "0"))) if tamanho is not None else 0
    objetos = []
    for objeto in raiz.iter("object"):
        rotulo = (objeto.findtext("name") or "").strip()
        caixa = objeto.find("bndbox")
        if caixa is None or not rotulo:
            continue
        try:
            xmin, ymin, xmax, ymax = (float(caixa.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            continue
        objetos.append((rotulo, xmin, ymin, xmax, ymax))
    return largura, altura, objetos


def para_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float, largura: int, altura: int
) -> tuple[float, float, float, float] | None:
    """Converte para (cx, cy, l, a) normalizados. Devolve None se a caixa for inválida."""
    x1, x2 = sorted((max(0.0, min(xmin, largura)), max(0.0, min(xmax, largura))))
    y1, y2 = sorted((max(0.0, min(ymin, altura)), max(0.0, min(ymax, altura))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (
        (x1 + x2) / 2 / largura,
        (y1 + y2) / 2 / altura,
        (x2 - x1) / largura,
        (y2 - y1) / altura,
    )


def hash_estavel(texto: str) -> int:
    return int(hashlib.sha1(texto.encode("utf-8")).hexdigest(), 16)


def bloco_da_imagem(nome: str) -> str:
    """Agrupa nomes consecutivos: India_000123 e India_000149 caem no mesmo bloco."""
    achado = _PADRAO_NOME.match(nome)
    if achado is None:
        return nome
    return f"{achado['prefixo']}#{int(achado['numero']) // TAMANHO_BLOCO}"


def particao_do_bloco(pais: str, bloco: str, frac_val: float = 0.1, frac_teste: float = 0.1) -> str:
    valor = hash_estavel(f"{pais}:{bloco}") % 10_000 / 10_000
    if valor < frac_teste:
        return "test"
    if valor < frac_teste + frac_val:
        return "val"
    return "train"


def _pasta_de_treino(caminho_xml: Path) -> Path | None:
    """Sobe na árvore até a pasta train, que define o país e onde estão as imagens."""
    return next((pasta for pasta in caminho_xml.parents if pasta.name == "train"), None)


def coletar_amostras(origem: Path, estatisticas: dict) -> list[Amostra]:
    """Acha os XML sob <pais>/train, aceitando annotations/ ou annotations/xmls/."""
    amostras = []
    for caminho_xml in sorted(origem.rglob("*.xml")):
        pasta_treino = _pasta_de_treino(caminho_xml)
        if pasta_treino is None:
            continue
        pais = pasta_treino.parent.name
        imagem = pasta_treino / "images" / f"{caminho_xml.stem}.jpg"
        if not imagem.exists():
            estatisticas["descartes"]["imagem_ausente"] += 1
            continue
        try:
            largura, altura, objetos = ler_anotacao(caminho_xml)
        except ET.ParseError:
            estatisticas["descartes"]["xml_invalido"] += 1
            continue
        if largura <= 0 or altura <= 0:
            estatisticas["descartes"]["tamanho_ausente"] += 1
            continue
        amostra = Amostra(caminho_xml.stem, pais, imagem)
        for rotulo, xmin, ymin, xmax, ymax in objetos:
            if rotulo not in MAPA_ROTULOS:
                estatisticas["rotulos_ignorados"][rotulo] += 1
                continue
            caixa = para_yolo(xmin, ymin, xmax, ymax, largura, altura)
            if caixa is None:
                estatisticas["descartes"]["caixa_invalida"] += 1
                continue
            estatisticas["rotulos_originais"][rotulo] += 1
            amostra.caixas.append((MAPA_ROTULOS[rotulo], *caixa))
        amostras.append(amostra)
    return amostras


def selecionar(amostras: list[Amostra], limite: int | None, frac_fundo: float) -> list[Amostra]:
    """Limita um país priorizando buracos, a classe mais rara, e guarda um pouco de fundo."""
    ordenadas = sorted(amostras, key=lambda a: hash_estavel(a.nome))
    com_buraco = [a for a in ordenadas if a.tem_classe(ID_BURACO)]
    so_rachadura = [a for a in ordenadas if a.caixas and not a.tem_classe(ID_BURACO)]
    fundo = [a for a in ordenadas if not a.caixas]
    positivas = com_buraco + so_rachadura
    if limite is not None:
        positivas = positivas[:limite]
    quantidade_fundo = round(len(positivas) * frac_fundo / (1 - frac_fundo))
    return positivas + fundo[:quantidade_fundo]


def _ligar(origem: Path, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        destino.unlink()
    try:
        os.link(origem, destino)
    except OSError:
        shutil.copy2(origem, destino)


def _rotulo_em_texto(amostra: Amostra) -> str:
    linhas = [f"{c} {cx:.6f} {cy:.6f} {l:.6f} {a:.6f}" for c, cx, cy, l, a in amostra.caixas]
    return "\n".join(linhas) + ("\n" if linhas else "")


def _yaml(pasta: Path, treino: str, validacao: str, teste: str) -> str:
    nomes = "\n".join(f"  {i}: {nome}" for i, nome in enumerate(CLASSES))
    return (
        f"path: {pasta.resolve().as_posix()}\n"
        f"train: {treino}\nval: {validacao}\ntest: {teste}\n"
        f"names:\n{nomes}\n"
    )


def _novo_contador() -> dict:
    return {"imagens": 0, "fundo": 0, **{classe: 0 for classe in CLASSES}}


def _somar(contador: dict, amostra: Amostra) -> None:
    contador["imagens"] += 1
    contador["fundo"] += 0 if amostra.caixas else 1
    for caixa in amostra.caixas:
        contador[CLASSES[caixa[0]]] += 1


def preparar(
    origem: Path,
    destino: Path,
    limite_por_pais: int | None = None,
    frac_fundo: float = 0.1,
    frac_val: float = 0.1,
    frac_teste: float = 0.1,
) -> dict:
    estatisticas: dict = {
        "classes": list(CLASSES),
        "mapa_rotulos": MAPA_ROTULOS,
        "tamanho_bloco": TAMANHO_BLOCO,
        "rotulos_originais": Counter(),
        "rotulos_ignorados": Counter(),
        "descartes": Counter(),
    }
    amostras = coletar_amostras(origem, estatisticas)
    if not amostras:
        raise SystemExit(f"Nenhuma anotação encontrada em {origem}. Confira a extração dos arquivos.")

    if destino.exists():
        shutil.rmtree(destino)

    por_pais: dict[str, list[Amostra]] = defaultdict(list)
    for amostra in amostras:
        por_pais[amostra.pais].append(amostra)

    contagem_particao = {p: _novo_contador() for p in PARTICOES}
    contagem_pais: dict[str, dict] = {}
    teste_por_pais: dict[str, list[Amostra]] = defaultdict(list)

    for pais in sorted(por_pais):
        contagem_pais[pais] = {"disponiveis": len(por_pais[pais]), **{p: _novo_contador() for p in PARTICOES}}
        for amostra in selecionar(por_pais[pais], limite_por_pais, frac_fundo):
            bloco = bloco_da_imagem(amostra.nome)
            particao = particao_do_bloco(amostra.pais, bloco, frac_val, frac_teste)
            _ligar(amostra.imagem, destino / "images" / particao / f"{amostra.nome}.jpg")
            rotulo = destino / "labels" / particao / f"{amostra.nome}.txt"
            rotulo.parent.mkdir(parents=True, exist_ok=True)
            rotulo.write_text(_rotulo_em_texto(amostra), encoding="utf-8")
            _somar(contagem_particao[particao], amostra)
            _somar(contagem_pais[pais][particao], amostra)
            if particao == "test":
                teste_por_pais[pais].append(amostra)

    (destino / "dados.yaml").write_text(
        _yaml(destino, "images/train", "images/val", "images/test"), encoding="utf-8"
    )

    # Uma pasta de teste por país, para medir se o modelo vai pior em algum deles.
    # O YOLO exige as chaves train e val no YAML, então elas apontam para o próprio teste.
    for pais, lista in teste_por_pais.items():
        pasta = destino / "paises" / pais
        for amostra in lista:
            _ligar(amostra.imagem, pasta / "images" / "test" / f"{amostra.nome}.jpg")
            rotulo = pasta / "labels" / "test" / f"{amostra.nome}.txt"
            rotulo.parent.mkdir(parents=True, exist_ok=True)
            rotulo.write_text(_rotulo_em_texto(amostra), encoding="utf-8")
        (pasta / "dados.yaml").write_text(
            _yaml(pasta, "images/test", "images/test", "images/test"), encoding="utf-8"
        )

    estatisticas.update(particoes=contagem_particao, por_pais=contagem_pais)
    for chave in ("rotulos_originais", "rotulos_ignorados", "descartes"):
        estatisticas[chave] = dict(sorted(estatisticas[chave].items()))
    (destino / "estatisticas.json").write_text(
        json.dumps(estatisticas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return estatisticas


def main() -> int:
    analisador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analisador.add_argument("--origem", type=Path, required=True, help="pasta com os países do RDD2022 extraídos")
    analisador.add_argument("--destino", type=Path, required=True, help="pasta de saída no formato YOLO")
    analisador.add_argument("--limite-por-pais", type=int, default=None, help="máximo de imagens com defeito por país")
    analisador.add_argument("--frac-fundo", type=float, default=0.1, help="fração de imagens sem defeito")
    args = analisador.parse_args()

    estatisticas = preparar(args.origem, args.destino, args.limite_por_pais, args.frac_fundo)
    print(json.dumps(estatisticas["particoes"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
