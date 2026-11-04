from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from tests.conftest import criar_amostra, escrever_xml
from via_atenta.preparar_dados import (
    PARTICOES,
    bloco_da_imagem,
    ler_anotacao,
    para_yolo,
    particao_do_bloco,
    preparar,
)


def test_le_coordenadas_decimais(tmp_path):
    xml = tmp_path / "a.xml"
    escrever_xml(xml, [("D40", 10.5, 20.0, 110.25, 220.0)], largura=720, altura=720)
    assert ler_anotacao(xml) == (720, 720, [("D40", 10.5, 20.0, 110.25, 220.0)])


def test_ignora_objeto_sem_caixa(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text(
        "<annotation><size><width>10</width><height>10</height></size>"
        "<object><name>D00</name></object></annotation>",
        encoding="utf-8",
    )
    assert ler_anotacao(xml) == (10, 10, [])


def test_para_yolo_normaliza():
    assert para_yolo(100, 200, 300, 400, 600, 600) == pytest.approx((1 / 3, 0.5, 1 / 3, 1 / 3))


def test_para_yolo_corta_o_que_passa_da_imagem():
    assert para_yolo(-50, -50, 300, 300, 600, 600) == pytest.approx((0.25, 0.25, 0.5, 0.5))


def test_para_yolo_aceita_coordenadas_invertidas():
    assert para_yolo(300, 400, 100, 200, 600, 600) == para_yolo(100, 200, 300, 400, 600, 600)


def test_para_yolo_rejeita_caixa_degenerada():
    assert para_yolo(100, 100, 100.5, 300, 600, 600) is None


def test_nomes_consecutivos_caem_no_mesmo_bloco():
    assert bloco_da_imagem("India_000100") == bloco_da_imagem("India_000149")
    assert bloco_da_imagem("India_000149") != bloco_da_imagem("India_000150")


def test_prefixo_composto_faz_parte_do_bloco():
    assert bloco_da_imagem("China_MotorBike_000100") != bloco_da_imagem("China_Drone_000100")


def test_particao_e_deterministica():
    assert particao_do_bloco("India", "India#3") == particao_do_bloco("India", "India#3")


def test_preparacao_completa(rdd_falso, tmp_path):
    destino = tmp_path / "yolo"
    estatisticas = preparar(rdd_falso, destino)

    total = sum(estatisticas["particoes"][p]["imagens"] for p in PARTICOES)
    assert total > 500
    assert all(estatisticas["particoes"][p]["imagens"] > 0 for p in PARTICOES)
    assert estatisticas["rotulos_ignorados"] == {"D44": 240}
    assert not list(destino.rglob("India_900000*")), "imagem sem anotação não pode entrar"

    yaml = (destino / "dados.yaml").read_text(encoding="utf-8")
    assert "test: images/test" in yaml and "0: buraco" in yaml and "1: rachadura" in yaml

    for particao in PARTICOES:
        imagens = {p.stem for p in (destino / "images" / particao).glob("*.jpg")}
        rotulos = {p.stem for p in (destino / "labels" / particao).glob("*.txt")}
        assert imagens == rotulos


def test_nenhum_bloco_aparece_em_duas_particoes(rdd_falso, tmp_path):
    destino = tmp_path / "yolo"
    preparar(rdd_falso, destino)
    blocos = defaultdict(set)
    for particao in PARTICOES:
        for imagem in (destino / "images" / particao).glob("*.jpg"):
            blocos[bloco_da_imagem(imagem.stem)].add(particao)
    assert blocos and all(len(particoes) == 1 for particoes in blocos.values())


def test_classes_e_coordenadas_dos_rotulos(rdd_falso, tmp_path):
    destino = tmp_path / "yolo"
    preparar(rdd_falso, destino)
    rotulo = next(destino.rglob("labels/*/India_000000.txt"))
    assert rotulo.read_text(encoding="utf-8").strip() == "0 0.333333 0.291667 0.333333 0.250000"
    rotulo = next(destino.rglob("labels/*/India_000001.txt"))
    assert [linha.split()[0] for linha in rotulo.read_text(encoding="utf-8").splitlines()] == ["1", "1"]


def test_imagem_de_fundo_tem_rotulo_vazio(rdd_falso, tmp_path):
    destino = tmp_path / "yolo"
    estatisticas = preparar(rdd_falso, destino)
    fundos = sum(estatisticas["particoes"][p]["fundo"] for p in PARTICOES)
    vazios = [p for p in (destino / "labels").rglob("*.txt") if p.stat().st_size == 0]
    assert fundos == len(vazios) > 0


def test_fracao_de_fundo_e_respeitada(rdd_falso, tmp_path):
    estatisticas = preparar(rdd_falso, tmp_path / "yolo", frac_fundo=0.1)
    fundos = sum(estatisticas["particoes"][p]["fundo"] for p in PARTICOES)
    total = sum(estatisticas["particoes"][p]["imagens"] for p in PARTICOES)
    assert fundos / total == pytest.approx(0.1, abs=0.01)


def test_limite_prioriza_buracos(tmp_path):
    raiz = tmp_path / "rdd"
    for i in range(100):
        criar_amostra(raiz, "Japan", i, [("D00", 10, 10, 100, 100)])
    for i in range(100, 110):
        criar_amostra(raiz, "Japan", i, [("D40", 10, 10, 100, 100)])
    estatisticas = preparar(raiz, tmp_path / "yolo", limite_por_pais=20, frac_fundo=0.1)
    assert sum(estatisticas["particoes"][p]["buraco"] for p in PARTICOES) == 10
    assert sum(estatisticas["particoes"][p]["imagens"] for p in PARTICOES) == 20


def test_pasta_de_teste_por_pais_bate_com_o_total(rdd_falso, tmp_path):
    destino = tmp_path / "yolo"
    estatisticas = preparar(rdd_falso, destino)
    for pais in ("India", "Czech"):
        pasta = destino / "paises" / pais
        esperado = estatisticas["por_pais"][pais]["test"]["imagens"]
        assert len(list((pasta / "images" / "test").glob("*.jpg"))) == esperado
        assert len(list((pasta / "labels" / "test").glob("*.txt"))) == esperado
        assert "val: images/test" in (pasta / "dados.yaml").read_text(encoding="utf-8")


def test_repetir_gera_o_mesmo_resultado(rdd_falso, tmp_path):
    primeira = preparar(rdd_falso, tmp_path / "a")
    segunda = preparar(rdd_falso, tmp_path / "b")
    assert primeira["particoes"] == segunda["particoes"]
    assert sorted(p.name for p in (tmp_path / "a" / "images" / "test").iterdir()) == sorted(
        p.name for p in (tmp_path / "b" / "images" / "test").iterdir()
    )


def test_estatisticas_sao_gravadas(rdd_falso, tmp_path):
    destino = tmp_path / "yolo"
    estatisticas = preparar(rdd_falso, destino)
    gravadas = json.loads((destino / "estatisticas.json").read_text(encoding="utf-8"))
    assert gravadas["particoes"] == estatisticas["particoes"]


def test_xml_corrompido_e_imagem_ausente_sao_contados(rdd_falso, tmp_path):
    pasta = rdd_falso / "India" / "train"
    (pasta / "annotations" / "xmls" / "India_000999.xml").write_text("<annotation>", encoding="utf-8")
    (pasta / "images" / "India_000999.jpg").write_bytes(b"x")
    escrever_xml(pasta / "annotations" / "xmls" / "India_000998.xml", [("D40", 1, 1, 50, 50)])
    estatisticas = preparar(rdd_falso, tmp_path / "yolo")
    assert estatisticas["descartes"] == {"imagem_ausente": 1, "xml_invalido": 1}


def test_aceita_xml_direto_em_annotations(rdd_falso, tmp_path):
    pasta = rdd_falso / "India" / "train" / "annotations"
    for xml in list((pasta / "xmls").glob("*.xml")):
        xml.rename(pasta / xml.name)
    (pasta / "xmls").rmdir()
    estatisticas = preparar(rdd_falso, tmp_path / "yolo")
    assert estatisticas["por_pais"]["India"]["disponiveis"] == 600


def test_xml_fora_de_train_e_ignorado(rdd_falso, tmp_path):
    escrever_xml(rdd_falso / "India" / "test" / "annotations" / "India_950000.xml", [("D40", 1, 1, 50, 50)])
    estatisticas = preparar(rdd_falso, tmp_path / "yolo")
    assert estatisticas["por_pais"]["India"]["disponiveis"] == 600
    assert estatisticas["descartes"] == {}


def test_falha_com_mensagem_clara_sem_anotacoes(tmp_path):
    with pytest.raises(SystemExit, match="Nenhuma anotação"):
        preparar(Path(tmp_path), tmp_path / "yolo")
