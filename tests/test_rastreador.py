from __future__ import annotations

import pytest

from via_atenta.detector import Deteccao
from via_atenta.rastreador import RastreadorIoU, iou


def deteccao(caixa=(100, 100, 200, 200), classe_id=0, confianca=0.8):
    return Deteccao(classe_id, ["buraco", "rachadura"][classe_id], confianca, caixa)


def test_iou_de_caixas_identicas_e_separadas():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_iou_de_caixa_vazia_nao_quebra():
    assert iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0


def test_confirma_uma_unica_vez_depois_de_tres_quadros():
    rastreador = RastreadorIoU(min_quadros=3)
    confirmacoes = []
    for quadro in range(10):
        _, novos = rastreador.atualizar([deteccao()], quadro)
        confirmacoes.append(len(novos))
    assert confirmacoes == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]


def test_ruido_de_um_quadro_nao_vira_ocorrencia():
    rastreador = RastreadorIoU(min_quadros=3, max_perdidos=1)
    rastreador.atualizar([deteccao()], 0)
    for quadro in range(1, 5):
        visiveis, novos = rastreador.atualizar([], quadro)
        assert not visiveis and not novos
    assert not rastreador.rastros


def test_acompanha_objeto_que_se_desloca():
    rastreador = RastreadorIoU(min_quadros=2)
    ids = set()
    for quadro in range(5):
        caixa = (100 + quadro * 10, 100, 200 + quadro * 10, 200)
        visiveis, _ = rastreador.atualizar([deteccao(caixa)], quadro)
        ids |= {r.id for r in visiveis}
    assert len(ids) == 1


def test_classes_diferentes_nao_se_misturam():
    rastreador = RastreadorIoU(min_quadros=1)
    rastreador.atualizar([deteccao(classe_id=0)], 0)
    _, novos = rastreador.atualizar([deteccao(classe_id=1)], 1)
    assert [r.classe for r in novos] == ["rachadura"]


def test_objeto_tolera_falhas_curtas():
    rastreador = RastreadorIoU(min_quadros=2, max_perdidos=3)
    rastreador.atualizar([deteccao()], 0)
    rastreador.atualizar([deteccao()], 1)
    rastreador.atualizar([], 2)
    rastreador.atualizar([], 3)
    visiveis, novos = rastreador.atualizar([deteccao()], 4)
    assert len(visiveis) == 1 and not novos


def test_objeto_some_depois_do_limite():
    rastreador = RastreadorIoU(min_quadros=1, max_perdidos=2)
    rastreador.atualizar([deteccao()], 0)
    for quadro in range(1, 4):
        rastreador.atualizar([], quadro)
    assert not rastreador.rastros


def test_dois_objetos_mantem_seus_ids():
    rastreador = RastreadorIoU(min_quadros=1)
    a, b = (0, 0, 100, 100), (300, 300, 400, 400)
    visiveis, _ = rastreador.atualizar([deteccao(a), deteccao(b)], 0)
    ids = {r.caixa: r.id for r in visiveis}
    visiveis, _ = rastreador.atualizar([deteccao(b), deteccao(a)], 1)
    assert {r.caixa: r.id for r in visiveis} == ids


def test_guarda_a_maior_confianca():
    rastreador = RastreadorIoU(min_quadros=1)
    rastreador.atualizar([deteccao(confianca=0.9)], 0)
    visiveis, _ = rastreador.atualizar([deteccao(confianca=0.5)], 1)
    assert visiveis[0].confianca == 0.5 and visiveis[0].confianca_maxima == 0.9
