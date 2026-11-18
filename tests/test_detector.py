from __future__ import annotations

import numpy as np
import pytest
import cv2

from cicloalerta.detector import Ajuste, DetectorONNX, decodificar, letterbox, nms

NOMES = {0: "buraco", 1: "rachadura"}
LIMIARES = np.array([0.3, 0.3], dtype=np.float32)


def saida_com(*predicoes):
    """Monta a saída (1, 6, N) a partir de tuplas (cx, cy, l, a, p_buraco, p_rachadura)."""
    return np.array(predicoes, dtype=np.float32).T[None]


def test_letterbox_preserva_proporcao():
    imagem = np.zeros((360, 640, 3), dtype=np.uint8)
    saida, ajuste = letterbox(imagem, (640, 640))
    assert saida.shape == (640, 640, 3)
    assert ajuste == Ajuste(1.0, 0, 140)
    assert saida[0, 0, 0] == 114
    assert saida[320, 320, 0] == 0


def test_nms_remove_sobreposicao_e_mantem_separados():
    caixas = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    assert nms(caixas, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]


def test_decodificar_desfaz_o_letterbox():
    # Caixa real em uma imagem 1280x720: (200, 100) a (600, 300). Na entrada 640x640,
    # a escala é 0,5 e a margem vertical é 140.
    saida = saida_com((200, 240, 200, 100, 0.9, 0.1))
    deteccoes = decodificar(saida, Ajuste(0.5, 0, 140), (720, 1280), NOMES, LIMIARES)
    assert len(deteccoes) == 1
    assert deteccoes[0].classe == "buraco"
    assert deteccoes[0].confianca == pytest.approx(0.9)
    assert deteccoes[0].caixa == pytest.approx((200, 100, 600, 300))


def test_limiar_e_por_classe():
    saida = saida_com((100, 100, 50, 50, 0.5, 0.0), (300, 300, 50, 50, 0.0, 0.5))
    limiares = np.array([0.6, 0.3], dtype=np.float32)
    deteccoes = decodificar(saida, Ajuste(1, 0, 0), (640, 640), NOMES, limiares)
    assert [d.classe for d in deteccoes] == ["rachadura"]


def test_nms_nao_mistura_classes():
    saida = saida_com((100, 100, 50, 50, 0.9, 0.0), (100, 100, 50, 50, 0.0, 0.8))
    assert len(decodificar(saida, Ajuste(1, 0, 0), (640, 640), NOMES, LIMIARES)) == 2


def test_caixas_ficam_dentro_da_imagem():
    saida = saida_com((5, 5, 100, 100, 0.9, 0.0))
    caixa = decodificar(saida, Ajuste(1, 0, 0), (640, 640), NOMES, LIMIARES)[0].caixa
    assert min(caixa) >= 0


def test_sem_nada_acima_do_limiar():
    assert decodificar(saida_com((1, 1, 2, 2, 0.1, 0.1)), Ajuste(1, 0, 0), (640, 640), NOMES, LIMIARES) == []


def test_recusa_formato_de_saida_desconhecido():
    with pytest.raises(ValueError, match="Exporte um modelo YOLO11"):
        decodificar(np.zeros((1, 300, 6), dtype=np.float32), Ajuste(1, 0, 0), (640, 640), NOMES, LIMIARES)


def test_le_nomes_e_tamanho_dos_metadados(sessao_falsa):
    metadados = {"names": "{0: 'buraco', 1: 'rachadura'}", "imgsz": "[416, 416]"}
    detector = DetectorONNX("x", sessao=sessao_falsa(metadados=metadados))
    assert detector.nomes == NOMES
    assert detector.imgsz == (416, 416)


def test_limiares_calibrados_substituem_o_padrao(sessao_falsa):
    detector = DetectorONNX("x", limiares={"buraco": 0.4}, limiar_padrao=0.2, sessao=sessao_falsa())
    assert detector.limiares.tolist() == pytest.approx([0.4, 0.2])


def test_preprocessamento_gera_tensor_rgb_normalizado(sessao_falsa):
    detector = DetectorONNX("x", sessao=sessao_falsa())
    quadro = np.zeros((640, 640, 3), dtype=np.uint8)
    quadro[..., 0] = 255  # azul no BGR do OpenCV
    tensor, _ = detector.preprocessar(quadro)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor[0, 2, 0, 0] == 1.0 and tensor[0, 0, 0, 0] == 0.0


def test_modelo_ausente_explica_o_que_fazer(tmp_path):
    with pytest.raises(FileNotFoundError, match="notebook de treino"):
        DetectorONNX(tmp_path / "nao_existe.onnx")


def test_limiares_json_ao_lado_do_modelo_sao_carregados(modelo_real, tmp_path):
    copia = tmp_path / "m.onnx"
    copia.write_bytes(modelo_real.read_bytes())
    (tmp_path / "limiares.json").write_text('{"person": 0.61}', encoding="utf-8")
    detector = DetectorONNX(copia)
    pessoa = next(i for i, nome in detector.nomes.items() if nome == "person")
    assert detector.limiares[pessoa] == pytest.approx(0.61)


def _rotulos_em_pixels(caminho, largura, altura):
    caixas = []
    for linha in caminho.read_text().splitlines():
        classe, cx, cy, l, a = map(float, linha.split())
        caixas.append(
            (int(classe), (cx - l / 2) * largura, (cy - a / 2) * altura, (cx + l / 2) * largura, (cy + a / 2) * altura)
        )
    return caixas


def test_modelo_real_acha_os_objetos_rotulados(modelo_real, coco128):
    """Confere decodificação e escala com um YOLO11 exportado de verdade, em imagens de vários formatos."""
    from cicloalerta.rastreador import iou

    detector = DetectorONNX(modelo_real, limiar_padrao=0.25)
    achados = esperados = 0
    for imagem in sorted((coco128 / "images" / "train2017").glob("*.jpg"))[:40]:
        rotulo = coco128 / "labels" / "train2017" / f"{imagem.stem}.txt"
        if not rotulo.exists():
            continue
        quadro = cv2.imread(str(imagem))
        deteccoes = detector.detectar(quadro)
        for classe, *caixa in _rotulos_em_pixels(rotulo, quadro.shape[1], quadro.shape[0]):
            esperados += 1
            achados += any(d.classe_id == classe and iou(d.caixa, caixa) >= 0.5 for d in deteccoes)
    assert esperados > 100
    assert achados / esperados > 0.45
