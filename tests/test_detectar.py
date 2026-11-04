from __future__ import annotations

import json

import cv2
import numpy as np

from via_atenta.detectar import LeitorAoVivo, desenhar, eh_fonte_de_imagens, filtrar_regiao, main, rotular_fonte
from via_atenta.detector import Deteccao


def deteccao(y1, y2):
    return Deteccao(0, "buraco", 0.9, (10, y1, 50, y2))


def test_filtra_o_topo_do_quadro():
    em_cima, embaixo = deteccao(0, 40), deteccao(500, 600)
    assert filtrar_regiao([em_cima, embaixo], 720, 0.25) == [embaixo]
    assert filtrar_regiao([em_cima, embaixo], 720, 0.0) == [em_cima, embaixo]


def test_identifica_tipo_de_fonte(tmp_path):
    assert eh_fonte_de_imagens(str(tmp_path))
    assert eh_fonte_de_imagens("rua.JPG")
    assert not eh_fonte_de_imagens("rua.mp4")
    assert not eh_fonte_de_imagens("0")
    assert not eh_fonte_de_imagens("http://192.168.0.10:8080/video")


def test_rotulo_da_fonte_nao_expoe_caminho_nem_senha():
    assert rotular_fonte("0") == "camera-0"
    assert rotular_fonte("/home/breno/videos/rua.mp4") == "rua.mp4"
    assert rotular_fonte("C:/Users/breno/rua.mp4") == "rua.mp4"
    assert rotular_fonte("rtsp://usuario:senha@192.168.0.10:554/stream") == "192.168.0.10"


def test_desenhar_nao_altera_o_original():
    quadro = np.zeros((100, 100, 3), dtype=np.uint8)
    saida = desenhar(quadro, [deteccao(10, 50)], fps=12.3)
    assert quadro.sum() == 0 and saida.sum() > 0


class CapturaFalsa:
    def __init__(self, total):
        self.total = total
        self.lidos = 0

    def read(self):
        if self.lidos >= self.total:
            return False, None
        self.lidos += 1
        return True, np.full((4, 4, 3), self.lidos, dtype=np.uint8)


def test_leitor_ao_vivo_termina_quando_a_camera_para():
    leitor = LeitorAoVivo(CapturaFalsa(200))
    quadros = list(leitor)
    leitor.parar()
    assert quadros
    assert quadros[-1][0, 0, 0] == 200 % 256


def test_leitor_ao_vivo_descarta_quadros_antigos():
    import time

    class CapturaLenta(CapturaFalsa):
        def read(self):
            time.sleep(0.002)
            return super().read()

    leitor = LeitorAoVivo(CapturaLenta(100))
    vistos = []
    for quadro in leitor:
        vistos.append(int(quadro[0, 0, 0]))
        time.sleep(0.02)  # consumidor mais lento que a câmera, como o modelo na CPU
    leitor.parar()
    assert len(vistos) < 50, "quadros antigos deveriam ter sido pulados"
    assert vistos == sorted(vistos)


def _video_de_teste(coco128, caminho):
    imagens = sorted((coco128 / "images" / "train2017").glob("*.jpg"))[:6]
    escritor = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    assert escritor.isOpened()
    for imagem in imagens:
        quadro = cv2.resize(cv2.imread(str(imagem)), (640, 480))
        for _ in range(6):
            escritor.write(quadro)
    escritor.release()


def test_ponta_a_ponta_em_video(modelo_real, coco128, tmp_path):
    video = tmp_path / "rua.avi"
    _video_de_teste(coco128, video)
    saida = tmp_path / "saida"
    codigo = main(["--fonte", str(video), "--modelo", str(modelo_real), "--saida", str(saida), "--sem-janela", "--conf", "0.4", "--salvar-video"])
    assert codigo == 0

    resumo = json.loads((saida / "desempenho.json").read_text(encoding="utf-8"))
    assert resumo["quadros"] == 36
    assert resumo["fps_medio"] > 0
    assert sum(resumo["ocorrencias"].values()) > 0
    linhas = (saida / "ocorrencias.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(linhas) == sum(resumo["ocorrencias"].values())
    assert (saida / json.loads(linhas[0])["recorte"]).exists()

    captura = cv2.VideoCapture(str(saida / "video_anotado.mp4"))
    assert int(captura.get(cv2.CAP_PROP_FRAME_COUNT)) == 36
    captura.release()


def test_ponta_a_ponta_em_pasta_de_imagens(modelo_real, coco128, tmp_path):
    saida = tmp_path / "saida"
    pasta = coco128 / "images" / "train2017"
    pasta_copia = tmp_path / "fotos"
    pasta_copia.mkdir()
    for imagem in sorted(pasta.glob("*.jpg"))[:3]:
        (pasta_copia / imagem.name).write_bytes(imagem.read_bytes())
    (pasta_copia / "quebrada.jpg").write_bytes(b"nao e imagem")

    assert main(["--fonte", str(pasta_copia), "--modelo", str(modelo_real), "--saida", str(saida), "--conf", "0.4"]) == 0
    resumo = json.loads((saida / "desempenho.json").read_text(encoding="utf-8"))
    assert resumo["quadros"] == 3
    assert len(list((saida / "anotadas").glob("*.jpg"))) == 3
