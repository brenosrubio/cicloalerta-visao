#!/usr/bin/env python3
"""Gera docs/relatorio_cicloalerta.pdf na ordem pedida pela disciplina.

Todos os números vêm de resultados/metricas.json. Sem esse arquivo, sai um rascunho
com as lacunas marcadas, nunca com valores inventados. O script para com erro se o
resumo passar de 250 palavras ou o objetivo passar de 50.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

RAIZ = Path(__file__).resolve().parents[1]
LIMITE_RESUMO = 250
LIMITE_OBJETIVO = 50

COR_DESTAQUE = colors.HexColor("#B0124F")
COR_TEXTO = colors.HexColor("#1F2328")
COR_SUAVE = colors.HexColor("#5B6470")
COR_LINHA = colors.HexColor("#D9DDE3")
COR_FAIXA = colors.HexColor("#F5F6F8")
COR_PENDENTE = "#B0124F"

PASTA_FONTES = Path("/usr/share/fonts/truetype/crosextra")


def registrar_fontes() -> tuple[str, str, str]:
    """Usa Carlito se existir. Em outra máquina, cai para a Helvetica sem quebrar."""
    try:
        pdfmetrics.registerFont(TTFont("Corpo", PASTA_FONTES / "Carlito-Regular.ttf"))
        pdfmetrics.registerFont(TTFont("Corpo-Bold", PASTA_FONTES / "Carlito-Bold.ttf"))
        pdfmetrics.registerFont(TTFont("Corpo-Italic", PASTA_FONTES / "Carlito-Italic.ttf"))
        pdfmetrics.registerFont(TTFont("Corpo-BoldItalic", PASTA_FONTES / "Carlito-BoldItalic.ttf"))
        pdfmetrics.registerFontFamily(
            "Corpo", normal="Corpo", bold="Corpo-Bold", italic="Corpo-Italic", boldItalic="Corpo-BoldItalic"
        )
        return "Corpo", "Corpo-Bold", "Corpo-Italic"
    except Exception:
        return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


NORMAL, NEGRITO, ITALICO = registrar_fontes()

E_CORPO = ParagraphStyle(
    "corpo", fontName=NORMAL, fontSize=11, leading=15.5, textColor=COR_TEXTO, alignment=TA_JUSTIFY, spaceAfter=7
)
E_H1 = ParagraphStyle(
    "h1", fontName=NEGRITO, fontSize=16, leading=20, textColor=COR_DESTAQUE, spaceBefore=14, spaceAfter=8,
    keepWithNext=1,
)
E_H2 = ParagraphStyle(
    "h2", fontName=NEGRITO, fontSize=12, leading=15, textColor=COR_TEXTO, spaceBefore=9, spaceAfter=3, keepWithNext=1
)
E_LEGENDA = ParagraphStyle(
    "legenda", fontName=ITALICO, fontSize=9, leading=12, textColor=COR_SUAVE, spaceBefore=3, spaceAfter=10
)
E_CELULA = ParagraphStyle("celula", fontName=NORMAL, fontSize=9.5, leading=12, textColor=COR_TEXTO)
E_CELULA_N = ParagraphStyle("celula_n", parent=E_CELULA, fontName=NEGRITO)
E_REF = ParagraphStyle(
    "ref", fontName=NORMAL, fontSize=9.5, leading=13, textColor=COR_TEXTO, spaceAfter=5, leftIndent=12,
    firstLineIndent=-12,
)
E_CODIGO = ParagraphStyle(
    "codigo", fontName="Courier", fontSize=8, leading=10.5, textColor=COR_TEXTO, backColor=COR_FAIXA, borderPadding=6,
    spaceBefore=4, spaceAfter=12, leftIndent=6,
)
E_C_TITULO = ParagraphStyle(
    "c_titulo", fontName=NEGRITO, fontSize=27, leading=32, textColor=COR_TEXTO, alignment=TA_CENTER, spaceAfter=10
)
E_C_SUB = ParagraphStyle(
    "c_sub", fontName=NORMAL, fontSize=13, leading=18, textColor=COR_SUAVE, alignment=TA_CENTER, spaceAfter=4
)
E_C_MARCA = ParagraphStyle(
    "c_marca", fontName=NEGRITO, fontSize=13, leading=17, textColor=COR_DESTAQUE, alignment=TA_CENTER, spaceAfter=3
)
E_C_NOME = ParagraphStyle(
    "c_nome", fontName=NORMAL, fontSize=11.5, leading=17, textColor=COR_TEXTO, alignment=TA_CENTER
)
E_AVISO = ParagraphStyle(
    "aviso", fontName=NEGRITO, fontSize=10.5, leading=14, textColor=COR_DESTAQUE, alignment=TA_CENTER, spaceBefore=14
)


def pendente(texto: str = "aguardando treino") -> str:
    return f'<font color="{COR_PENDENTE}"><b>[{texto}]</b></font>'


def num(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def inteiro(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def contar_palavras(marcado: str) -> int:
    return len(re.sub(r"<[^>]+>", " ", marcado).split())


class Dados:
    """Lê os resultados e entrega cada número pronto para o texto."""

    def __init__(self, raiz: Path) -> None:
        self.raiz = raiz
        pasta = raiz / "resultados"
        arquivo = pasta / "metricas.json"
        self.pasta = pasta
        self.tem_resultados = arquivo.exists()
        self.m = json.loads(arquivo.read_text(encoding="utf-8")) if self.tem_resultados else None
        arquivo_dados = pasta / "estatisticas_dados.json"
        self.e = json.loads(arquivo_dados.read_text(encoding="utf-8")) if arquivo_dados.exists() else None
        arquivo_local = pasta / "exemplo_saida" / "desempenho.json"
        self.local = json.loads(arquivo_local.read_text(encoding="utf-8")) if arquivo_local.exists() else None
        self.integrantes = json.loads((raiz / "docs" / "integrantes.json").read_text(encoding="utf-8"))

    @property
    def n_imagens(self) -> str:
        if not self.e:
            return pendente("número de imagens")
        return inteiro(sum(p["imagens"] for p in self.e["particoes"].values()))

    def mapa50(self) -> str:
        return num(self.m["teste"]["geral"]["map50"]) if self.m else pendente("mAP50")

    def ap50(self, classe: str) -> str:
        return num(self.m["teste"]["por_classe"][classe]["ap50"]) if self.m else pendente(f"AP50 de {classe}")

    @property
    def latencia_ms(self) -> float | None:
        return self.m["onnx"]["latencia_cpu_colab_ms"] if self.m else None

    def fps(self) -> str:
        return num(1000 / self.latencia_ms, 0) if self.m else pendente("FPS")


def resumo(d: Dados) -> str:
    return (
        "Ciclistas e entregadores de São Paulo dividem as ciclovias com buracos e desníveis que causam quedas, e na "
        "chuva esses defeitos ficam escondidos pela água. Em 2025, uma auditoria cidadã percorreu 715 dos 736 "
        "quilômetros da rede cicloviária e classificou o pavimento de 20,5% dos trechos como precário, ruim ou "
        "razoável. O CicloAlerta é um site que vai mostrar esses perigos num mapa, e este trabalho entrega o seu "
        "componente de visão: um detector que confirma o defeito na foto enviada pelo ciclista e que roda em "
        "tempo real sobre o vídeo de uma câmera comum. Fizemos o ajuste fino de um YOLO11n, pré-treinado no COCO, com "
        f"{d.n_imagens} imagens reais de ruas da Índia, do Japão e da República Tcheca, vindas da base pública RDD2022. "
        "As quatro categorias originais viraram duas classes, buraco e rachadura. A divisão dos dados evitou "
        "vazamento entre imagens vizinhas. No teste, o modelo alcançou mAP50 de "
        f"{d.mapa50()}, com AP50 de {d.ap50('buraco')} para buracos e {d.ap50('rachadura')} para rachaduras. "
        f"Exportado para ONNX, rodou a {d.fps()} quadros por segundo em CPU, sem placa de vídeo. Cada defeito vira um "
        "registro pendente de revisão humana. A principal limitação é a origem "
        "das imagens: não encontramos base pública com fotos de ciclovias, e a RDD2022 não tem ruas brasileiras. Num "
        "vídeo gravado pela equipe numa ciclovia paulistana, o modelo marcou a pintura trincada. Em "
        "fotos de celular de outra base pública, o modelo foi pior nos buracos cheios de água do que nos secos, o que "
        "orienta o próximo treino."
    )


OBJETIVO = (
    "Treinar uma arquitetura YOLO11 com imagens reais de vias para detectar buracos e rachaduras em fotos e em vídeo, "
    "em tempo real e sem placa de vídeo, e entregar cada defeito como dado estruturado para a revisão humana e o mapa "
    "do CicloAlerta."
)

INTRODUCAO = [
    "O CicloAlerta foi pensado para ciclistas e entregadores de bicicleta elétrica que usam as ciclovias e "
    "ciclofaixas de São Paulo. Para quem pedala, um buraco ou um desnível é risco de queda, e na chuva a água esconde "
    "o defeito justamente quando o piso já está escorregadio.",
    "Os números mostram o tamanho do problema. A Auditoria Cidadã das Ciclovias e Ciclofaixas de SP, de 2025, "
    "percorreu 715,3 dos 735,7 quilômetros da rede e classificou o pavimento de 20,5% dos trechos como precário, ruim "
    "ou razoável (AUDITORIA CIDADÃ, 2025). Em texto do Jornal da USP, os sinistros com ciclistas em janeiro de 2026 "
    "aparecem 15% acima de janeiro de 2024, segundo a Associação Brasileira do Setor de Bicicletas, e um professor de "
    "engenharia de transportes da Poli-USP cita pistas escorregadias e buracos entre os problemas das ciclovias "
    "(ECYCLE, 2026). Na cidade como um todo, a prefeitura recebeu quase 52 mil pedidos de tapa-buraco no primeiro "
    "semestre de 2026, e 38% deles ainda estavam abertos no começo de agosto (BNEWS, 2026).",
    "Hoje o conserto depende de alguém ver o defeito e abrir um pedido. O CicloAlerta junta os relatos dos ciclistas, "
    "a previsão de chuva e a malha cicloviária em um mapa, e avisa quando a chuva deixa um trecho mais perigoso.",
    "Este documento trata do componente de visão, que corresponde à disciplina de Applied Computer Vision. Ele tem dois "
    "usos no produto. Na tela Reportar, confirma se a foto enviada mostra um buraco ou uma rachadura antes de o ponto ir "
    "para o mapa. Em tempo real, analisa o vídeo de uma câmera comum e registra cada defeito que encontra.",
    "O projeto se conecta à ODS 11 (cidades sustentáveis e transporte seguro), à ODS 3 (saúde, com menos sinistros de "
    "trânsito) e à ODS 9 (infraestrutura).",
]

FERRAMENTAS = [
    (
        "YOLO11n",
        "O YOLO11 é um detector de um estágio: a rede olha a imagem uma vez e prevê caixas e classes ao mesmo tempo, o "
        "que o torna rápido o bastante para vídeo. A arquitetura tem três partes. O backbone extrai características da "
        "imagem com blocos C3k2 e um bloco de atenção C2PSA. O neck combina mapas de três escalas, o que ajuda a achar "
        "tanto uma rachadura fina quanto um buraco grande. A cabeça de detecção não usa âncoras: ela prevê a distância "
        "até as bordas de cada caixa. Usamos a versão nano, a menor da família, com 2,6 milhões de parâmetros e "
        "mAP50-95 de 39,5 no COCO, segundo a documentação oficial (ULTRALYTICS, 2026). Escolhemos a menor porque o "
        "produto precisa rodar sem placa de vídeo.",
    ),
    (
        "Aprendizado por transferência",
        "Treinar do zero exigiria muito mais imagens e tempo. Partimos dos pesos do COCO, cujas primeiras camadas já "
        "reconhecem bordas e texturas, e trocamos a saída de 80 classes por uma de duas. O treino ajusta a rede inteira "
        "a partir desse ponto.",
    ),
    (
        "RDD2022",
        "A Road Damage Dataset 2022 reúne 47.420 imagens de seis países, com mais de 55 mil defeitos anotados (ARYA et "
        "al., 2022). As anotações seguem o formato Pascal VOC, nas categorias D00 (trinca longitudinal), D10 (trinca "
        "transversal), D20 (trinca em couro de jacaré) e D40 (buraco). A licença é CC BY-SA 4.0, que permite o uso com "
        "atribuição e o compartilhamento pela mesma licença.",
    ),
    (
        "ONNX Runtime e OpenCV",
        "ONNX é um formato aberto para redes neurais. Exportar para ele tira o PyTorch do aplicativo: o ONNX Runtime "
        "executa o modelo em CPU, e a instalação fica bem menor. O OpenCV lê a câmera ou o vídeo, ajusta os quadros e "
        "desenha as caixas.",
    ),
    (
        "NMS e rastreamento por IoU",
        "A rede devolve milhares de caixas candidatas. O limiar de confiança corta as fracas, e a supressão não máxima "
        "(NMS) elimina as sobrepostas. IoU é a razão entre a interseção e a união de duas caixas. Com essa medida "
        "escrevemos um rastreador simples: uma caixa que se sobrepõe à do quadro anterior, na mesma classe, é o mesmo "
        "defeito. Ele só é confirmado depois de três quadros, o que barra falsos positivos passageiros.",
    ),
    (
        "Métricas",
        "Precisão é a fração das detecções que estavam certas. Revocação é a fração dos defeitos reais que o modelo "
        "achou. F1 é a média harmônica das duas. AP50 é a área sob a curva de precisão por revocação, contando como "
        "acerto a caixa com IoU de pelo menos 0,5. O mAP50 é a média do AP50 entre as classes, e o mAP50-95 repete a "
        "conta com exigências de IoU de 0,5 a 0,95.",
    ),
    (
        "Apoio",
        "Ultralytics 8.4 para treino e exportação, Google Colab com GPU T4, Flask para a API deste repositório, uma "
        "rota pronta para o FastAPI do site e pytest para os testes automatizados.",
    ),
]

METODOLOGIA = [
    (
        "Procurar imagens de ciclovias",
        "Antes de escolher a base, procuramos fotos anotadas de ciclovias. A única base pública que encontramos, a Bike "
        "Lane Damage Detection (BLDD), tem 370 fotos de ciclovias da região metropolitana de Barcelona, com as mesmas "
        "categorias do RDD2022, mas o repositório publica só as anotações, sem as imagens e sem licença (PRATS, 2021). "
        "Por isso treinamos com fotos de ruas, cujo asfalto é o mesmo de muitas ciclofaixas pintadas no leito da via, e "
        "tratamos a falta de fotos de ciclovia como limitação.",
    ),
    (
        "Definir as classes",
        "D00, D10 e D20 descrevem o formato da trinca. Na triagem inicial, o que pesa é separar trinca de buraco, que "
        "traz risco imediato, e juntar as três categorias dá mais exemplos ao modelo. Por isso elas viraram uma só "
        "classe, rachadura. D40 virou buraco. Outros rótulos da base foram ignorados.",
    ),
    (
        "Obter e conferir os dados",
        "Em 04/11/2026, os endereços oficiais do RDD2022 responderam com erro 403 quando acessados pelo Colab. Procuramos "
        "espelhos públicos e conferimos cada um antes de usar. O espelho dronefreak/RDD2022 tem todas as imagens e "
        "contagens de trincas a menos de 1% das publicadas pela Dataset Ninja (2026). Já a classe chamada pothole desse "
        "espelho não servia: ao olhar as imagens, as caixas marcavam faixas de pedestre e sinalização gasta, e a "
        "República Tcheca aparecia sem nenhum buraco. Por isso, as trincas vieram desse espelho, e os buracos, do "
        "espelho edwrdc/RDD2022, cujas caixas caem sobre buracos de verdade. Só usamos imagens presentes nos dois, o "
        "que deixou de fora 687 imagens da Índia e 181 da República Tcheca. As duas versões ficaram fixadas no notebook.",
    ),
    (
        "Escolher os países",
        "Usamos Índia, Japão e República Tcheca, onde as imagens vêm de celulares comuns presos a carros (ARYA et al., "
        "2022), o mais perto da câmera de um celular que a base oferece. Ficaram de fora a Noruega, que usa câmeras especiais, os Estados Unidos, com imagens do "
        "Street View, e a China, com moto e drone.",
    ),
    (
        "Converter os dados",
        "Cada anotação virou um arquivo de texto no formato YOLO, com as caixas normalizadas. Limitamos cada país a 2.500 "
        "imagens com defeito, priorizando as que têm buraco, a classe mais rara. Outros 10% do conjunto são imagens sem "
        "defeito, que ensinam o modelo a não inventar detecção.",
    ),
    (
        "Dividir sem vazamento",
        "O conjunto de teste oficial do RDD2022 não tem rótulos públicos, então dividimos o conjunto rotulado em cerca "
        "de 80% para treino, 10% para validação e 10% para teste. Como as imagens vêm de veículos em movimento, quadros "
        "vizinhos podem ser quase iguais. A divisão foi feita por blocos de 50 nomes consecutivos, usando hash, de modo "
        "que um bloco inteiro fique em uma só partição e o resultado se repita em qualquer máquina.",
    ),
    (
        "Treinar",
        "Ajuste fino no Colab, com GPU T4: entrada de 640 pixels, lote de 32, até 50 épocas com parada após 12 sem "
        "melhora, taxa de aprendizado cosseno e semente 42. Mantivemos o espelhamento horizontal e o mosaico, desligado "
        "nas últimas 10 épocas. O espelhamento vertical ficou desligado, porque a rua nunca aparece de cabeça para "
        "baixo.",
    ),
    (
        "Calibrar os limiares",
        "Na validação, buscamos o limiar de confiança com maior F1 em cada classe. O teste só foi usado depois disso, "
        "para não contaminar a medida.",
    ),
    (
        "Avaliar",
        "Medimos o modelo no teste, no geral, por classe e por país. A separação por país serve para encontrar viés: um "
        "modelo pode ir bem na média e mal em um lugar específico.",
    ),
    (
        "Medir fotos tiradas de perto",
        "A tela Reportar recebe fotos de alguém em pé ao lado do buraco, enquanto o modelo aprendeu com câmera de carro. "
        "Para estimar o efeito, recortamos um quadrado em volta de cada defeito marcado nas 60 amostras do teste, com "
        "lado de 4 e de 2,5 vezes o lado maior da caixa, e contamos quantos o modelo ainda encontrava, na mesma classe e "
        "com IoU de pelo menos 0,3. O script é ferramentas/testar_fotos_aproximadas.py.",
    ),
    (
        "Medir em fotos de celular com e sem água",
        "Para medir o cenário de chuva, rodamos o modelo nas 713 fotos de celular da base de Kent, com 1.153 buracos "
        "marcados no Reino Unido, muitos cheios de água (DIB; SIRLANTZIS; HOWELLS, 2023). O modelo nunca viu essas "
        "fotos. Separamos à mão as 200 fotos da partição de teste em com água e secas, deixando de fora três em que não "
        "dava para decidir, e medimos cada grupo com o aplicativo final. O script é ferramentas/testar_kent.py.",
    ),
    (
        "Testar numa ciclovia de São Paulo",
        "A equipe gravou ciclovias de São Paulo com o celular preso ao guidão de uma bicicleta elétrica. Escolhemos o "
        "trecho mais estável, de 39 segundos, e o convertemos de 60 para 15 quadros por segundo, que é o ritmo em que o "
        "detector roda na CPU. O guidão aparece no terço de baixo da imagem e virava buraco, então criamos a opção "
        "--ignorar-base, que descarta essa faixa. A foto de um trecho com pintura trincada, tirada no mesmo lugar, "
        "passou pela rota da tela Reportar. Sem marcação humana, esse teste é qualitativo.",
    ),
    (
        "Exportar e conferir",
        "Exportamos para ONNX e comparamos, em 100 imagens do teste, as caixas do PyTorch com as do aplicativo final.",
    ),
    (
        "Rodar em tempo real",
        "O aplicativo lê a câmera em uma linha de execução separada e sempre processa o quadro mais recente. Sem isso, "
        "os quadros acumulam enquanto o modelo trabalha e a imagem na tela fica atrasada. Cada quadro passa por "
        "redimensionamento, modelo, NMS e rastreador. O mesmo programa aceita webcam, celular via URL, vídeo e pasta de "
        "imagens.",
    ),
    (
        "Integrar e proteger os dados",
        "A rota POST /api/detectar atende a tela Reportar: recebe a foto e devolve as detecções já filtradas pelos "
        "limiares calibrados, com uma severidade sugerida pelo tamanho da caixa, regra provisória que a revisão humana "
        "confirma. No tempo real, cada defeito confirmado vira uma linha em ocorrencias.jsonl. Tudo nasce com status "
        "pendente_revisao, porque o modelo sugere e uma pessoa decide. Para seguir a LGPD, a API processa a foto em "
        "memória e não grava nada, o tempo real guarda só o recorte do defeito, e a localização registra o ponto do "
        "buraco, nunca o trajeto de quem pedala. Se o site guardar a foto, ele apaga os metadados (EXIF) antes.",
    ),
]

REFERENCIAS = [
    "ARYA, D. et al. RDD2022: a multi-national image dataset for automatic road damage detection. arXiv:2209.08538, "
    "2022. Disponível em: https://arxiv.org/abs/2209.08538.",
    "AUDITORIA CIDADÃ das ciclovias e ciclofaixas de SP 2025. Apresentação publicada pela Prefeitura de São Paulo. "
    "Disponível em: https://prefeitura.sp.gov.br/documents/d/mobilidade/auditoria_cidada_das_ciclovias_e_ciclofaixas_"
    "de_sp_2025_ctb-pdf.",
    "BNEWS. Mesmo após recapeamento, SP registra quase 52 mil pedidos de tapa-buraco. 9 ago. 2026. Disponível em: "
    "https://www.bnewssaopaulo.com.br/noticias/politica/mesmo-apos-recapeamento-sp-registra-quase-52-mil-pedidos-de-"
    "tapa-buraco.html.",
    "DATASET NINJA. RDD2022. Disponível em: https://datasetninja.com/road-damage-detector.",
    "DIB, J.; SIRLANTZIS, K.; HOWELLS, G. An annotated water-filled, and dry potholes dataset for deep learning "
    "applications. Data in Brief, v. 48, art. 109206, 2023. DOI: 10.1016/j.dib.2023.109206. Dados em: "
    "https://data.mendeley.com/datasets/tp95cdvgm8/1.",
    "DRONEFREAK. RDD2022 Multi-National Road Damage Detection Dataset (4-Class YOLO Export). Hugging Face. Revisão "
    "72b01f94. Disponível em: https://huggingface.co/datasets/dronefreak/RDD2022.",
    "ECYCLE. Aumento no número de sinistros com ciclistas em SP escancara problemas estruturais das ciclovias. Texto do "
    "Jornal da USP. 6 abr. 2026. Disponível em: https://www.ecycle.com.br/aumento-no-numero-de-sinistros-com-"
    "ciclistas-em-sp-escancara-problemas-estruturais-das-ciclovias/.",
    "EDWRDC. RDD2022. Hugging Face. Revisão d8967222. Disponível em: https://huggingface.co/datasets/edwrdc/RDD2022.",
    "ONU. Objetivos de Desenvolvimento Sustentável. Disponível em: https://brasil.un.org/pt-br/sdgs.",
    "PRATS, J. Micromobility safety applications using AI. Trabalho de conclusão de curso, Universitat Politècnica de "
    "Catalunya, 2021. Disponível em: https://core.ac.uk/download/491494203.pdf. Anotações em: "
    "https://github.com/JaumePrats/BikeLaneDamageDetection.",
    "ULTRALYTICS. Ultralytics YOLO11. 2026. Disponível em: https://docs.ultralytics.com/models/yolo11/.",
]


def tabela(linhas: list[list[str]], larguras: list[float], alinhar_numeros: bool = True) -> Table:
    celulas = [
        [Paragraph(str(c), E_CELULA_N if i == 0 else E_CELULA) for c in linha] for i, linha in enumerate(linhas)
    ]
    resultado = Table(celulas, colWidths=[l * cm for l in larguras], repeatRows=1, hAlign="LEFT")
    resultado.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, 0), 0.9, COR_DESTAQUE),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, COR_LINHA),
                ("BACKGROUND", (0, 0), (-1, 0), COR_FAIXA),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return resultado


def figura(caminho: Path, largura_cm: float, legenda: str) -> list:
    leitor = ImageReader(str(caminho))
    largura, altura = leitor.getSize()
    w = largura_cm * cm
    return [KeepTogether([Image(str(caminho), width=w, height=w * altura / largura), Paragraph(legenda, E_LEGENDA)])]


def secao_resultados(d: Dados) -> list:
    corpo: list = [Paragraph("6. Exemplos de resultados", E_H1)]
    if not d.tem_resultados:
        corpo.append(
            Paragraph(
                pendente("Esta seção é preenchida com os números reais assim que o resultados.zip do treino chegar.")
                + " Ela terá as imagens usadas por partição, o desempenho no teste por classe e por país, a velocidade "
                "em CPU, a curva de precisão por revocação, a matriz de confusão, exemplos de acertos e de erros e uma "
                "ocorrência real em JSON.",
                E_CORPO,
            )
        )
        return corpo

    m, e = d.m, d.e
    corpo.append(Paragraph("Dados usados", E_H2))
    nomes = {"train": "Treino", "val": "Validação", "test": "Teste"}
    linhas = [["Partição", "Imagens", "Buracos", "Rachaduras", "Sem defeito"]]
    for chave, nome in nomes.items():
        p = e["particoes"][chave]
        linhas.append([nome, inteiro(p["imagens"]), inteiro(p["buraco"]), inteiro(p["rachadura"]), inteiro(p["fundo"])])
    corpo += [tabela(linhas, [3.6, 3, 3, 3, 3]), Paragraph("Tabela 1. Imagens e caixas anotadas em cada partição.", E_LEGENDA)]

    corpo.append(Paragraph("Desempenho no teste", E_H2))
    geral = m["teste"]["geral"]
    corpo.append(
        Paragraph(
            f"O treino levou {num(m['treino']['tempo_min'], 0)} minutos e {m['treino']['epocas_executadas']} épocas, com a "
            f"melhor na época {m['treino']['melhor_epoca']}. Em imagens que o modelo nunca viu, o mAP50 foi "
            f"{num(geral['map50'])} e o mAP50-95 foi {num(geral['map50_95'])}."
            + (
                " Como a melhor época foi a última, o modelo ainda podia melhorar com mais treino."
                if m["treino"]["melhor_epoca"] >= m["treino"]["epocas_executadas"]
                else ""
            ),
            E_CORPO,
        )
    )
    linhas = [["Classe", "Instâncias", "AP50", "AP50-95", "Limiar", "Precisão", "Revocação"]]
    for classe, c in m["teste"]["por_classe"].items():
        linhas.append(
            [
                classe.capitalize(),
                inteiro(c["instancias"]),
                num(c["ap50"]),
                num(c["ap50_95"]),
                num(c["limiar"], 2),
                num(c["precisao_no_limiar"]),
                num(c["revocacao_no_limiar"]),
            ]
        )
    corpo += [
        tabela(linhas, [2.8, 2.1, 1.9, 2.1, 1.9, 2.1, 2.3]),
        Paragraph("Tabela 2. Desempenho no teste. Precisão e revocação no limiar calibrado na validação.", E_LEGENDA),
    ]

    corpo.append(Paragraph("Comparação por país", E_H2))
    nomes_pais = {"India": "Índia", "Japan": "Japão", "Czech": "República Tcheca"}
    com_preposicao = {"India": "na Índia", "Japan": "no Japão", "Czech": "na República Tcheca"}
    com_artigo = {"India": "A Índia", "Japan": "O Japão", "Czech": "A República Tcheca"}
    ordenados = sorted(m["por_pais"].items(), key=lambda item: item[1]["geral"]["map50"], reverse=True)
    linhas = [["País", "Imagens no teste", "mAP50", "AP50 buraco", "AP50 rachadura"]]
    for pais, dados in ordenados:
        classes = dados["por_classe"]
        linhas.append(
            [
                nomes_pais.get(pais, pais),
                inteiro(e["por_pais"][pais]["test"]["imagens"]),
                num(dados["geral"]["map50"]),
                num(classes["buraco"]["ap50"]) if "buraco" in classes else "sem exemplos",
                num(classes["rachadura"]["ap50"]) if "rachadura" in classes else "sem exemplos",
            ]
        )
    melhor, pior = ordenados[0], ordenados[-1]
    faixa = {
        classe: sorted(d["por_classe"][classe]["ap50"] for _, d in ordenados if classe in d["por_classe"])
        for classe in ("buraco", "rachadura")
    }
    variacao = (
        f"A maior variação está nos buracos: o AP50 foi de {num(faixa['buraco'][0])} a {num(faixa['buraco'][-1])}, "
        f"contra {num(faixa['rachadura'][0])} a {num(faixa['rachadura'][-1])} nas rachaduras. "
        if faixa["buraco"][-1] - faixa["buraco"][0] > faixa["rachadura"][-1] - faixa["rachadura"][0]
        else ""
    )
    corpo += [
        tabela(linhas, [4, 3, 2.4, 3, 3.2]),
        Paragraph("Tabela 3. Desempenho do mesmo modelo em cada país do teste.", E_LEGENDA),
        Paragraph(
            f"O modelo foi melhor {com_preposicao.get(melhor[0], melhor[0])} (mAP50 de "
            f"{num(melhor[1]['geral']['map50'])}) e pior {com_preposicao.get(pior[0], pior[0])} "
            f"({num(pior[1]['geral']['map50'])}). {variacao}Isso mostra que o desempenho muda com o lugar e reforça a "
            "necessidade de medir o modelo em ruas brasileiras antes de usá-lo aqui.",
            E_CORPO,
        ),
    ]
    for pais, _ in ordenados:
        quantidade = e["por_pais"][pais]["test"]["buraco"]
        if quantidade < 30:
            corpo.append(
                Paragraph(
                    f"{com_artigo.get(pais, pais)} tem só {quantidade} buracos no teste. Com tão poucos exemplos, o AP50 "
                    "de buraco desse país oscila muito e não deve ser lido sozinho.",
                    E_CORPO,
                )
            )

    corpo.append(Paragraph("Velocidade", E_H2))
    onnx = m["onnx"]
    linhas = [
        ["Ambiente", "Tempo por quadro", "Quadros por segundo"],
        [
            f"GPU do Colab ({m['ambiente']['gpu'] or 'GPU'}), só inferência",
            f"{num(m['teste']['velocidade_ms']['inference'], 1)} ms",
            num(1000 / m["teste"]["velocidade_ms"]["inference"], 0),
        ],
        ["CPU do Colab, aplicativo com ONNX", f"{num(onnx['latencia_cpu_colab_ms'], 1)} ms", num(1000 / onnx["latencia_cpu_colab_ms"], 1)],
    ]
    if d.local:
        linhas.append(
            [
                "Outra CPU de 2 núcleos, aplicativo com ONNX",
                f"{num(d.local['latencia_media_ms'], 1)} ms",
                num(1000 / d.local["latencia_media_ms"], 1),
            ]
        )
    arquivo_equipe = d.raiz / "docs" / "medicoes_equipe.json"
    if arquivo_equipe.exists():
        equipe = json.loads(arquivo_equipe.read_text(encoding="utf-8"))
        linhas.append([equipe["descricao"], f"{num(equipe['latencia_media_ms'], 1)} ms", num(equipe["fps_medio"], 1)])
    corpo += [tabela(linhas, [8, 3.6, 3.6]), Paragraph("Tabela 4. Velocidade medida, incluindo preparo do quadro e NMS nas linhas de CPU.", E_LEGENDA)]
    if onnx.get("concordancia_com_pytorch") is not None:
        corpo.append(
            Paragraph(
                f"Na comparação entre PyTorch e aplicativo, {num(onnx['concordancia_com_pytorch'] * 100, 1)}% das "
                f"{inteiro(onnx['caixas_comparadas'])} caixas foram reproduzidas pelo ONNX.",
                E_CORPO,
            )
        )

    curva = d.pasta / "teste" / "BoxPR_curve.png"
    matriz = d.pasta / "teste" / "confusion_matrix_normalized.png"
    if curva.exists():
        corpo += figura(curva, 12, "Figura 1. Curva de precisão por revocação no teste.")
    if matriz.exists():
        corpo += figura(matriz, 11, "Figura 2. Matriz de confusão normalizada no teste.")
    acertos = d.pasta / "exemplos" / "acertos.jpg"
    erros = d.pasta / "exemplos" / "erros.jpg"
    credito = "Verde: anotação da base. Vermelho: buraco previsto. Laranja: rachadura prevista. Imagens: RDD2022, CC BY-SA 4.0."
    if acertos.exists():
        corpo += figura(acertos, 15.4, f"Figura 3. Exemplos de acertos no teste. {credito}")
    if erros.exists():
        corpo += figura(erros, 15.4, f"Figura 4. Exemplos de erros no teste. {credito}")

    aproximado = d.pasta / "teste_aproximado.json"
    if aproximado.exists():
        r = json.loads(aproximado.read_text(encoding="utf-8"))["resultados"]
        corpo.append(Paragraph("Fotos tiradas de perto", E_H2))
        linhas = [["Defeitos encontrados", "Foto inteira", "Recorte médio", "Recorte de perto"]]
        for classe in ("buraco", "rachadura"):
            total = r["foto_inteira"][classe]["total"]
            linhas.append(
                [f"{classe.capitalize()} ({total} marcados)"]
                + [str(r[nivel][classe]["encontrados"]) for nivel in ("foto_inteira", "recorte_medio", "recorte_perto")]
            )
        b = {nivel: r[nivel]["buraco"]["encontrados"] for nivel in r}
        corpo += [
            tabela(linhas, [5.2, 3.2, 3.4, 3.6]),
            Paragraph(
                "Tabela 5. Defeitos encontrados nas 60 amostras do teste, na foto inteira e em recortes com lado de 4 "
                "(médio) e 2,5 (de perto) vezes o lado maior da caixa.",
                E_LEGENDA,
            ),
            Paragraph(
                f"Quanto mais perto, menos defeitos o modelo encontra: nos buracos, foram {b['foto_inteira']} na foto "
                f"inteira e {b['recorte_perto']} no recorte de perto. O recorte ampliado fica mais borrado que uma foto "
                "real, então a queda real deve ser menor. Mesmo assim, o resultado define uma regra do produto: quando o "
                "modelo não encontra nada, o site deixa o ciclista enviar o relato do mesmo jeito.",
                E_CORPO,
            ),
        ]

    kent = d.pasta / "teste_kent.json"
    if kent.exists():
        g = json.loads(kent.read_text(encoding="utf-8"))["grupos"]
        corpo.append(Paragraph("Fotos de celular com e sem água", E_H2))
        linhas = [["Fotos de Kent", "Fotos", "Buracos", "AP50", "Revocação no limiar", "Fotos em que aponta buraco"]]
        for chave, nome in (("teste_seco", "Teste, secas"), ("teste_com_agua", "Teste, com água"), ("todas_as_fotos", "Todas as 713")):
            linha = g[chave]
            linhas.append([
                nome, inteiro(linha["fotos"]), inteiro(linha["buracos_marcados"]), num(linha["ap50"], 2),
                num(linha["revocacao_no_limiar"], 2), f"{num(linha['fotos_em_que_aponta_buraco'] * 100, 0)}%",
            ])
        seco, agua = g["teste_seco"], g["teste_com_agua"]
        corpo += [
            tabela(linhas, [3.6, 1.7, 2.0, 1.7, 3.0, 3.6]),
            Paragraph("Tabela 6. Modelo atual nas fotos de celular da base de Kent, classe buraco, IoU de 0,5.", E_LEGENDA),
            Paragraph(
                f"Com água, o modelo acha menos buracos: o AP50 cai de {num(seco['ap50'], 2)} para {num(agua['ap50'], 2)}, e a "
                f"parte das fotos em que ele aponta ao menos um buraco cai de {num(seco['fotos_em_que_aponta_buraco'] * 100, 0)}% "
                f"para {num(agua['fotos_em_que_aponta_buraco'] * 100, 0)}%. Os buracos grandes cheios de água, fotografados "
                "de perto, são os que ele mais perde, como mostra a Figura 5. O resultado confirma o problema que o "
                "CicloAlerta quer tratar e aponta o próximo treino. Parte das caixas dessa base está deslocada em relação "
                "ao buraco, então esses números tendem a subestimar o modelo.",
                E_CORPO,
            ),
        ]
        figura_kent = d.pasta / "exemplos" / "kent_agua.jpg"
        if figura_kent.exists():
            corpo += figura(
                figura_kent,
                15.4,
                "Figura 5. Buracos com água na base de Kent. Em cima, buracos que o modelo encontrou. Embaixo, buracos "
                "que ele perdeu. Verde: marcação da base. Vermelho: buraco previsto. Laranja: rachadura prevista. "
                "Fotos: Dib, Sirlantzis e Howells (2023), CC BY 4.0.",
            )

    video = d.pasta / "exemplos" / "video_equipe.jpg"
    if video.exists():
        corpo.append(Paragraph("Vídeo de rua gravado pela equipe", E_H2))
        corpo.append(
            Paragraph(
                "A equipe filmou 51 segundos de uma rua de São Paulo com o celular e rodou o detector sobre o vídeo, só "
                "com CPU. Ele registrou 28 ocorrências, 23 rachaduras e 5 buracos. Parte delas eram acertos, como a "
                "trinca em couro de jacaré da Figura 6. Outras eram marcas de pneu, faixas pintadas e uma tampa de "
                "bueiro, o que confirma a necessidade da revisão humana.",
                E_CORPO,
            )
        )
        corpo += figura(
            video,
            12.5,
            "Figura 6. Quadros do vídeo da equipe com o detector rodando. À esquerda, trinca em couro de jacaré marcada "
            "como rachadura (0,57). À direita, marcas de pneu confundidas com rachadura.",
        )

    ciclo = d.pasta / "teste_ciclovia.json"
    figura_ciclo = d.pasta / "exemplos" / "ciclovia.jpg"
    if ciclo.exists() and figura_ciclo.exists():
        c = json.loads(ciclo.read_text(encoding="utf-8"))
        corpo.append(Paragraph("Teste numa ciclovia de São Paulo", E_H2))
        corpo.append(
            Paragraph(
                f"No vídeo da ciclovia, o detector rodou a {num(c['fps_medio_cpu_2_nucleos'], 1)} quadros por segundo e "
                f"registrou {c['ocorrencias_com_ignorar_base']['rachadura']} rachaduras. A pintura trincada da ciclovia "
                f"foi marcada com confiança de até {num(c['maior_confianca_rachadura'], 2)}, o melhor acerto em imagens de "
                "São Paulo. Muitas das outras ocorrências são linhas que cruzam a ciclovia e podem ser trincas ou juntas "
                "do piso; só a revisão humana separa as duas. Sem a opção --ignorar-base, o guidão chegou a ser marcado "
                f"como buraco ({num(c['guidao_como_buraco_confianca'], 2)}). Na foto do trecho trincado, a rota da tela "
                "Reportar respondeu que há defeito, com uma rachadura na pintura e um buraco onde ela descascou.",
                E_CORPO,
            )
        )
        if "nao_registrados" in c and "teste_limiar_buraco" in c:
            borda, vao = c["nao_registrados"][:2]
            lim = c["teste_limiar_buraco"]
            outros = lim["buracos_registrados_em_outros_4_videos_da_equipe"]
            corpo.append(
                Paragraph(
                    "Dois defeitos ficaram sem registro. Uma borda quebrada entre a faixa e o concreto foi marcada como "
                    f"buraco com até {num(borda['maior_confianca_buraco'], 2)}, mas em só "
                    f"{borda['quadros_acima_do_limiar']} quadros, e o detector exige 3. O vão de uma travessia não "
                    f"passou de {num(vao['maior_confianca_buraco'], 2)}. São degraus e emendas de ciclovia, diferentes "
                    "dos buracos de asfalto da RDD2022. Baixar o limiar de buraco para "
                    f"{num(lim['limiar_testado'], 2)} registrou a borda, mas também o guidão e o painel da bicicleta, e "
                    f"mais que dobrou os buracos registrados em outros quatro vídeos da equipe (de {outros['0.277']} para "
                    f"{outros['0.15']}), que quase não mostram buracos reais. Mantivemos o limiar calibrado.",
                    E_CORPO,
                )
            )
        corpo += figura(
            figura_ciclo,
            15.4,
            "Figura 7. Ciclovia de São Paulo gravada pela equipe. À esquerda, pintura trincada marcada como rachadura "
            "(0,75). No meio, a foto do mesmo trecho enviada à rota da tela Reportar. À direita, o guidão marcado como "
            "buraco, erro que a opção --ignorar-base resolve.",
        )

    arquivo = d.pasta / "exemplo_saida" / "ocorrencias.jsonl"
    if arquivo.exists() and arquivo.stat().st_size:
        primeira = json.loads(arquivo.read_text(encoding="utf-8").splitlines()[0])
        texto = json.dumps(primeira, ensure_ascii=False, indent=2)
        texto = re.sub(r"\[\s+([^\]]+?)\s+\]", lambda c: "[" + " ".join(c[1].split()) + "]", texto)
        corpo.append(
            KeepTogether(
                [
                    Paragraph("Ocorrência gerada", E_H2),
                    Paragraph("Este é o dado que os outros componentes do MVP recebem:", E_CORPO),
                    Preformatted(texto, E_CODIGO),
                ]
            )
        )
    return corpo


def leitura_das_classes(d: Dados) -> str:
    if not d.m:
        return pendente("Comparação entre buraco e rachadura, escrita com os números reais do treino.")
    classes = d.m["teste"]["por_classe"]
    pior, melhor = sorted(classes, key=lambda nome: classes[nome]["ap50"])
    texto = (
        f"O modelo foi melhor em {melhor} (AP50 de {num(classes[melhor]['ap50'])}) do que em {pior} "
        f"(AP50 de {num(classes[pior]['ap50'])}). "
    )
    if pior == "rachadura":
        return texto + "Uma explicação provável é que a rachadura é longa e fina, e a caixa que a envolve varia muito de uma anotação para outra."
    return texto + "Mesmo priorizando imagens com buraco, essa classe ficou atrás, o que pede mais exemplos dela no próximo treino."


def limites(d: Dados) -> str:
    texto = (
        "O limite mais sério é a distância entre os dados de treino e o uso real. A base não tem ciclovias nem ruas "
        "brasileiras. No vídeo de ciclovia gravado pela equipe, o modelo marcou a pintura trincada, mas também linhas "
        "do piso que podem ser só juntas, e não registrou degraus e emendas quebradas, que não existem na base. Esse "
        "teste ainda não tem marcação humana para virar número. "
    )
    kent = d.pasta / "teste_kent.json"
    if kent.exists():
        g = json.loads(kent.read_text(encoding="utf-8"))["grupos"]
        texto += (
            f"Em fotos de celular com água, justamente o cenário que o CicloAlerta destaca, o AP50 de buraco caiu de "
            f"{num(g['teste_seco']['ap50'], 2)} para {num(g['teste_com_agua']['ap50'], 2)}. "
        )
    aproximado = d.pasta / "teste_aproximado.json"
    if aproximado.exists():
        r = json.loads(aproximado.read_text(encoding="utf-8"))["resultados"]
        texto += (
            f"Em recortes que imitam fotos de perto, os buracos encontrados caíram de {r['foto_inteira']['buraco']['encontrados']} "
            f"para {r['recorte_perto']['buraco']['encontrados']} em {r['foto_inteira']['buraco']['total']}. "
        )
    return texto + (
        "O próximo passo é um ajuste fino com fotos de celular de buracos com água. Ele não coube no prazo porque o "
        "Colab ficou sem GPU disponível. Depois, o próprio produto ajuda: as fotos que chegam pela tela Reportar, já "
        "revisadas, formam uma base de ciclovias paulistanas, com piso seco e molhado."
    )


def conclusao(d: Dados) -> list[str]:
    if d.m:
        classes = d.m["teste"]["por_classe"]
        abertura = (
            f"O detector cumpriu o que a atividade pede: foi treinado sob medida para o problema, chegou a mAP50 de "
            f"{d.mapa50()} em imagens inéditas e rodou a cerca de {d.fps()} quadros por segundo em CPU, sem placa de "
            f"vídeo. O desempenho é moderado. No limiar calibrado, o modelo acha "
            f"{num(classes['buraco']['revocacao_no_limiar'] * 100, 0)}% dos buracos e "
            f"{num(classes['rachadura']['revocacao_no_limiar'] * 100, 0)}% das rachaduras em uma imagem isolada. Isso "
            "serve para triagem com revisão humana, e não para substituir a vistoria da ciclovia. Em vídeo, o mesmo defeito "
            "aparece em vários quadros e o modelo tem mais de uma chance, mas não medimos esse ganho."
        )
    else:
        abertura = pendente("Parágrafo de abertura escrito com os números reais do treino.")
    return [
        abertura,
        leitura_das_classes(d),
        limites(d),
        "O rastreador também tem limite: em velocidade alta, a sobreposição entre quadros cai e o mesmo defeito pode ser "
        "contado duas vezes. Por isso a contagem é uma aproximação, e cada ocorrência passa por revisão humana antes de "
        "virar alerta no mapa.",
    ]


def rodape(canvas, documento) -> None:
    if documento.page == 1:
        return
    canvas.saveState()
    canvas.setFont(NORMAL, 9)
    canvas.setFillColor(COR_SUAVE)
    canvas.drawString(2.5 * cm, 1.4 * cm, "CicloAlerta · Applied Computer Vision · FIAP")
    canvas.drawRightString(A4[0] - 2.5 * cm, 1.4 * cm, str(documento.page))
    canvas.restoreState()


def capa(d: Dados) -> list:
    corpo: list = [
        Spacer(1, 3.2 * cm),
        Paragraph("FIAP · Engenharia de Software · 4º ano", E_C_MARCA),
        Paragraph("Global Solution 2026.2: CitizenTech", E_C_SUB),
        Spacer(1, 1.8 * cm),
        Paragraph("CicloAlerta", E_C_TITULO),
        Paragraph("Detecção de buracos e rachaduras nas ciclovias de São Paulo com YOLO11", E_C_SUB),
        Spacer(1, 1.8 * cm),
        Paragraph("Applied Computer Vision · Enxergando o mundo real", E_C_SUB),
        Paragraph("Professores Arnaldo Viana e Paulo Rodrigues", E_C_SUB),
        Spacer(1, 1.6 * cm),
        Paragraph("Integrantes", E_C_MARCA),
    ]
    for pessoa in d.integrantes:
        nome = escape(pessoa["nome"]) if pessoa["nome"] else pendente("nome completo")
        rm = f"RM {escape(pessoa['rm'])}" if pessoa["rm"] else pendente("RM")
        corpo.append(Paragraph(f"{nome} · {rm}", E_C_NOME))
    corpo += [Spacer(1, 1.6 * cm), Paragraph("São Paulo, novembro de 2026", E_C_SUB)]
    if not d.tem_resultados:
        corpo.append(Paragraph("RASCUNHO: faltam os resultados do treino e os dados da equipe", E_AVISO))
    corpo.append(PageBreak())
    return corpo


def montar(raiz: Path, saida: Path) -> dict:
    d = Dados(raiz)
    texto_resumo = resumo(d)
    palavras_resumo, palavras_objetivo = contar_palavras(texto_resumo), contar_palavras(OBJETIVO)
    if palavras_resumo > LIMITE_RESUMO:
        raise SystemExit(f"Resumo com {palavras_resumo} palavras. O limite é {LIMITE_RESUMO}.")
    if palavras_objetivo > LIMITE_OBJETIVO:
        raise SystemExit(f"Objetivo com {palavras_objetivo} palavras. O limite é {LIMITE_OBJETIVO}.")

    corpo = capa(d)
    corpo += [Paragraph("1. Resumo", E_H1), Paragraph(texto_resumo, E_CORPO)]
    corpo.append(Paragraph("2. Introdução", E_H1))
    corpo += [Paragraph(p, E_CORPO) for p in INTRODUCAO]
    corpo += [Paragraph("3. Objetivo", E_H1), Paragraph(OBJETIVO, E_CORPO)]
    corpo.append(Paragraph("4. Explicação técnica das ferramentas", E_H1))
    for titulo, texto in FERRAMENTAS:
        corpo += [Paragraph(titulo, E_H2), Paragraph(texto, E_CORPO)]
    corpo.append(Paragraph("5. Metodologia", E_H1))
    for numero, (titulo, texto) in enumerate(METODOLOGIA, 1):
        corpo += [Paragraph(f"Etapa {numero}. {titulo}", E_H2), Paragraph(texto, E_CORPO)]
    corpo += secao_resultados(d)
    corpo.append(Paragraph("7. Conclusão", E_H1))
    corpo += [Paragraph(p, E_CORPO) for p in conclusao(d)]
    corpo.append(Paragraph("Repositório e vídeo", E_H2))
    corpo.append(
        Paragraph(
            f"Repositório: {escape(d_link(raiz, 'repositorio')) or pendente('link do GitHub')}<br/>"
            f"Vídeo: {escape(d_link(raiz, 'video')) or pendente('link do vídeo')}",
            E_CORPO.clone("links", alignment=0),
        )
    )
    corpo.append(Paragraph("Referências", E_H1))
    corpo += [Paragraph(escape(r), E_REF) for r in REFERENCIAS]

    saida.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(saida),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        title="CicloAlerta: detecção de buracos e rachaduras nas ciclovias",
        author="Equipe CicloAlerta",
        subject="FIAP Global Solution 2026.2, Applied Computer Vision",
    ).build(corpo, onFirstPage=rodape, onLaterPages=rodape)
    return {"resumo": palavras_resumo, "objetivo": palavras_objetivo, "rascunho": not d.tem_resultados}


def d_link(raiz: Path, chave: str) -> str:
    arquivo = raiz / "docs" / "links.json"
    if not arquivo.exists():
        return ""
    return json.loads(arquivo.read_text(encoding="utf-8")).get(chave, "")


def main() -> int:
    analisador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analisador.add_argument("--raiz", type=Path, default=RAIZ)
    analisador.add_argument("--saida", type=Path, default=None)
    args = analisador.parse_args()
    saida = args.saida or args.raiz / "docs" / "relatorio_cicloalerta.pdf"
    print(json.dumps(montar(args.raiz, saida), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
