#!/usr/bin/env python3
"""Gera docs/relatorio_via_atenta.pdf na ordem pedida pela disciplina.

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
        "Quem anda por São Paulo conhece o problema: muitas vezes o buraco só entra na fila de conserto depois que alguém reclama. "
        "No primeiro semestre de 2026, a prefeitura recebeu quase 52 mil pedidos de tapa-buraco, e 38% deles ainda "
        "estavam abertos no começo de agosto. A Via Atenta propõe usar uma câmera comum, como a de um celular preso ao "
        "painel, para reconhecer defeitos na via enquanto o veículo circula. Este trabalho entrega o componente de "
        f"visão do produto. Fizemos o ajuste fino de um YOLO11n, pré-treinado no COCO, com {d.n_imagens} imagens reais "
        "de ruas da Índia, do Japão e da República Tcheca, vindas da base pública RDD2022. As quatro categorias "
        "originais viraram duas classes, buraco e rachadura, porque é essa diferença que pesa na triagem dos reparos. "
        "A divisão entre treino, validação e teste foi feita por blocos de imagens vizinhas, para que quadros "
        f"quase iguais não vazassem para o teste. No teste, o modelo alcançou mAP50 de {d.mapa50()}, com AP50 de "
        f"{d.ap50('buraco')} para buracos e {d.ap50('rachadura')} para rachaduras. Exportado para ONNX, rodou a "
        f"{d.fps()} quadros por segundo em CPU, sem placa de vídeo. Um rastreador simples junta as aparições do mesmo "
        "defeito e descarta ruído de um quadro só. Cada defeito confirmado vira uma ocorrência em JSON, pendente de "
        "revisão humana, e só o recorte do pavimento é guardado. A principal limitação é que a base não tem imagens "
        "brasileiras, então o desempenho nas nossas ruas ainda precisa ser medido."
    )


OBJETIVO = (
    "Treinar uma arquitetura YOLO11 com imagens reais de vias para detectar buracos e rachaduras em tempo real, em "
    "câmera comum e sem placa de vídeo, e entregar cada defeito confirmado como dado estruturado para a revisão humana "
    "e para os demais componentes da Via Atenta."
)

INTRODUCAO = [
    "A Via Atenta foi pensada para quem depende das vias urbanas todos os dias: motociclistas, ciclistas, passageiros "
    "de ônibus e pedestres com mobilidade reduzida, além das equipes de zeladoria que consertam o pavimento. Para "
    "essas pessoas, um buraco significa risco de queda, atraso e gasto com manutenção.",
    "Os números dão o tamanho da fila. Em São Paulo, a prefeitura recebeu quase 52 mil pedidos de tapa-buraco no "
    "primeiro semestre de 2026. No começo de agosto, 38% deles continuavam sem solução, embora a Secretaria Municipal "
    "de Subprefeituras informe tempo médio de atendimento de seis dias (BNEWS, 2026). Nas rodovias, a Pesquisa CNT de "
    "2025 avaliou mais de 114 mil quilômetros e classificou 62% da malha como regular, ruim ou péssima. O mesmo estudo "
    "calcula que o pavimento ruim encarece o transporte rodoviário em 31%, em média (PORTAL DO TRÂNSITO, 2025).",
    "O conserto costuma começar com uma pessoa que vê o defeito, abre um pedido e descreve o lugar. Assim, a fila "
    "depende de quem tem tempo para reclamar, e a prefeitura só fica sabendo do defeito quando alguém decide "
    "registrá-lo.",
    "Nossa proposta é usar câmeras que já existem para reconhecer o defeito enquanto o veículo circula. Este documento "
    "trata do componente de visão do produto, que corresponde à disciplina de Applied Computer Vision: um detector "
    "treinado com imagens reais, que roda em tempo real e entrega dados estruturados aos outros componentes.",
    "O projeto se conecta à ODS 11 (cidades e comunidades sustentáveis), porque ajuda a manter a infraestrutura urbana "
    "em condições de uso, e à ODS 9 (indústria, inovação e infraestrutura), porque barateia a inspeção das vias.",
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
        "Ultralytics 8.4 para treino e exportação, Google Colab com GPU T4, Flask para a API e pytest para os testes "
        "automatizados do repositório.",
    ),
]

METODOLOGIA = [
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
        "Usamos Índia, Japão e República Tcheca, onde as imagens vêm de celulares presos a carros, como no uso previsto "
        "(ARYA et al., 2022). Ficaram de fora a Noruega, que usa câmeras especiais, os Estados Unidos, com imagens do "
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
        "Cada defeito confirmado vira uma linha em ocorrencias.jsonl, e uma API HTTP atende quem preferir enviar "
        "imagens. As ocorrências nascem com status pendente_revisao, porque o modelo sugere e uma pessoa decide. Para "
        "seguir a LGPD, guardamos só o recorte do defeito, nunca o quadro inteiro, e a API descarta a imagem depois de "
        "responder. A localização é opcional.",
    ),
]

REFERENCIAS = [
    "ARYA, D. et al. RDD2022: a multi-national image dataset for automatic road damage detection. arXiv:2209.08538, "
    "2022. Disponível em: https://arxiv.org/abs/2209.08538.",
    "BNEWS. Mesmo após recapeamento, SP registra quase 52 mil pedidos de tapa-buraco. 9 ago. 2026. Disponível em: "
    "https://www.bnewssaopaulo.com.br/noticias/politica/mesmo-apos-recapeamento-sp-registra-quase-52-mil-pedidos-de-"
    "tapa-buraco.html.",
    "DATASET NINJA. RDD2022. Disponível em: https://datasetninja.com/road-damage-detector.",
    "DRONEFREAK. RDD2022 Multi-National Road Damage Detection Dataset (4-Class YOLO Export). Hugging Face. Revisão "
    "72b01f94. Disponível em: https://huggingface.co/datasets/dronefreak/RDD2022.",
    "EDWRDC. RDD2022. Hugging Face. Revisão d8967222. Disponível em: https://huggingface.co/datasets/edwrdc/RDD2022.",
    "ONU. Objetivos de Desenvolvimento Sustentável. Disponível em: https://brasil.un.org/pt-br/sdgs.",
    "PORTAL DO TRÂNSITO. Qualidade das rodovias brasileiras melhorou em 2025, diz CNT. 21 dez. 2025. Disponível em: "
    "https://www.portaldotransito.com.br/noticias/mobilidade-e-tecnologia/seguranca/qualidade-das-rodovias-"
    "brasileiras-melhorou-em-2025-diz-cnt/.",
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

    arquivo = d.pasta / "exemplo_saida" / "ocorrencias.jsonl"
    if arquivo.exists() and arquivo.stat().st_size:
        primeira = json.loads(arquivo.read_text(encoding="utf-8").splitlines()[0])
        corpo.append(Paragraph("Ocorrência gerada", E_H2))
        corpo.append(Paragraph("Este é o dado que os outros componentes do MVP recebem:", E_CORPO))
        texto = json.dumps(primeira, ensure_ascii=False, indent=2)
        texto = re.sub(r"\[\s+([^\]]+?)\s+\]", lambda c: "[" + " ".join(c[1].split()) + "]", texto)
        corpo.append(Preformatted(texto, E_CODIGO))
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


def conclusao(d: Dados) -> list[str]:
    if d.m:
        classes = d.m["teste"]["por_classe"]
        abertura = (
            f"O detector cumpriu o que a atividade pede: foi treinado sob medida para o problema, chegou a mAP50 de "
            f"{d.mapa50()} em imagens inéditas e rodou a cerca de {d.fps()} quadros por segundo em CPU, sem placa de "
            f"vídeo. O desempenho é moderado. No limiar calibrado, o modelo acha "
            f"{num(classes['buraco']['revocacao_no_limiar'] * 100, 0)}% dos buracos e "
            f"{num(classes['rachadura']['revocacao_no_limiar'] * 100, 0)}% das rachaduras em uma imagem isolada. Isso "
            "serve para triagem com revisão humana, e não para substituir a inspeção da via. Em vídeo, o mesmo defeito "
            "aparece em vários quadros e o modelo tem mais de uma chance, mas não medimos esse ganho."
        )
    else:
        abertura = pendente("Parágrafo de abertura escrito com os números reais do treino.")
    return [
        abertura,
        leitura_das_classes(d),
        "O limite mais sério é a origem das imagens. A base não tem ruas brasileiras, e as condições de asfalto, luz e "
        "tráfego são outras. O próximo passo é coletar imagens em São Paulo, medir o modelo nelas e usá-las em um novo "
        "ajuste fino. O rastreador também tem limite: em velocidade alta, a sobreposição entre quadros cai e o mesmo "
        "defeito pode ser contado duas vezes. Por isso a contagem é uma aproximação, e cada ocorrência passa por revisão "
        "humana antes de virar chamado.",
    ]


def rodape(canvas, documento) -> None:
    if documento.page == 1:
        return
    canvas.saveState()
    canvas.setFont(NORMAL, 9)
    canvas.setFillColor(COR_SUAVE)
    canvas.drawString(2.5 * cm, 1.4 * cm, "Via Atenta · Applied Computer Vision · FIAP")
    canvas.drawRightString(A4[0] - 2.5 * cm, 1.4 * cm, str(documento.page))
    canvas.restoreState()


def capa(d: Dados) -> list:
    corpo: list = [
        Spacer(1, 3.2 * cm),
        Paragraph("FIAP · Engenharia de Software · 4º ano", E_C_MARCA),
        Paragraph("Global Solution 2026.2: CitizenTech", E_C_SUB),
        Spacer(1, 1.8 * cm),
        Paragraph("Via Atenta", E_C_TITULO),
        Paragraph("Detecção de buracos e rachaduras em tempo real com YOLO11", E_C_SUB),
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
        title="Via Atenta: detecção de buracos e rachaduras em tempo real",
        author="Equipe Via Atenta",
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
    saida = args.saida or args.raiz / "docs" / "relatorio_via_atenta.pdf"
    print(json.dumps(montar(args.raiz, saida), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
