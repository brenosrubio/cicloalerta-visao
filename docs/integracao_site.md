# Integração com o site do CicloAlerta

Este guia é para quem cuida da API do site (FastAPI). Ele mostra como colocar a rota `POST /api/detectar` no ar usando o modelo deste repositório, que alimenta a tela **Reportar**.

## 1. Copie estes arquivos para o repositório do site

```text
cicloalerta/__init__.py
cicloalerta/detector.py
cicloalerta/analise.py
modelos/cicloalerta.onnx
modelos/limiares.json
```

`analise.py` não depende do Flask, então o site não precisa instalar nada além de:

```text
numpy>=1.26
opencv-python-headless>=4.9
onnxruntime>=1.20
python-multipart>=0.0.9
```

Use `opencv-python-headless` no container, porque ele não traz as bibliotecas de janela.

## 2. Crie a rota

```python
# app/visao.py
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from cicloalerta.analise import analisar_imagem
from cicloalerta.detector import DetectorONNX

TAMANHO_MAXIMO = 8 * 1024 * 1024

router = APIRouter()
detector = DetectorONNX(Path("modelos/cicloalerta.onnx"))  # lê limiares.json da mesma pasta


@router.post("/api/detectar")
async def detectar(imagem: UploadFile = File(...)):
    dados = await imagem.read()
    if len(dados) > TAMANHO_MAXIMO:
        return JSONResponse({"erro": "Imagem maior que 8 MB."}, status_code=413)
    corpo, codigo = analisar_imagem(dados, detector)
    return JSONResponse(corpo, status_code=codigo)
```

No `main.py` do site:

```python
from app.visao import router as visao
app.include_router(visao)
```

O modelo é carregado uma vez, quando o servidor sobe. Cada foto leva cerca de 50 a 100 ms em CPU.

## 3. O que o site faz com a resposta

A resposta tem o formato descrito em [contrato_ocorrencias.md](contrato_ocorrencias.md). Na tela Reportar:

1. O site envia a foto no campo `imagem`.
2. Se `tem_defeito` for `true`, o site cria o relato com `POST /api/relatos`:
   - `origem`: `visao`
   - `tipo`: a `classe` da detecção mais confiante (`buraco` ou `rachadura`)
   - `severidade`: a `severidade_sugerida` dessa detecção
   - `lat` e `lon`: a posição informada pelo celular, só do ponto do buraco
3. Se `tem_defeito` for `false`, o site avisa que não encontrou defeito e deixa a pessoa enviar mesmo assim, como relato com `origem` `app`. O modelo erra, e a pessoa que está na frente do buraco sabe mais do que ele.

Três cuidados:

- **Não aplique outro corte de confiança.** As detecções já vêm filtradas pelos limiares calibrados (0,28 para buraco e 0,20 para rachadura). Um corte de 0,5 faria o site quase nunca criar relato.
- **Apague o EXIF** antes de guardar a foto, se o site guardar. Os metadados podem ter a localização exata e o modelo do celular. Com o Pillow, basta salvar uma cópia sem o campo `exif`.
- **O relato nasce para revisão.** A resposta vem com `status: pendente_revisao`; o modelo sugere e uma pessoa confirma.

## 4. Teste rápido sem o site

Com este repositório:

```bash
python -m cicloalerta.api --porta 8000
curl -F imagem=@foto.jpg http://127.0.0.1:8000/api/detectar
```

A rota do Flask e a do FastAPI chamam a mesma função `analisar_imagem`, que é coberta pelos testes em `tests/test_api.py`.
