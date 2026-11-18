# Contrato de integração do componente de visão

Este é o que os outros componentes do CicloAlerta (dados, automação, IA generativa, MVP) podem esperar da visão.

## Ocorrências em arquivo

O detector grava uma linha JSON por defeito confirmado em `saida/ocorrencias.jsonl`. Um defeito é confirmado depois de aparecer em 3 quadros, e cada linha é escrita uma única vez.

| Campo | Tipo | Significado |
|---|---|---|
| `id` | texto | Identificador da ocorrência |
| `versao_contrato` | texto | Versão deste contrato, hoje `1.0` |
| `registrada_em` | texto | Data e hora em UTC, formato ISO 8601 |
| `classe` | texto | `buraco` ou `rachadura` |
| `confianca` | número | Maior confiança do modelo, de 0 a 1 |
| `quadros_observados` | inteiro | Em quantos quadros o defeito já tinha aparecido quando foi registrado |
| `caixa_normalizada` | lista | `[x1, y1, x2, y2]`, de 0 a 1 em relação ao quadro |
| `fonte` | texto | `camera-0`, o host da URL ou o nome do arquivo, sem caminho local |
| `modelo` | texto | Nome do modelo usado |
| `status` | texto | Sempre nasce como `pendente_revisao` |
| `latitude`, `longitude` | número ou nulo | Só existem se forem informadas com `--latitude` e `--longitude` |
| `recorte` | texto ou nulo | Caminho do recorte, relativo à pasta de saída |

Exemplo:

```json
{
  "id": "a1b2c3d4e5f6",
  "versao_contrato": "1.0",
  "registrada_em": "2026-11-10T14:03:22+00:00",
  "classe": "buraco",
  "confianca": 0.81,
  "quadros_observados": 3,
  "caixa_normalizada": [0.31, 0.62, 0.48, 0.78],
  "fonte": "rua.mp4",
  "modelo": "cicloalerta",
  "status": "pendente_revisao",
  "latitude": null,
  "longitude": null,
  "recorte": "recortes/a1b2c3d4e5f6.jpg"
}
```

Os valores acima são só uma ilustração do formato.

A confiança mede o quanto o modelo acredita que aquilo é um defeito. Ela não mede a gravidade do defeito, e não deve ser usada para priorizar conserto.

## API HTTP

```bash
python -m cicloalerta.api --modelo modelos/cicloalerta.onnx --porta 8000
```

`GET /saude` responde com as classes do modelo:

```json
{"status": "ok", "classes": ["buraco", "rachadura"]}
```

`POST /api/detectar` é a rota combinada no contrato do MVP. Ela recebe `multipart/form-data` com a foto no campo `imagem` (o campo `foto` também é aceito), até 8 MB. A rota antiga, `POST /v1/deteccoes`, faz exatamente o mesmo.

```json
{
  "largura": 1280,
  "altura": 960,
  "tempo_ms": 54.2,
  "status": "pendente_revisao",
  "tem_defeito": true,
  "deteccoes": [
    {"classe": "buraco", "confianca": 0.41, "caixa": [402.0, 446.5, 611.3, 560.0], "severidade_sugerida": "media"}
  ]
}
```

- `caixa` é `[x1, y1, x2, y2]` em pixels da foto enviada.
- As detecções já passaram pelos limiares calibrados na validação (0,28 para buraco e 0,20 para rachadura). O site não deve aplicar outro corte, como 0,5: o modelo raramente passa disso e quase nenhum relato seria criado.
- `severidade_sugerida` vem de uma regra provisória pelo tamanho da caixa na foto. Buraco que ocupa 4% da foto ou mais é `alta`, e o resto é `media`. Rachadura que ocupa 10% ou mais é `media`, e o resto é `baixa`. A regra não foi validada e a revisão humana confirma.
- A foto não é gravada. Se o site guardar a foto, ele precisa apagar os metadados (EXIF) antes, porque eles podem ter a localização e o aparelho de quem tirou.

| Código | Quando |
|---|---|
| 200 | Foto processada, mesmo sem defeito (`tem_defeito: false`) |
| 400 | Nenhum arquivo nos campos `imagem` ou `foto` |
| 413 | Arquivo maior que 8 MB |
| 415 | O arquivo não é uma imagem válida |

A API não tem autenticação e sobe em `127.0.0.1`. Para expor na internet, ela precisa ficar atrás do servidor do site ou de um gateway com autenticação.

## Integração com o site

O passo a passo para quem cuida da API do site está em [integracao_site.md](integracao_site.md).
