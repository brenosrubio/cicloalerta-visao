# Roteiro do vídeo de demonstração

O vídeo vale 5 dos 10 pontos da disciplina, e a regra pede a solução funcionando. Por isso a maior parte do tempo mostra o programa rodando. O slide não define duração. A sugestão é ficar em torno de 3 minutos.

## Antes de gravar

- [ ] Rode o aplicativo uma vez no computador que vai aparecer no vídeo (veja o README).
- [ ] Grave de 1 a 2 minutos de rua com o celular. Quem filma não pode ser quem dirige. Filmar com o carro parado ou a pé em uma rua com buracos também serve.
- [ ] Evite enquadrar rostos e placas de perto.
- [ ] Deixe abertos: o terminal, a pasta `saida/` e os gráficos em `resultados/teste/`.
- [ ] Teste a captura de tela e o microfone.

Se não der para filmar a rua, use `docs/demo/demo_rdd2022.mp4`, que já está no repositório. Nesse caso, diga no vídeo que são imagens do teste da base, sem relação com São Paulo, e que o movimento de aproximação foi simulado.

## Roteiro

### 0:00 a 0:20. O problema

**Tela:** o rosto de quem fala, ou uma foto de buraco tirada pela equipe.

> "No primeiro semestre de 2026, São Paulo recebeu quase 52 mil pedidos de tapa-buraco, e 38% ainda estavam abertos no começo de agosto. Muitas vezes o defeito só entra na fila quando alguém reclama. A Via Atenta usa uma câmera comum para encontrar esse defeito enquanto o carro passa."

### 0:20 a 0:50. Como treinamos

**Tela:** o notebook no Colab e `resultados/treino/results.png`.

> "Fizemos o ajuste fino de um YOLO11n com a base pública RDD2022, usando imagens reais de ruas da Índia, do Japão e da República Tcheca. As categorias originais viraram duas classes: buraco e rachadura. Conferimos os dados antes de treinar: em uma das fontes, a classe chamada pothole marcava faixas de pedestre, então trocamos a fonte dos buracos. Separamos as imagens por blocos, para que quadros quase iguais não caíssem no treino e no teste ao mesmo tempo."

### 0:50 a 1:50. Funcionando em tempo real

**Tela:** o terminal e, em seguida, a janela do detector. Este é o trecho principal, então deixe rodar sem cortes.

```bash
python -m via_atenta.detectar --fonte rua.mp4 --ignorar-topo 0.3
```

Se a equipe usar o celular ao vivo:

```bash
python -m via_atenta.detectar --fonte http://IP_DO_CELULAR:8080/video
```

> "Aqui o modelo está rodando em tempo real, só na CPU, sem placa de vídeo. O número no canto mostra os quadros por segundo. A caixa vermelha é buraco, a laranja é rachadura. Um defeito só aparece depois de ser visto em três quadros, o que corta detecções passageiras."

### 1:50 a 2:20. O que sai do outro lado

**Tela:** `saida/ocorrencias.jsonl` aberto, um recorte de `saida/recortes/`, e depois a API:

```bash
python -m via_atenta.api --porta 8000
curl -F imagem=@rua.jpg http://127.0.0.1:8000/v1/deteccoes
```

> "Cada defeito confirmado vira um registro em JSON, que os outros componentes do produto consomem. Ele nasce como pendente de revisão: o modelo sugere e uma pessoa decide. Para respeitar a LGPD, guardamos só o recorte do pavimento, nunca o quadro inteiro."

### 2:20 a 2:50. Resultados e limites

**Tela:** a tabela de resultados do README e a curva `resultados/teste/BoxPR_curve.png`.

> "No teste, com imagens que o modelo nunca viu, chegamos a mAP50 de 0,50, e o modelo acha cerca de metade dos defeitos em uma imagem isolada. É um resultado para triagem, com revisão humana. Medimos também por país, para enxergar viés: o modelo foi melhor no Japão do que na Índia e na República Tcheca. O maior limite é que a base não tem ruas brasileiras. O próximo passo é coletar imagens de São Paulo e treinar de novo."

Mostre um exemplo de erro de `resultados/exemplos/erros.jpg`. Isso mostra que a equipe entende o modelo.

### 2:50 a 3:00. Encerramento

> "Via Atenta. O código está no GitHub, e o link está no relatório."

## Depois de gravar

- [ ] Publique no YouTube como não listado.
- [ ] Teste o link em uma janela anônima.
- [ ] Coloque o link em `docs/links.json` e no README.
- [ ] Refaça o relatório com `python ferramentas/gerar_relatorio.py`.
