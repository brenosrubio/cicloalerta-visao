# Roteiro do vídeo de demonstração

O vídeo é gravado no OBS: os slides da apresentação e, no meio, o detector rodando de verdade no VS Code, só com CPU. Lendo as falas inteiras, ele fica com uns 7 a 8 minutos.

## Antes de gravar

Deixe na pasta do projeto:

- `ciclovia.mp4` (também em `docs/demo/ciclovia.mp4`): os primeiros 23 segundos do vídeo que a equipe gravou numa ciclovia de São Paulo, já convertidos para 15 quadros por segundo (`ffmpeg -i original.mp4 -t 23 -vf fps=15 -an ciclovia.mp4`). O trecho seguinte é de ciclovia em bom estado, sem nada para marcar.
- A foto da mesma ciclovia já vem no repositório, em `docs/demo/foto_ciclovia.jpg`.

Abra dois terminais no VS Code. No segundo, deixe a API ligada:

```powershell
.venv\Scripts\python -m cicloalerta.api --porta 8000
```

Feche abas e notificações que mostrem dados pessoais. Nenhuma tela deve mostrar rosto ou placa de perto.

## Falas, slide por slide

### 1. CicloAlerta

Este é o CicloAlerta, o nosso MVP da Global Solution: um site que vai mostrar no mapa onde estão os buracos e os trechos perigosos das ciclovias de São Paulo. Aqui eu mostro o componente de visão computacional, da disciplina de Applied Computer Vision.

### 2. Buraco na ciclovia derruba, e a chuva esconde o buraco

Para quem pedala, um buraco ou um desnível é risco de queda, e na chuva a água esconde o defeito. Em 2025, uma auditoria cidadã percorreu 715 quilômetros de ciclovias de São Paulo e classificou o pavimento de 20,5% dos trechos como precário, ruim ou razoável. Os sinistros com ciclistas em janeiro de 2026 foram 15% maiores que em janeiro de 2024. A proposta do CicloAlerta é mostrar esses perigos no mapa e avisar quando a chuva deixa um trecho mais arriscado. O mapa e os alertas ficam com as outras matérias do projeto. Aqui eu mostro a parte de visão computacional, que procura buracos e rachaduras nas fotos e no vídeo.

### 3. Da foto do ciclista ao mapa

O componente de visão tem duas entradas. Na tela Reportar, o ciclista manda a foto de um buraco e o modelo confirma se há defeito. Em tempo real, ele analisa o vídeo de uma câmera comum. Nos dois casos, o YOLO11n em ONNX roda só no processador, um filtro aplica o limiar de cada classe, e no vídeo o rastreador só confirma um defeito depois de três quadros. O resultado vira um relato que uma pessoa revisa antes de o ponto aparecer no mapa.

### 4. 6.659 fotos reais de ruas de três países

Procuramos fotos de ciclovias, mas a única base pública de defeitos em ciclovia, de Barcelona, publicou só as marcações, sem as imagens. Então usamos a base pública RDD2022, com fotos reais de ruas da Índia, do Japão e da República Tcheca. O asfalto é o mesmo de muitas ciclofaixas, que são pintadas na própria rua. As categorias de trinca viraram uma classe, rachadura, e a de buraco ficou como buraco. Antes de treinar, conferimos as marcações: em uma das cópias da base, a classe chamada pothole marcava faixas de pedestre e sinalização, então pegamos os buracos de outra cópia. E separamos as imagens por blocos, para que fotos quase iguais não caíssem no treino e no teste ao mesmo tempo.

### 5. YOLO11n: 50 épocas em 66 minutos

Fizemos o ajuste fino de um YOLO11n, a menor versão da família, partindo de pesos já treinados no COCO. Foram 50 épocas na GPU T4 do Colab, em 66 minutos. A curva mostra o mAP50 na validação subindo até 0,496. A melhor época foi a última, o que indica que mais épocas ainda podem ajudar.

### 6. Agora, o vídeo

Diga: “Agora, o detector rodando.” Vá para o VS Code e rode no primeiro terminal:

```powershell
.venv\Scripts\python -m cicloalerta.detectar --fonte ciclovia.mp4 --ignorar-base 0.3
```

- **Ao dar Enter:** “Este vídeo a gente gravou numa ciclovia de São Paulo, com o celular preso no guidão. O ignorar-base faz o programa desconsiderar a parte de baixo da imagem, onde aparece o guidão.”
- **Com a janela aberta:** “Ele roda só no processador, sem placa de vídeo, a uns quinze quadros por segundo, que é o número verde no canto. A caixa laranja é rachadura, e a vermelha é buraco. Um defeito só é registrado depois de aparecer em três quadros.”
- **Nos primeiros segundos, quando a caixa laranja pegar a pintura rachada:** “Aqui ele marca a pintura trincada da ciclovia como rachadura.”
- **No meio do vídeo, quando aparecerem as caixas finas e compridas:** “Essas linhas finas são as juntas entre as placas de concreto, que para o modelo parecem trinca. Por isso tudo passa por revisão.”
- **Quando a janela fechar:** “No fim, o programa mostra o desempenho e quantas ocorrências registrou. O buraco registrado é a borda quebrada do começo, que eu mostro no slide de erros.”

Depois, no primeiro terminal (a API fica ligada no segundo):

```powershell
curl.exe -s -F imagem=@docs\demo\foto_ciclovia.jpg http://127.0.0.1:8000/api/detectar | .venv\Scripts\python -m json.tool
```

- **Com a resposta na tela:** “Esta é a rota que a tela Reportar do site vai chamar. Mando uma foto que tirei dessa ciclovia, e a resposta volta com tem_defeito verdadeiro, a caixa de cada defeito, a confiança e uma severidade sugerida, sempre pendente de revisão.”

Volte para os slides.

### 7. Cada defeito confirmado vira um registro

Cada defeito confirmado vira um registro em JSON, que os outros componentes do produto consomem. Ele nasce como pendente de revisão: o modelo sugere e uma pessoa decide. Para respeitar a LGPD, guardamos só o recorte do pavimento, nunca o quadro inteiro, e a localização só entra se for informada.

### 8. 665 fotos que o modelo nunca viu

No teste, com fotos que o modelo nunca viu, chegamos a mAP50 de 0,50. Nos limiares que calibramos na validação, ele acerta 63% das vezes que aponta um buraco e encontra cerca de metade dos defeitos em uma foto isolada. É um resultado para triagem, com uma pessoa revisando.

### 9. Medimos por país para enxergar viés

Medimos também por país, para enxergar viés: o modelo foi melhor no Japão do que na Índia e na República Tcheca. No caso tcheco, o teste tinha só 12 buracos, então o número oscila bastante. O maior limite é que nenhuma dessas bases tem ciclovias nem ruas brasileiras.

### 10. Fotos de perto: o ponto fraco

A tela Reportar recebe fotos de alguém em pé ao lado do buraco, mas o modelo aprendeu com câmera de carro. Para medir isso, recortamos um quadrado em volta de cada defeito das amostras do teste. Quanto mais perto, menos defeitos ele acha: nos buracos, caiu de 18 para 12 em 51. Por isso, quando o modelo não acha nada, o site deixa o ciclista enviar o relato mesmo assim.

### 11. Com água, o modelo acha menos buracos

Para medir o cenário de chuva, usamos 713 fotos de celular de buracos de uma base pública britânica, muitas com água. Separamos as fotos de teste em secas e com água. Nas secas, o modelo aponta o buraco em 77% das fotos; com água, em 61%. Os buracos grandes cheios de água, fotografados de perto, são os que ele mais perde. Isso confirma o problema do CicloAlerta e mostra qual deve ser o próximo treino.

### 12. Tempo real sem placa de vídeo

O modelo roda em tempo real sem placa de vídeo: no nosso computador, só com o processador, ficou entre 14 e 16 quadros por segundo. Para rodar fora do PyTorch, exportamos para ONNX e conferimos que as caixas saem iguais.

### 13. Onde o modelo acerta

À esquerda, um quadro do vídeo que gravamos na ciclovia: a pintura trincada foi marcada como rachadura, com confiança 0,75. À direita, fotos do teste da base em que as caixas do modelo caem em cima das marcações humanas.

### 14. Onde o modelo erra

Ele também erra. Na nossa ciclovia, esta borda quebrada entre a faixa e o concreto ficou no limite. Ela é o buraco que apareceu no resumo da demonstração, mas no vídeo original passou do limiar em só dois quadros e não foi registrada, porque o detector exige três. O vão de uma travessia, mais adiante na mesma ciclovia, nem passou do limiar. São degraus e emendas de ciclovia, e o modelo aprendeu buraco redondo de asfalto de rua. O guidão também virou buraco, até usarmos a opção ignorar-base. Num vídeo de rua, marcas de pneu, faixas pintadas e uma tampa de bueiro viraram defeito. Na grade da base, as caixas verdes sem caixa do modelo por cima são defeitos que ele não achou. É por isso que toda ocorrência nasce pendente de revisão.

### 15. O que ainda falta

O maior limite é que o treino não tem fotos de ciclovia nem do Brasil, e com água o modelo acha menos buracos. Na ciclovia que gravamos, ele marcou a pintura trincada, mas não registrou degraus e emendas quebradas, que não existem na base de treino. O próximo passo é um ajuste fino com fotos de buracos com água, que não coube no prazo porque o Colab ficou sem GPU. Depois, o próprio produto ajuda: as fotos que chegam pela tela Reportar, já revisadas, viram a base para treinar de novo. Os erros que vimos, como marcas de pneu e tampas de bueiro, entram como exemplos do que não é defeito.

### 16. CicloAlerta

Esse foi o componente de visão do CicloAlerta. O código, o modelo e o relatório estão no GitHub, e o link também está no relatório. Obrigado.

## Depois de gravar

- [ ] Publique no YouTube como não listado.
- [ ] Teste o link numa janela anônima.
- [ ] Coloque o link em `docs/links.json` e no README.
- [ ] Refaça o relatório com `python ferramentas/gerar_relatorio.py`.
