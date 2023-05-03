# chroma-key-visao

### Repositorio com a correção do exercício de visão computacional

### Objetivo: Utilizar Processamento Digital de Imagens para um algoritmo de Chroma Key. 

### Passo a Passo:

1. Leitura das imagens com HSV para facilitar a manipulação
2. Criação das máscaras para cada imagem com o intervalo de verde desejado.
3. Inversão da máscara para colocar o background.
4. Criação de uma máscara com os tons de verdes olhando o canal Hue do HSV da imagem inteira. 
5. Em cima da mascara criada no passo anterior diminuo a Saturation do HSV para equilibrar os tons de verde.
5. Reconstruo a imagem balanceada com os tons de verde acinzentados.
6. Redimensiono a imagem de background para o shape da imagem original e trunco a interpolação em 1.
7. Aplico a máscara para caso seja background pegar a imagem redimensionada, senão pegar a imagem balanceada com os tons de verde acinzentados.
8. Crio uma mascara com Sobel para tratar os contornos em cima da máscara invertida do passo 3.
9. Aonde a máscara de Sobel for diferente de 0 eu pego a imagem final do passo 7 borrada, senão pego a mesma imagem sem borrar. 
10. Terminou! :D
