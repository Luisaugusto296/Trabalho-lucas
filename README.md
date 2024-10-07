# Documentação do Código de Processamento de Imagens

Autor: Luis Augusto

Este documento descreve um código para processamento de imagens utilizando as bibliotecas OpenCV e Matplotlib, que foram solicitadas mediante atividade disponivilizada pelo professor lucas.
Foi ultilizado o jupiter notebook, o mesmo se encontra aqui neste projeto
Bibliotecas Utilizadas: 
- **OpenCV**: Para manipulação e processamento de imagens.
- **NumPy**: Para operações numéricas.
- **Matplotlib**: Para visualização de imagens., logo abaixo esta o codigo devidamente comentado explixanto cada passo

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para exibir a imagem
def show_image(image, title='Imagem'):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Converte de BGR para RGB
    plt.title(title)
    plt.axis('off')
    plt.show()

# Leitura de imagens
imagem = cv2.imread('natureza.png')  # Lê a imagem original
imagem2 = cv2.imread('R.jpeg')        # Lê outra imagem
cv2.imshow("dragon ball", imagem2)    # Exibe a segunda imagem
imagem3 = cv2.imread('OIP.bmp')       # Lê mais uma imagem
cv2.imshow("superman", imagem3)       # Exibe a terceira imagem
cv2.waitKey(0)                        # Aguarda uma tecla
cv2.destroyAllWindows()               # Fecha as janelas abertas

# 1. Pré-processamento de Imagens
# Conversão de Cores: Converte a imagem original para escala de cinza
imagem_com_cor_diferente = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
show_image(imagem_com_cor_diferente, 'Imagem em Cinza')

# Redimensionamento: Redimensiona a imagem para 200x200 pixels
imagem_redimensionada = cv2.resize(imagem, (200, 200))
show_image(imagem_redimensionada, 'Imagem com tamanho redimensionada')

# Equalização de Histograma: Melhora o contraste da imagem em escala de cinza
equalizaacao = cv2.equalizeHist(imagem_com_cor_diferente)
show_image(equalizaacao, 'equalizacao')

# 2. Aplicação de Filtros
# Desfoque (Blur): Aplica um filtro Gaussiano para suavizar a imagem
filtrodaimagem = cv2.GaussianBlur(imagem, (5, 5), 0)
show_image(filtrodaimagem, 'Imagem com filtro')

# Detecção de Bordas (Canny): Detecta bordas na imagem em escala de cinza
bordas = cv2.Canny(imagem_com_cor_diferente, 100, 200)
show_image(bordas, 'Bordas')

# 3. Detecção de Características
# Detecção de Cantos (Harris): Detecta cantos na imagem e os destaca
cantos = cv2.cornerHarris(imagem_com_cor_diferente, 2, 3, 0.04)
harris = imagem.copy()
harris[cantos > 0.01 * cantos.max()] = [0, 0, 255]  # Destaca os cantos em vermelho
show_image(harris, 'CantosHarris')

# Detecção de Contornos: Encontra contornos na imagem em escala de cinza
contornos, _ = cv2.findContours(imagem_com_cor_diferente, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contorno = cv2.drawContours(imagem.copy(), contornos, -1, (0, 255, 0), 2)  # Desenha contornos em verde
show_image(contorno, 'detectacao de contornos')

# Pontos de Interesse (SIFT): Detecta e desenha pontos de interesse
sift = cv2.SIFT_create()
keypoints, descritores = sift.detectAndCompute(imagem_com_cor_diferente, None)
sift = cv2.drawKeypoints(imagem, keypoints, None)
show_image(sift, 'SIFT')

# 4. Transformações Geométricas
# Rotação: Rotaciona a imagem em torno do seu centro
altura, largura = imagem.shape[:2]
rotacao = cv2.getRotationMatrix2D((largura/2, altura/2), 45, 1)
imagem_rotacionada = cv2.warpAffine(imagem, rotacao, (largura, altura))
show_image(imagem_rotacionada, 'Imagem Rotacionata')

# Translação: Translada a imagem em 50 pixels tanto em x quanto em y
translacao = np.float32([[1, 0, 50], [0, 1, 50]])
imagem_translada = cv2.warpAffine(imagem, translacao, (largura, altura))
show_image(translacao, 'translacao')

# 5. Operações Morfológicas
# Erosão: Reduz as dimensões das áreas brancas na imagem
kernel = np.ones((5, 5), np.uint8)
erosao = cv2.erode(imagem_com_cor_diferente, kernel, iterations=1)
show_image(erosao, 'erosao')

# Dilatação: Aumenta as dimensões das áreas brancas na imagem
imagemdilatada = cv2.dilate(imagem_com_cor_diferente, kernel, iterations=1)
show_image(imagemdilatada, 'dilatacao')

# 6. Segmentação de Imagens
# Limiarização: Converte a imagem em uma imagem binária
_, limiar = cv2.threshold(imagem_com_cor_diferente, 127, 255, cv2.THRESH_BINARY)
show_image(limiar, 'Limiarizacao')
