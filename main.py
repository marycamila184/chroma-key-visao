# ===============================================================================
# Tarefa - Chroma Key com suavização de contornos e balanceamento dos tons de verde.
# -------------------------------------------------------------------------------
# Autor: Mary Camila
# ===============================================================================

import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

BACKGROUND_IMG = 'background.jpg'

# ============================================================================================================================
# Metodos comuns para as funções
# ============================================================================================================================


def binarizeImage(img, threshold):
    imgOut = np.zeros((img.shape[0], img.shape[1], 1), img.dtype)
    # Binarizo a imagem baseado no threshold passado por parametro
    imgOut = np.where(img > threshold, 1, 0)
    imgOut = imgOut.reshape(imgOut.shape[0], imgOut.shape[1], 1)
    imgOut = imgOut.astype(np.float32)

    return imgOut


def gaussBlurImage(img, size):
    imgOut = np.zeros((img.shape[0], img.shape[1]), img.dtype)
    # Borro a imagem usando o metodo gaussiano para ter mais controle
    imgOut = cv2.GaussianBlur(img, (size, size), 0)

    return imgOut


def applyMask(img, mask):
    imgOut = np.zeros((img.shape[0], img.shape[1], 3), img.dtype)
    imgOut = cv2.bitwise_and(img, img, mask=mask)

    return imgOut


def applySobel(img, size):
    imgOut = np.zeros((img.shape[0], img.shape[1], 1), img.dtype)
    # Aplicando sobel no eixo X
    sobelX = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=size)
    # Aplicando sobel no eixo y
    sobelY = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=size)
    # Magnitude - sqrt(dx^2 + dy^2)
    imgOut = cv2.magnitude(sobelX, sobelY)
    imgOut = imgOut.reshape(imgOut.shape[0], imgOut.shape[1], 1)
    imgOut = np.clip(imgOut, 0, 1)  # Truncando em 1

    return imgOut

# ============================================================================================================================
# Metodo para remocao do fundo verde
# ============================================================================================================================

def removeGreenBackground(imagesListHsv):
    listMasks = []

    for i in range(0, len(imagesListHsv)):
        imgOut = np.zeros(
            (imagesListHsv[i].shape[0], imagesListHsv[i].shape[1], 3), imagesListHsv[i].dtype)

        # H - Hue - Pego somente o tom de verde que vai de aprox de 75 a 120 graus - Convertendo para 0...255 Fica em 45 ate 70
        # S - Saturation - Comeco em 0.4 para retirar os pixels mais brancos e defino o limite da saturacao em 1 para pegar o verde mais puro.
        # V - Value/Brightness - Comeco em 0.4 para retirar os pixels mais escuros e defino o limite em 1 para pegar o verde sem tom preto.
        lowerGreen = np.array([0.19, 0.39, 0.39])
        upperGreen = np.array([0.28, 1, 1])

        # Crio minha mascara para aplicar o background
        imgOut = createMaskColor(lowerGreen, upperGreen, imagesListHsv[i])
        #cv2.imwrite(str(i+10)+" - binaria.bmp", imgOut * 255)

        listMasks.append(imgOut)

    return listMasks

def createMaskColor(lowColor, upColor, img):
    # Gero a mascara invertida
    imgOut = cv2.inRange(img, lowColor, upColor)
    # Seto a mascara como float para padronizar
    imgOut = imgOut.astype(np.float32)
    imgOut = imgOut / 255

    return imgOut

# ============================================================================================================================
# Metodo para colocar a imagem no chroma key
# ============================================================================================================================

def applyChromaKey(imagesList, maskList):
    imgback = cv2.imread(BACKGROUND_IMG)
    imgback = imgback.reshape(imgback.shape[0], imgback.shape[1], 3)
    imgback = imgback.astype(np.float32) / 255

    for i in range(0, len(imagesList)):
        imgOut = np.zeros(
            (imagesList[i].shape[0], imagesList[i].shape[1], 3), imagesList[i].dtype)

        # Inverte Mascara
        mask = cv2.bitwise_not(maskList[i].astype(np.uint8))
        mask = mask.astype(np.float32) / 255
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)        

        imgRoi = applyMask(imagesList[i], mask.astype(np.uint8))
        #cv2.imwrite(str(i+10)+" - imgRoi.bmp", imgRoi * 255)

        # Gero uma mascara com os tons de verdes remanescentes nas imagens para equilibrar eles nos proximos passos
        # Aplico a mascara na região de interesse, dentro do corvo, corvos ou bordas.       
        imgG = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2HSV)
        imgG = imgG.astype(np.float32)
        
        h, s, v = imgG[:, :, 0], imgG[:, :, 1], imgG[:, :, 2]
        remainGreenMask = np.where(np.logical_and(h/255 >= 0.3, h/255 <= 0.5), 1, 0)
        #cv2.imwrite(str(i+10)+" - RemainGreen.bmp", remainGreenMask * 255)

        # Equilibro os tons mexendo no S. Da para mexer no V tambem, mas fico meio artificial.
        sNoG = np.zeros((s.shape[0], s.shape[1]), s.dtype)
        sNoG = np.where(remainGreenMask == 1, s * 0.2, s)

        # Recrio a imagem com o S balanceado e agora podemos fazer as operações com o background
        # Equilibro o verde nas coisas ao redor e dentro do objeto de interesse
        balanced = cv2.merge((h, sNoG, v))
        balanced = cv2.cvtColor(balanced, cv2.COLOR_HSV2BGR)
        #cv2.imwrite(str(i+10)+" - balanced.bmp", balanced * 255)

        ## Começo a trabalhar o resize
        backResized = cv2.resize(imgback, (imagesList[i].shape[1], imagesList[i].shape[0]), interpolation=cv2.INTER_CUBIC)
        backResized = np.clip(backResized, 0, 1)
        #cv2.imwrite(str(i+10)+" - resized.bmp", backResized * 255)

        imgFinal = np.where(mask == 1, balanced, backResized)
        #cv2.imwrite(str(i+10)+" - imgFinal.bmp", imgFinal * 255)

        ## Começo a tratar os contornos do chromakey
        #Borro a imagem final ja com os tons de verde tratados
        imgBlurBorder = gaussBlurImage(imgFinal, 7)     
        #cv2.imwrite(str(i+10)+" - imgBlur.bmp", imgBlurBorder * 255)  
             
        # Crio a mascara no Sobel
        maskSobel = applySobel(mask, 3)
        edgeBlur = np.where(maskSobel != 0, imgBlurBorder, 0)
        #cv2.imwrite(str(i+10)+" - edgeBlur.bmp", edgeBlur * 255)

        # Aonde for contorno pefo o borrado senao pego a imagem final, ja balanceada.
        imgOut = np.where(edgeBlur != 0, edgeBlur, imgFinal)
        cv2.imwrite(str(i + 1)+" - imgOut.bmp", imgOut * 255)

# ============================================================================================================================
# Metodo principal
# ============================================================================================================================


def main():

    listImages = []
    listImagesHsv = []

    for img in glob.glob("*.bmp"):
        imgG = cv2.imread(img)
        if imgG is None:
            print('Erro abrindo a imagem .\n')
        imgG = cv2.cvtColor(imgG, cv2.COLOR_BGR2HSV)
        imgG = imgG.reshape(imgG.shape[0], imgG.shape[1], 3)
        imgG = imgG.astype(np.float32) / 255
        listImagesHsv.append(imgG)

        imgOriginal = cv2.imread(img)
        imgOriginal = imgOriginal.astype(np.float32) / 255
        listImages.append(imgOriginal)

    listMasks = removeGreenBackground(listImagesHsv)
    applyChromaKey(listImages, listMasks)


if __name__ == '__main__':
    main()
