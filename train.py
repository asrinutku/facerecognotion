import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

""" cascade oluşturuyoruz """
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

""" egitecegimiz fotografların dosya yolunu veriyoruz """
dosyayolu='yuzverileri'

def resimlerial(path):
    """ ilgili dosyaya ulasmak icin os'u kullanıyoruz"""
    resimlerinyolu = [os.path.join(path, f) for f in os.listdir(path)]
    resimler = []
    resimetiketleri = []

    for resimyolu in resimlerinyolu:
        print(resimyolu)
        """ resmi alıp PIL kutuphanesının convert methodu ile resmi gri ve net bir bicimde elde ediyoruz """
        convertedilmisresim = Image.open(resimyolu).convert('L')
        """ resmi numpy dizisine ceviriyoruz """
        resim = np.array(convertedilmisresim, 'uint8')

        """ resim isimlerini bize uygun formata donusturuyoruz"""
        resimadları = int(os.path.split(resimyolu)[1].split(".")[0].replace("face-", ""))
        print(resimadları, "EĞİTİLİYOR")

        """ ogretecegimiz yuzlerı resimler icinde tespit ediyoruz"""
        yuzler = cascade.detectMultiScale(resim)

        for (x, y, w, h) in yuzler:
            resimler.append((resim[y:y+h, x:x+w]))
            resimetiketleri.append(resimadları)
            cv2.imshow("EGİTİM", resim[y:y+h, x:x+w])
            cv2.waitKey(15)

    return resimler, resimetiketleri

resimler , resimetiketler = resimlerial(dosyayolu)
cv2.imshow("test", resimler[0])
cv2.waitKey(1)

""" eğitim """
recognizer.train(resimler, np.array(resimetiketler))

""" eğitim verilerini kaydediyoruz"""
recognizer.write('training/trainer.yml')

cv2.destroyAllWindows()