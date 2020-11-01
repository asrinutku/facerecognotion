import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()

""" eğittimiz verileri dosyadan okuyoruz """
recognizer.read('training/trainer.yml')

""" cascade oluşturma """
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

""" dosya yolu"""
path ='yuzverileri'

kamera = cv2.VideoCapture(0)

bulunaninsanismi = 0

yaziskalasi = 0.96
yazisekli = cv2.FONT_HERSHEY_TRIPLEX
taninanyuzsayisi = 0
while 1:
    _, image = kamera.read()
    """ kamera goruntusunu gri tona cevirip icindeki yuzlerı tespit ediyoruz """
    griton = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yuzler = cascade.detectMultiScale(griton, scaleFactor=1.2, minNeighbors=5)

    """ x y w h : yuzun sağ-sol , alt-üst köşelerinin konumunu tutacak"""
    for(x, y, w, h) in yuzler:

        """ eğittiğimiz kişileri kamera goruntusunde yakalayıp tanıyoruz"""
        bulunaninsannumarasi , dogrulukyuzdesi = recognizer.predict(griton[y:y+h, x:x+w])

        """ taninan insanların kafası etrafına dikdortgen ciziyoruz"""
        cv2.rectangle(image, (x-13, y-13), (x+w+19, y+h+19), (0, 0, 128), 3)

        if dogrulukyuzdesi <= 75:
            if (bulunaninsannumarasi == 1):
                bulunaninsanismi = 'asrin'
                taninanyuzsayisi +=1
                print("BULUNDU", bulunaninsanismi, taninanyuzsayisi)

            if (bulunaninsannumarasi == 2):
                bulunaninsanismi = 'salah'
                taninanyuzsayisi += 1
                print("BULUNDU", bulunaninsanismi,taninanyuzsayisi)

            if (bulunaninsannumarasi == 3):
                bulunaninsanismi = 'khaleesi'
                taninanyuzsayisi += 1
                print("BULUNDU", bulunaninsanismi, taninanyuzsayisi)
        else:
            bulunaninsannumarasi = "KISI TANINMADI"

        if(taninanyuzsayisi >= 0):
            print("KAMERADA ANLIK TANINAN KISI SAYISI : ", taninanyuzsayisi)


        """ bulunan kisinin ismini ekrana yazdırıyoruz"""
        cv2.putText(image, str(bulunaninsanismi), (x, y+h), yazisekli, yaziskalasi, (255, 0, 0))

        """ ekranda kac kisinin yuzunun tanındıgını buluyoruz"""
        cv2.putText(image, str(taninanyuzsayisi), (x+135, y + h), yazisekli, yaziskalasi, (0, 0, 255))

        """ ekran cıktısı alıyoruz """
        cv2.imshow("Goruntu",image)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            kamera.release()
            cv2.destroyAllWindows()
            break

    taninanyuzsayisi = 0
