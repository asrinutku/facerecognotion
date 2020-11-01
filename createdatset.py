import cv2
import time

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

""" internetten aldıgımız egitilmis seti programımıza ekliyoruz"""
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

""" yuz tanımada ekrana cizecegimiz dikdortgenin sınırları icin bi degisken belirliyoruz"""
cizgisinirlari = 50

i = 0
kisi_id = input("Kişinin numarası :")

while 1:
    _, img = cam.read()

    """ kameradan alınan goruntuyu yuz tanıma icin gri tona ceviriyoruz"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """ gritona cevirdigimiz goruntu icinde yuzlerı bulmak icin detectMultiscale kullanıyoruz"""
    """ fonksiyonda verdigimiz parametreler : goruntu skalalama , yuz aranırken komsu piksel tutma sayısı , min-max nesne boyutu , cascade kullanımı parametresi"""
    yuzler = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100), maxSize=(1000, 1000), flags=cv2.CASCADE_SCALE_IMAGE)

    for(x, y, w, h) in yuzler:
        i = i+1
        time.sleep(0.1)
        cv2.imwrite("yuzverileri/face-"+kisi_id+"."+str(i)+".jpg", gray[y-cizgisinirlari:y+h+cizgisinirlari, x-cizgisinirlari:x+w+cizgisinirlari])
        cv2.rectangle(img, (x-cizgisinirlari, y-cizgisinirlari), (x+w+cizgisinirlari, y+h+cizgisinirlari), (255, 0, 0), 2)
        cv2.waitKey(125)


    """ bir kisinin yuzunu kaydetmek ve kamerada tanımak icin 20 fotograf alıyoruz"""
    if i > 19:
        print("Yeterince fotograf alındı")
        print("Kaydedilen kişinin numarası :", kisi_id)
        cam.release()
        cv2.destroyAllWindows()
        break
