import tensorflow as tf
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from PyQt5 import QtWidgets, QtCore
import sys
from tasarim import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap

class App(QtWidgets.QMainWindow):
    def __init__(self): #Nesne oluşturuluyor
        super(App, self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.history=""
        self.class_names=[]
        self.model=tf.keras.Sequential()
        self.test_images=[]
        self.m_c=0
        self.test_acc=0
        #Buttonlar
        self.ui.label_9.setVisible(False)
        self.ui.pushButton_4.clicked.connect(self.egit)
        self.ui.pushButton.clicked.connect(self.basari)
        self.ui.pushButton_6.clicked.connect(self.resim_test)
        self.ui.pushButton_5.clicked.connect(self.kayip)
        self.ui.pushButton_2.clicked.connect(self.matrix)
        #self.ui.pushButton_3.clicked.connect(self.roc)
        #Eğitim kısmı
    def egit(self):
        self.ui.label_7.setText("Eğitim gerçekleştiriliyor...")
        fashion_mnist = tf.keras.datasets.fashion_mnist

        (train_images, train_labels), (self.test_images, test_labels) = fashion_mnist.load_data()

        self.class_names = ['T-shirt/üst', 'Pantolon', 'Süveter', 'Elbise', 'Kot','Sandalet', 'Gömlek', 'Spor ayakkabı', 'Çanta', 'Bilek boyu bot']

        train_images = train_images / 255.0

        test_images = self.test_images / 255.0

        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])


        self.model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        self.history= self.model.fit(train_images, train_labels, epochs=5)

        test_loss, self.test_acc = self.model.evaluate(test_images,  test_labels, verbose=2)

        self.ui.label_7.setText("Eğitim Tamamlandı.")
        self.ui.label_8.setText(str(round(self.test_acc*100,2))+"%")

        
        self.ui.groupBox_6.setEnabled(True)
        #Grafikler
    def basari(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['accuracy'])
        plt.title('model başarısı')
        plt.ylabel('Başarı')
        plt.xlabel('epoch')
        plt.show()
        
    def kayip(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['loss'])
        plt.title('model kaybı')
        plt.ylabel('Kayıp')
        plt.xlabel('epoch')
        plt.show()
    
    def matrix(self):
        # y_pred=self.model.predict_classes(self.test_images)
        # mat=confusion_matrix(self.class_names, y_pred)
        # print(mat)
        if self.m_c==0:
            self.ui.label_9.setVisible(True)
            self.m_c=1
        else:
            self.ui.label_9.setVisible(False)
            self.m_c=0
    
    # def roc(self):
    #     print(metrics.roc_curve(self.class_names, self.test_acc))

    #Test
    def resim_test(self):
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        resim, _ = QFileDialog.getOpenFileName(self, "Dosya Seç", "~", "Resim Dosyaları (*.jpg)")
        
        img = image.load_img(resim, grayscale=True, target_size=(28, 28))
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x /= 255
        
        predictions = probability_model.predict(x)
        a=np.argmax(predictions[0])
        #print(self.class_names[a])
        
        pixmap = QPixmap(resim)
        self.ui.label_6.setPixmap(pixmap)
        
        self.ui.label_2.setText("Sonuç: Bu nesne bir "+ str(self.class_names[a])+" dır.")

#Pencereyi göstermek için
def pencere():
    pencere=QtWidgets.QApplication(sys.argv)
    win=App()
    win.show()
    sys.exit(pencere.exec_())

pencere()