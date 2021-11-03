import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_confusion_matrix

from PyQt5 import QtWidgets, QtCore, QtGui
import sys
from tasarim import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog
import csv


class App(QtWidgets.QMainWindow):
    def __init__(self): #Nesne oluşturuluyor
        super(App, self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        
        #genel değişkenler class a ait
        self.model = QtGui.QStandardItemModel(self)

        self.sutunsayisi=0
        self.oznitelik_dizisi=[]
        
        self.dataset=pd.read_csv('dataset\diabetes.csv')
        #Veri setinin 5 satırı yazılır
        #print(self.dataset.head())
        self.dataset_konum="dataset\diabetes.csv"
        
        self.komsular = np.arange(1,6)
        self.train_dogrulugu =np.empty(len(self.komsular))
        self.test_dogrulugu = np.empty(len(self.komsular))
        
        self.y_tahmin=np.empty(0)
        self.y_skor=np.empty(0)

        self.ui.spinBox.setMinimum(1)

        self.y_test=[]
        

        #Kodlar buraya yazılıyor
        self.ui.tableView.setModel(self.model)
        self.ui.tableView.horizontalHeader().setStretchLastSection(True)
        
        self.ui.tableView.clicked.connect(self.oznitelik_sec)
        self.ui.pushButton.clicked.connect(self.egit)
        self.ui.pushButton_2.clicked.connect(self.tahmin)
        self.ui.pushButton_3.clicked.connect(self.karisiklik_matrisi)
        self.ui.pushButton_4.clicked.connect(self.egitim_test_basari)
        self.ui.pushButton_5.clicked.connect(self.roc_egrisi)
        self.ui.pushButton_6.clicked.connect(self.oznitelik_temizle)
        self.ui.radioButton.clicked.connect(self.normal_durum)
        self.ui.radioButton_2.clicked.connect(self.oznitelik_durum)
        self.ui.pushButton_7.clicked.connect(self.tablo_doldur)
    
    def tablo_doldur(self):
        with open(self.dataset_konum, "r") as fileInput:
            for row in csv.reader(fileInput):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                
                self.model.appendRow(items)
        self.sutunsayisi= len(items)
        self.ui.pushButton.setEnabled(True)
        self.ui.pushButton_7.setEnabled(False)
    
    def normal_durum(self):
        self.ui.pushButton.setEnabled(True)
        self.ui.spinBox.setMaximum(5)
    
    def oznitelik_durum(self):
        if len(self.oznitelik_dizisi)==0:
           self.ui.pushButton.setEnabled(False) 
        else:
            self.ui.spinBox.setMaximum(len(self.oznitelik_dizisi))
    
    def oznitelik_sec(self):
        index=(self.ui.tableView.selectionModel().currentIndex())
        value=index.sibling(index.row(),index.column()).data()
        
        for i in range(self.sutunsayisi):
            if i ==8:
                continue
            if value == self.ui.tableView.model().item(0,i).text():
                if value not in self.oznitelik_dizisi:
                    self.oznitelik_dizisi.append(value)
        
        #print(self.oznitelik_dizisi)
        self.ui.label_8.setText(str(self.oznitelik_dizisi).replace('[','').replace(']',''))
        if self.ui.radioButton_2.isChecked():
            self.ui.spinBox.setMaximum(len(self.oznitelik_dizisi))
        self.ui.pushButton.setEnabled(True)
        
    def oznitelik_temizle(self):
        self.oznitelik_dizisi=[]
        self.ui.label_8.setText(str(self.oznitelik_dizisi).replace('[','').replace(']',''))
        if self.ui.radioButton_2.isChecked():
            self.ui.pushButton.setEnabled(False)
            
    def egit(self):
        self.ui.groupBox_3.setEnabled(True)
        self.ui.textEdit_2.setText("")
        
        if self.ui.radioButton.isChecked():
            if self.ui.checkBox.isChecked():
                pca=PCA(n_components=int(self.ui.spinBox.text()))
                pca.fit(self.dataset.drop('Outcome',axis=1))
                X=pca.transform(self.dataset.drop('Outcome',axis=1))
                self.ui.textEdit_2.setText(str(pca.explained_variance_))
            else:
                X = self.dataset.drop('Outcome',axis=1).values
                
            y = self.dataset['Outcome'].values
            
        if self.ui.radioButton_2.isChecked():
            if self.ui.checkBox.isChecked():
                pca=PCA(n_components=int(self.ui.spinBox.text()))
                pca.fit(self.dataset[self.oznitelik_dizisi])
                X=pca.transform(self.dataset[self.oznitelik_dizisi])
                self.ui.textEdit_2.setText(str(pca.explained_variance_))
            else:
                X = self.dataset[self.oznitelik_dizisi].values
            
            y = self.dataset['Outcome'].values
            
        X_train, X_test, y_train, self.y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        
        print(X_train.shape)

        for i,k in enumerate(self.komsular):
            #n_neighbors: kullanılacak komşu sayısı, metric: komşuların yakınlığını belirlemede kullanılan yöntem, minkowski:mesafeye dayalı yöntem, p: 2 öklid mesafesini
            Model = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p = 2) 

            #Model eğitiliyor
            Model.fit(X_train, y_train)
            
            self.train_dogrulugu[i] = Model.score(X_train, y_train)
            self.test_dogrulugu[i] = Model.score(X_test, self.y_test) 

            self.y_tahmin = Model.predict(X_test)
            self.y_skor = Model.predict_proba(X_test)

        dogruluk=round(accuracy_score(self.y_test, self.y_tahmin)*100,2)
        #print("Doğruluk:",dogruluk)
        self.ui.label_6.setText(str(dogruluk)+"%")

    def tahmin(self):
        plt.title('Tahmini ve gerçek değerler')
        plt.plot(self.y_tahmin, label='Tahmini değerler')
        plt.plot(self.y_test, label='Gerçek değerler')
        plt.legend()
        plt.xlabel('Test veri sayısı')
        plt.ylabel('Örnek sayısı')
        plt.show()

    def karisiklik_matrisi(self):
        cm = confusion_matrix(self.y_test, self.y_tahmin)
        print(cm)
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        plt.show()
    
    def egitim_test_basari(self):
        plt.title('Komşu sayısına göre k-nn doğruluk tablosu')
        plt.plot(self.komsular, self.test_dogrulugu, label='Test Doğruluğu')
        plt.plot(self.komsular, self.train_dogrulugu, label='Eğitim Doğruluğu')
        plt.legend()
        plt.xlabel('Komuş sayısı')
        plt.ylabel('Doğruluk')
        plt.show()

    def roc_egrisi(self):
        fpr, tpr, threshold = roc_curve(self.y_test, self.y_skor[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.title('Roc Eğrisi')
        plt.plot(fpr, tpr, 'b', label = 'Başarı = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Oranı')
        plt.xlabel('False Positive Oranı')
        plt.show()


#Pencereyi göstermek için
def pencere():
    pencere=QtWidgets.QApplication(sys.argv)
    win=App()
    win.show()
    sys.exit(pencere.exec_())

pencere()