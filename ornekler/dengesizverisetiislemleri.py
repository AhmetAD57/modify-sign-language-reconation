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
from imblearn.under_sampling import NearMiss
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier ,BaggingClassifier, StackingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
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

        self.dataset=pd.read_csv('dataset\pima-indians-diabetes.csv')
        self.dataset_konum="dataset\pima-indians-diabetes.csv"
        
        self.komsular = np.arange(1,6)
        self.train_dogrulugu =np.empty(len(self.komsular))
        self.test_dogrulugu = np.empty(len(self.komsular))
        
        self.y_tahmin=np.empty(0)
        self.y_skor=np.empty(0)

        self.y_test=[]
        
        self.t_check=0

        #Kodlar buraya yazılıyor
        self.ui.tableView.setModel(self.model)
        self.ui.tableView.horizontalHeader().setStretchLastSection(True)
        
        self.ui.pushButton.clicked.connect(self.egit)
        self.ui.pushButton_2.clicked.connect(self.tahmin)
        self.ui.pushButton_3.clicked.connect(self.karisiklik_matrisi)
        self.ui.pushButton_4.clicked.connect(self.egitim_test_basari)
        self.ui.pushButton_5.clicked.connect(self.roc_egrisi)
        self.ui.pushButton_7.clicked.connect(self.tablo_doldur)
        self.ui.checkBox.clicked.connect(self.t_durum)
        self.ui.comboBox.currentTextChanged.connect(self.a_durum)

    
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

    def a_durum(self):
        a_ogrenme=self.ui.comboBox.currentText()
        
        if a_ogrenme=="Bagging":
            self.ui.label_19.setText("SVM")
        
        if a_ogrenme=="Stacking":
            self.ui.label_19.setText("RandomForestClassifier")   
        
        if a_ogrenme=="Boosting":
            self.ui.label_19.setText("Ada Boost Classifier")   
        
        if a_ogrenme=="Voting":
            self.ui.label_19.setText("LogisticRegression, RandomForest")   


    def t_durum(self):
        if self.t_check==0:
            self.ui.groupBox_5.setEnabled(True)
            self.ui.pushButton_4.setEnabled(False)
            self.t_check=1
        else:
            self.ui.groupBox_5.setEnabled(False)
            self.ui.pushButton_4.setEnabled(True)
            self.t_check=0

    def egit(self):
        self.ui.groupBox_3.setEnabled(True)
        
        X = self.dataset.drop('diabetes',axis=1).values
        y = self.dataset['diabetes'].values
        
        #tüm veri seti dengeli
        if self.ui.radioButton_2.isChecked():
            nm=NearMiss()
            X, y=nm.fit_sample(X, y)
        
        #eğitim seti dengeli
        X_train, X_test, y_train, self.y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        
        if self.ui.radioButton_3.isChecked():
            nm=NearMiss()
            X_train, y_train=nm.fit_sample(X_train, y_train)
        
        #print(X.shape)
        if self.t_check==0:
            for i,k in enumerate(self.komsular):
                #n_neighbors: kullanılacak komşu sayısı, metric: komşuların yakınlığını belirlemede kullanılan yöntem, minkowski:mesafeye dayalı yöntem, p: 2 öklid mesafesini
                Model = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p = 2) 

                #Model eğitiliyor
                Model.fit(X_train, y_train)
                
                self.train_dogrulugu[i] = Model.score(X_train, y_train)
                self.test_dogrulugu[i] = Model.score(X_test, self.y_test) 

                self.y_tahmin = Model.predict(X_test)
                self.y_skor = Model.predict_proba(X_test)

        if self.t_check==1:
            
            to_algoritmasi= self.ui.comboBox.currentText()

            if to_algoritmasi=="Bagging":
                algoritma=BaggingClassifier(base_estimator=svm.SVC(), n_estimators=10, random_state=0)
                
            if to_algoritmasi=="Stacking":
                algoritma=StackingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=10, random_state=42)), ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))], final_estimator=LogisticRegression())
                
            if to_algoritmasi=="Boosting":
                algoritma=AdaBoostClassifier(n_estimators=100, random_state=0)
                
            if to_algoritmasi=="Voting":
                clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
                clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
                clf3 = GaussianNB()
                
                algoritma=VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
            
            algoritma.fit(X_train, y_train)
            
            self.y_tahmin = algoritma.predict(X_test)
            self.y_skor = algoritma.predict_proba(X_test)
         
        dogruluk=round(accuracy_score(self.y_test, self.y_tahmin)*100,2)
        #print("Doğruluk:",dogruluk)
        self.ui.label_6.setText(str(dogruluk)+"%")

    def tahmin(self):
        plt.title('Tahmini ve gerçek değerler')
        plt.plot(self.y_tahmin*10, label='Tahmini değerler')
        plt.plot(self.y_test*10, label='Gerçek değerler')
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