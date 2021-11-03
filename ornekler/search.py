import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, plot_confusion_matrix, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

        self.dataset=pd.read_csv('dataset\diabetes.csv')
        self.dataset_konum="dataset\diabetes.csv"
        
        self.clf=DecisionTreeClassifier()

        
        
        self.ui.tableView.setModel(self.model)
        self.ui.tableView.horizontalHeader().setStretchLastSection(True)

        self.ui.pushButton.clicked.connect(self.islem_yap)
        self.ui.pushButton_2.clicked.connect(self.tablo_doldur)

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
        self.ui.pushButton_2.setEnabled(False)
    
    def islem_yap(self):
        X = self.dataset.drop('Outcome',axis=1).values
        y = self.dataset['Outcome'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.95, random_state = 0) #Gridsearch zorlanıyor
        
        #print(X_train.shape)

        o_algoritma= self.ui.comboBox_2.currentText()
        
        if o_algoritma=="SVM":
            classifier=svm.SVC(gamma='auto')
            parameters=[{'C': [1,10,20], 'kernel': ['rbf','linear']}]
        
        if o_algoritma=="RandomForest":
            classifier=RandomForestClassifier()
            parameters=[{'n_estimators': [1,5,10]}]
        
        if o_algoritma=="LogisticRegression":
            classifier=LogisticRegression(solver='liblinear',multi_class='auto')
            parameters=[{'C': [1,5,10]}]
        
        optimizasyon_algoritmasi= self.ui.comboBox.currentText()
        
        if optimizasyon_algoritmasi=="Grid Search":
            algoritma = GridSearchCV(estimator =classifier, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1)
        
        if optimizasyon_algoritmasi=="Random Search":
            algoritma = RandomizedSearchCV(estimator =classifier, param_distributions = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1)
        
        algoritma.fit(X_train, y_train)
        
        accuracy_t = algoritma.predict(X_test)
        
        accuracy=accuracy_score(y_test,accuracy_t)
        
        self.ui.label_12.setText(str(round(accuracy*100,2))+"%")
        self.ui.label_7.setText(str(round(algoritma.best_score_*100,2))+"%")
        self.ui.label_8.setText(str(round(precision_score(y_test,accuracy_t)*100,2))+"%")
        self.ui.label_9.setText(str(round(recall_score(y_test,accuracy_t)*100,2))+"%")
        self.ui.label_10.setText(str(round(f1_score(y_test,accuracy_t)*100,2))+"%")
        self.ui.textEdit.setText(str(algoritma.best_params_))
        
                   
#Pencereyi göstermek için
def pencere():
    pencere=QtWidgets.QApplication(sys.argv)
    win=App()
    win.show()
    sys.exit(pencere.exec_())

pencere()