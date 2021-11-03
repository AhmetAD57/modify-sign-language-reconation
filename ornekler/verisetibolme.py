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
        self.model1 = QtGui.QStandardItemModel(self)
        self.model2 = QtGui.QStandardItemModel(self)
        self.satirsayisi=0
        self.sutunsayisi=0
        self.dataset=""
        
        #Kodlar buraya yazılıyor
        self.ui.tableView.setModel(self.model)
        self.ui.tableView.horizontalHeader().setStretchLastSection(True)
        self.ui.tableView_2.setModel(self.model1)
        self.ui.tableView_2.horizontalHeader().setStretchLastSection(True)
        self.ui.tableView_3.setModel(self.model2)
        self.ui.tableView_3.horizontalHeader().setStretchLastSection(True)


        self.ui.pushButton_2.clicked.connect(self.csv_yukle)
        self.ui.pushButton.clicked.connect(self.islem_yap)
        self.ui.comboBox.currentIndexChanged.connect(self.degis)

    def degis(self):
        islem= self.ui.comboBox.currentText()

        if islem=="K-fold":
            self.ui.label_2.setEnabled(False)
            self.ui.spinBox.setEnabled(False)

            self.ui.label_5.setEnabled(True)
            self.ui.comboBox_2.setEnabled(True)
            self.ui.label_6.setEnabled(True)
        else:
            self.ui.label_2.setEnabled(True)
            self.ui.spinBox.setEnabled(True)

            self.ui.label_5.setEnabled(False)
            self.ui.comboBox_2.setEnabled(False)
            self.ui.label_6.setEnabled(False)

    def csv_yukle(self):
        self.dataset, _ = QFileDialog.getOpenFileName(self, "Dosya Seç", "~", "Csv Dosyaları (*.csv)")
        self.tablo_doldur(self.dataset)
        
    def tablo_doldur(self, fileName):
        self.model.clear()
        self.satirsayisi=0

        with open(fileName, "r") as fileInput:
            for row in csv.reader(fileInput):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.satirsayisi=self.satirsayisi+1
                self.model.appendRow(items)
            
            self.satirsayisi=self.satirsayisi-1
            self.sutunsayisi= len(items)
        
        print(self.satirsayisi, self.sutunsayisi)
        self.ui.label_7.setText("Toplam örnek sayısı= "+str(self.satirsayisi))

    def islem_yap(self):
        self.model1.clear()
        self.model2.clear()
        satir=0
        
        islem= self.ui.comboBox.currentText()
        test_boyutu= int(self.ui.spinBox.text())
        
        test_adeti=int(self.satirsayisi*(test_boyutu/100))
        
        egitim_adeti=self.satirsayisi-test_adeti
        
        if islem=="Hold-out":
            with open(self.dataset, "r") as fileInput1:
                for row in csv.reader(fileInput1):    
                    items = [
                        QtGui.QStandardItem(field)
                        for field in row
                    ]
                    satir=satir+1
                    if (satir <=test_adeti+1) & (satir!=1):
                        self.model2.appendRow(items)
                    
                    if(satir> test_adeti+1):
                        self.model1.appendRow(items)
                
                self.ui.label_3.setText("Eğitim seti örnek sayısı= "+str(egitim_adeti))
                self.ui.label_4.setText("Test seti örnek sayısı= "+str(test_adeti))
        
        if islem=="K-fold":
            ust_sinir=0
            alt_sinir=0
            test_orani= int(self.satirsayisi*(0.2))
            
            katman= int(self.ui.comboBox_2.currentText())
            
            if katman==1:
                ust_sinir=0
                alt_sinir=test_orani
                print(ust_sinir, alt_sinir)
            if katman==2:
                ust_sinir=test_orani
                alt_sinir=2*test_orani
                print(ust_sinir, alt_sinir)
            if katman==3:
                ust_sinir=2*test_orani
                alt_sinir=3*test_orani
                print(ust_sinir, alt_sinir)
            if katman==4:
                ust_sinir=3*test_orani
                alt_sinir=4*test_orani
                print(ust_sinir, alt_sinir)
            if katman==5:
                ust_sinir=4*test_orani
                alt_sinir=5*test_orani
                print(ust_sinir, alt_sinir)

            with open(self.dataset, "r") as fileInput2:
                for row in csv.reader(fileInput2):    
                    items = [
                        QtGui.QStandardItem(field)
                        for field in row
                    ]
                    
                    satir=satir+1

                    if satir ==1:
                        continue
                    
                    if (alt_sinir+1 >= satir > ust_sinir+1):
                        self.model2.appendRow(items)
                    
                    else:
                        self.model1.appendRow(items)
                
                self.ui.label_3.setText("Eğitim seti örnek sayısı: "+str(self.satirsayisi-test_orani))
                self.ui.label_4.setText("Test seti örnek sayısı: "+str(test_orani))


#Pencereyi göstermek için
def pencere():
    pencere=QtWidgets.QApplication(sys.argv)
    win=App()
    win.show()
    sys.exit(pencere.exec_())

pencere()