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
        self.satirsayisi=0
        self.sutunsayisi=0

        #Kodlar buraya yazılıyor
        self.ui.tableView.setModel(self.model)
        self.ui.tableView.horizontalHeader().setStretchLastSection(True)
        self.ui.pushButton.clicked.connect(self.csv_yukle)
        self.ui.pushButton_2.clicked.connect(self.onislem)
        self.ui.pushButton_3.clicked.connect(self.veri_duzelt)
    

    def csv_yukle(self):
        dataset, _ = QFileDialog.getOpenFileName(self, "Dosya Seç", "~", "Csv Dosyaları (*.csv)")
        self.tablo_doldur(str(dataset))
    
    def tablo_doldur(self, fileName):
        self.model.clear()
        
        with open(fileName, "r") as fileInput:
            for row in csv.reader(fileInput):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.satirsayisi=self.satirsayisi+1
                self.model.appendRow(items)
            
        self.sutunsayisi= len(items)
        #print(self.satirsayisi, self.sutunsayisi) 
    
    def onislem(self):
        if self.ui.checkBox.isChecked(): #Cisiyet checkbox
            for i in range(self.satirsayisi):
                if i==0:
                    continue
                model = self.ui.tableView.model().item(i,4)
                cinsiyet = model.text()
                
                if cinsiyet=="male":
                    model.setText("0")
                if cinsiyet=="female":
                    model.setText("1")
        
        if self.ui.checkBox_2.isChecked(): #Kabin checkbox
            for i in range(self.satirsayisi):
                if i==0:
                    continue
                model = self.ui.tableView.model().item(i,10)
                kabin = model.text()[:1]
                
                if kabin=="A":
                    model.setText("0")
                if kabin=="B":
                    model.setText("1")    
                if kabin=="C":
                    model.setText("2")
                if kabin=="D":
                    model.setText("3")
                if kabin=="E":
                    model.setText("4")   
                if kabin=="F":
                    model.setText("5")
                if kabin=="G":
                    model.setText("6")
                if kabin=="T":
                    model.setText("7")
                if kabin=="N":
                    model.setText("8")
        
        if self.ui.checkBox_3.isChecked(): #Kapı checkbox
            for i in range(self.satirsayisi):
                if i==0:
                    continue
                model = self.ui.tableView.model().item(i,11)
                kapi = model.text()                
                
                if kapi=="S":
                    model.setText("0")
                if kapi=="C":
                    model.setText("1")    
                if kapi=="Q":
                    model.setText("2")
                
    def veri_duzelt(self):
        yas_adeti=0
        yas_toplam=0
        if self.ui.checkBox_5.isChecked(): #Yaş checkbox
            for i in range(self.satirsayisi):
                if i==0:
                    continue
                model = self.ui.tableView.model().item(i,5)
                yas = model.text()
                
                if yas != "":
                    yas_adeti=yas_adeti+1
                    yas_toplam=yas_toplam+int(float(yas))
            
            ort_yas=yas_toplam//yas_adeti
            
            for i in range(self.satirsayisi):
                if i==0:
                    continue    
                model = self.ui.tableView.model().item(i,5)
                yas = model.text()        
                if yas == "":
                    model.setText(str(ort_yas))

        if self.ui.checkBox_6.isChecked(): #Kabin checkbox
            for i in range(self.satirsayisi):
                if i==0:
                    continue
                model = self.ui.tableView.model().item(i,10)
                kabind = model.text()
                
                if kabind == "":
                   model.setText("N")
            
#Pencereyi göstermek için
def pencere():
    pencere=QtWidgets.QApplication(sys.argv)
    win=App()
    win.show()
    sys.exit(pencere.exec_())

pencere()