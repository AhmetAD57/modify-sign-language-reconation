from PyQt5 import QtWidgets, QtCore
import sys
from tasarim import Ui_MainWindow


class App(QtWidgets.QMainWindow):
    def __init__(self): #Nesne oluşturuluyor
        super(App, self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        
        #Kodlar buraya yazılıyor
        #1
        self.ui.pushButton.clicked.connect(self.yaziyaz)
        #2
        self.ui.comboBox.currentIndexChanged.connect(self.comboislem)
        self.ui.comboBox_2.currentIndexChanged.connect(self.comboislem)
        self.ui.comboBox_3.currentIndexChanged.connect(self.comboislem)
        #3
        self.timer = QtCore.QTimer()
        self.s=0
        self.timer.timeout.connect(self.lcd)
        self.ui.pushButton_2.clicked.connect(self.lcd_basla)
        self.ui.pushButton_3.clicked.connect(self.lcd_dur)
    
    
    def yaziyaz(self):
        self.ui.label_9.setText(self.ui.lineEdit.text())
    
    def comboislem(self):
        islem=self.ui.comboBox_2.currentText()
        c1=int(self.ui.comboBox.currentText())
        c2=int(self.ui.comboBox_3.currentText())
        
        if islem =="+":
            self.ui.label_7.setText(str(c1+c2))
        elif islem =="-":
            self.ui.label_7.setText(str(c1-c2))
        elif islem =="*":
            self.ui.label_7.setText(str(c1*c2))
        else:
            self.ui.label_7.setText(str(c1/c2))   

    def lcd(self):
        self.s+=1
        self.ui.lcdNumber.display(self.s)
    
    def lcd_basla(self):
        self.timer.start(100)
    def lcd_dur(self):
        self.timer.stop()
        self.s=0
        self.ui.lcdNumber.display(self.s)


#Pencereyi göstermek için
def pencere():
    pencere=QtWidgets.QApplication(sys.argv)
    win=App()
    win.show()
    sys.exit(pencere.exec_())

pencere()