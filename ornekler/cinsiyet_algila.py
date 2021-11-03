#Genel
import sys, os, math, random, glob, shutil, subprocess, webbrowser
from distutils.dir_util import copy_tree

#TensorFlow
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.applications import vgg16

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16, VGG19

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

#PyQt5
from PyQt5 import QtWidgets, QtCore, QtGui
from tasarim import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog, QMessageBox

#Yardımcı
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import itertools


class App(QtWidgets.QMainWindow):
    def __init__(self): #Nesne oluşturuluyor
        super(App, self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)


        self.dataset=""
        self.aktif_dataset=""
        self.piccount=0
        self.da_piccount=0
        self.subdirs=[]
        self.dircounts=[]
        self.test_o=0
        self.egitim_o=0
        self.select_c=0
        
        for path, subdirs, files in os.walk("arttirilmis_veriseti"):
            for name in files:
                self.da_piccount=self.da_piccount+1
        

        self.secili_model="AlexNet"
        self.model_adi=""
        self.epochs=60
        self.batch_size=64
        self.callbacks=[]
        self.class_lar=["men", "women"]
        self.selected_optimizer="Adam"
        

        self.test_model_path=""
        check=False
        for path, subdirs, files in os.walk("Models"):
            for name in files:
                if(not check):
                    self.ui.pushButton_7.setEnabled(True)
                    self.test_model_path=name
                    check=True
                self.ui.comboBox_3.addItem(name)
        self.sub_process_check=False
        self.sub_process=""
        self.ui.label_27.setPixmap(QtGui.QPixmap("Plts/"+self.test_model_path.replace(".h5","")+".png"))
        

        self.ui.pushButton.clicked.connect(self.dataset_folder)
        self.ui.pushButton_2.clicked.connect(self.show_folder_info)
        self.ui.pushButton_3.clicked.connect(self.dataset_bol)
        self.ui.pushButton_4.clicked.connect(self.veri_arttirma)
        self.ui.comboBox.currentTextChanged.connect(self.dataset_sec)


        self.ui.radioButton_4.clicked.connect(self.rb_durum)    
        self.ui.radioButton_3.clicked.connect(self.rb_durum)
        self.ui.comboBox_2.currentTextChanged.connect(self.model_sec)  
        self.ui.pushButton_9.clicked.connect(self.model_ad)
        self.ui.pushButton_5.clicked.connect(self.egit)
        self.ui.checkBox.stateChanged.connect(self.model_ozellikler)
        self.ui.checkBox_2.stateChanged.connect(self.model_ozellikler)
        self.ui.spinBox_3.valueChanged.connect(self.model_ozellikler)
        self.ui.comboBox_4.currentTextChanged.connect(self.opimizer_sec)
        self.ui.spinBox_4.valueChanged.connect(self.epoch_batchsize_sec)
        self.ui.spinBox_5.valueChanged.connect(self.epoch_batchsize_sec)

        
        self.ui.pushButton_6.clicked.connect(self.test)
        self.ui.pushButton_7.clicked.connect(self.tensorboard_log)
        self.ui.comboBox_3.currentTextChanged.connect(self.model_select)
        self.ui.pushButton_8.clicked.connect(self.model_bilgi_getir)

    #Veri seti
    def dataset_folder(self):
        dialog = QFileDialog()
        self.dataset = dialog.getExistingDirectory(self, 'Veri Seti Klasörü')
        
        if(self.dataset!=""):
            self.piccount=0
            self.subdirs.clear()
            self.dircounts.clear()

            ctr=False
            ct=0
            for path, subdirs, files in os.walk(self.dataset):
                if(not(ctr)):
                    self.subdirs=subdirs
                    ctr=True
                #print(subdirs)
                for name in files:
                    ct+=1
                    self.piccount+=1
                    #print(os.path.join(path, name))
                self.dircounts.append(ct)
                ct=0
            
            self.dircounts.pop(0)
            
            if(self.piccount> 0):
                self.ui.pushButton_2.setEnabled(True)
                self.ui.pushButton_3.setEnabled(True)
                self.ui.pushButton_4.setEnabled(True)
                self.ui.comboBox.setEnabled(True)


            if(self.da_piccount>0):
                self.ui.comboBox.clear()
                self.ui.comboBox.addItem("Normal veri seti")
                self.ui.comboBox.addItem("Arttırılmış veri seti")
            
            self.aktif_dataset=self.dataset
            
            #Bilgi çubuğu
            self.ui.label_26.setText("Veri seti: Normal")
            self.ui.label_39.setText("Resim sayısı: "+str(self.piccount))
            self.ui.label_40.setText("Sınıf sayısı: "+str(len(self.subdirs)))

            self.ui.label_5.setText("Veri seti: Normal")
            self.ui.label_28.setText("Resim sayısı: "+str(self.piccount))
            self.ui.label_29.setText("Sınıf sayısı: "+str(len(self.subdirs)))
            

            print("-----------------")
            print("Dataset yol:", self.dataset)
            print("Resim sayisi: ", self.piccount)
            
            
        else:
            print("Veri seti yok")
            

    def show_folder_info(self):
        plt.bar(self.subdirs, self.dircounts, color ='blue', width = 0.4) 
        
        plt.title("Veri Seti İçeriği") 
        plt.xlabel("Sınıflar") 
        plt.ylabel("Örnek sayısı")
        
        plt.show()

    def dataset_bol(self):
        self.test_o=int(self.ui.spinBox.text())
        self.egitim_o= 100 - int(self.ui.spinBox.text())

        self.ui.label_6.setText(str(math.floor(self.piccount*(int(self.ui.spinBox.text())/100))))
        self.ui.label_7.setText(str(self.piccount-(int(self.ui.label_6.text()))))
        
        #Bilgi çubuğu
        self.ui.label_41.setText("Test oranı: "+str(self.test_o)+"%, Eğitim oranı: "+str(self.egitim_o)+"%")

        self.ui.label_35.setText("Test oranı: "+str(self.test_o)+"%, Eğitim oranı: "+str(self.egitim_o)+"%")

    def veri_arttirma(self):
        files_m = glob.glob('arttirilmis_veriseti/men/*')
        for m in files_m:
            os.remove(m)
        files_w = glob.glob('arttirilmis_veriseti/women/*')
        for w in files_w:
            os.remove(w)

        
        arttirma_orani=int(self.ui.spinBox_2.text())
        arttirilacak_resim_sayisis=int(self.piccount*(arttirma_orani/100))
        print("Arttırılacak resim sayısı:", arttirilacak_resim_sayisis)

        datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest')
        
        for z in range(arttirilacak_resim_sayisis): 
            class_n=""
            if(self.select_c==0):
                class_n="men"
                self.select_c=1
            elif(self.select_c==1):
                class_n="women"
                self.select_c=0
            
            s_image=random.choice(os.listdir(self.dataset+"/"+class_n))
            
            img = load_img(self.dataset+"/"+class_n+"/"+s_image)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            
            
            for batch in datagen.flow(x, batch_size=1, save_to_dir='arttirilmis_veriseti/'+class_n, save_format='png'):
                break
                
            print("Resim Yol:", 'arttirilmis_veriseti/'+class_n+"/"+s_image)    

        print("Veri seti kopyalanıyor")
        copy_tree(self.dataset+"/men", "arttirilmis_veriseti\men")
        copy_tree(self.dataset+"/women", "arttirilmis_veriseti\women")
        print("Veri seti kopyalandı")
        
        self.ui.comboBox.clear()
        self.ui.comboBox.addItem("Normal veri seti")
        self.ui.comboBox.addItem("Arttırılmış veri seti")

    def dataset_sec(self):
        sec=self.ui.comboBox.currentText()

        if(sec=="Normal veri seti"):
            self.aktif_dataset=self.dataset
            
            #Bilgi çubuğu
            self.ui.label_26.setText("Veri seti: Normal")
            self.ui.label_39.setText("Resim sayısı:"+ str(self.piccount))

            self.ui.label_5.setText("Veri seti: Normal")
            self.ui.label_28.setText("Resim sayısı:"+ str(self.piccount))
            
        if(sec=="Arttırılmış veri seti"):
            self.aktif_dataset="arttirilmis_veriseti"
            #Bilgi çubuğu
            self.ui.label_26.setText("Veri seti: Arttırılmış ("+ str(int((self.da_piccount-self.piccount)*100/self.piccount))+"%)")
            self.ui.label_39.setText("Resim sayısı:"+ str(self.da_piccount))

            self.ui.label_5.setText("Veri seti: Arttırılmış ("+ str(int((self.da_piccount-self.piccount)*100/self.piccount))+"%)")
            self.ui.label_28.setText("Resim sayısı:"+ str(self.da_piccount))

        print("Aktif veri seti: ", self.aktif_dataset)
            

    
    
    #Eğitim

    def rb_durum(self):
        if self.ui.radioButton_3.isChecked():
            self.ui.groupBox_9.setEnabled(True)
            self.ui.groupBox_10.setEnabled(False)
            
            self.ui.label_19.setText("Epochs: "+ str(self.ui.spinBox_4.text()))
            self.ui.label_21.setText("Batch size: "+ str(self.ui.spinBox_5.text()))
        
        if self.ui.radioButton_4.isChecked():
            self.ui.groupBox_10.setEnabled(True)
            self.ui.groupBox_9.setEnabled(False)
            
            self.ui.label_19.setText("Epochs: "+ str(self.epochs))
            self.ui.label_21.setText("Batch size: "+ str(self.batch_size))


    def model_sec(self):
        self.secili_model=self.ui.comboBox_2.currentText()
        self.ui.label_16.setText("Model: "+ self.secili_model)

    def opimizer_sec(self):
        self.selected_optimizer=self.ui.comboBox_4.currentText()
        self.ui.label_32.setText("Optimizer: "+ self.selected_optimizer)
    
    def model_ad(self):
        if(self.ui.lineEdit.text()!="" and self.aktif_dataset!="" and self.test_o!=0):
            self.ui.label_17.setText("Tam ad: "+str(self.ui.lineEdit.text())+"-"+str(self.secili_model)+"-"+str(self.selected_optimizer)+".h5")
            self.model_adi=str(self.ui.lineEdit.text())+"-"+str(self.secili_model)+"-"+str(self.selected_optimizer)
            self.ui.pushButton_5.setEnabled(True)
            self.ui.label_47.setStyleSheet("color: green;")
            self.ui.label_47.setText("Hazır")
        else:
            self.ui.pushButton_5.setEnabled(False)
            self.ui.label_47.setStyleSheet("color: red;")
            self.ui.label_47.setText("Eğitim için bazı alanlar eksik")
    
    # Ek özellikler
    def model_ozellikler(self):
        self.callbacks.clear()
        self.ui.label_33.setText("Ek özellikler: TensorBoard")
        
        if(self.ui.checkBox.isChecked()==True):
            self.callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=int(self.ui.spinBox_3.text())))
            self.ui.label_33.setText(self.ui.label_33.text()+", Early stopping")
        
        if(self.ui.checkBox_2.isChecked()==True):
            self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath="Models/"+ self.model_adi+".h5", verbose=1, monitor='val_accuracy', mode='max',save_best_only=True))
            self.ui.label_33.setText(self.ui.label_33.text()+", Model check point")

    
    def epoch_batchsize_sec(self):
        self.ui.label_19.setText("Epochs: "+ str(self.ui.spinBox_4.text()))
        self.ui.label_21.setText("Batch size: "+ str(self.ui.spinBox_5.text()))

    def egit(self):
        sinif_sayisi=2
        
        if self.ui.radioButton_3.isChecked():
                self.epochs=int(self.ui.spinBox_4.text())
                self.batch_size=int(self.ui.spinBox_5.text())
        else:
            self.epochs=60
            self.batch_size=64
        
        
        print("---Eğitim---")
        print("Model: ", self.secili_model)
        print("Optimizer: ", self.selected_optimizer)
        print("Epochs: ", self.epochs)
        print("Batch size: ", self.batch_size)
        print("Test oranı: ", self.test_o)
        print("------------")


        #TransferLearning
        #input shapes for models
        IMAGE_SHAPE = (224, 224, 3)
        CLASS_NAMES = np.array(self.class_lar)
        
        if(self.secili_model=="AlexNet"):
            IMAGE_SHAPE = (227, 227, 3) 
        if(self.secili_model=="VGG16"):
            model = VGG16(input_shape=IMAGE_SHAPE)
        if(self.secili_model=="VGG19"):
            model = VGG19(input_shape=IMAGE_SHAPE)
        
        # 20% validation set 80% training set
        image_generator = ImageDataGenerator(rescale=1/255, validation_split=self.test_o/100)
        
        train_data_gen = image_generator.flow_from_directory(directory=self.aktif_dataset, batch_size=self.batch_size, classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]), shuffle=True, subset="training")
        test_data_gen = image_generator.flow_from_directory(directory=self.aktif_dataset, batch_size=self.batch_size, classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]), shuffle=True, subset="validation")
        
        if(self.secili_model!="AlexNet"):  
            # remove the last fully connected layer
            model.layers.pop()
            # freeze all the weights of the model except the last 4 layers
            for layer in model.layers[:-4]:
                layer.trainable = False
                
            output = Dense(sinif_sayisi, activation="softmax")
                
            # connect that dense layer to the model
            output = output(model.layers[-1].output)
            model = Model(inputs=model.inputs, outputs=output)
            # print the summary of the model architecture
            # model.summary()
            # training the model using adam optimizer


        if(self.secili_model=="AlexNet"):
            model = tensorflow.keras.models.Sequential([
            tensorflow.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            tensorflow.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            tensorflow.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(4096, activation='relu'),
            tensorflow.keras.layers.Dropout(0.5),
            tensorflow.keras.layers.Dense(4096, activation='relu'),
            tensorflow.keras.layers.Dropout(0.5),
            tensorflow.keras.layers.Dense(sinif_sayisi, activation='softmax')
            ])

        model.compile(loss="categorical_crossentropy", optimizer=self.selected_optimizer, metrics=["accuracy"])
            
        training_steps_per_epoch = np.ceil(train_data_gen.samples / self.batch_size)
        validation_steps_per_epoch = np.ceil(test_data_gen.samples / self.batch_size)

        #Tensorboard
        self.callbacks.append(TensorBoard(log_dir="tensorboard_logs/"+self.model_adi+"-log", histogram_freq=1, write_graph=True, update_freq='epoch', profile_batch=2, embeddings_freq=1))

        model.fit(train_data_gen, steps_per_epoch=training_steps_per_epoch, validation_data=test_data_gen, validation_steps=validation_steps_per_epoch, epochs=self.epochs, callbacks=self.callbacks)
        
        Y_pred = model.predict(test_data_gen, test_data_gen.samples // test_data_gen.batch_size+1)
        
        y_pred = np.argmax(Y_pred, axis=1)
        
        print('Confusion Matrix')
        #print(confusion_matrix(self.test_data_gen.classes, y_pred))
        self.plot_confusion_matrix(confusion_matrix(test_data_gen.classes, y_pred), ["men","women"])
        
        model.save("Models/"+self.model_adi+".h5")
        
        self.ui.label_47.setText("Model eğitildi ve kayıt edildi")

        self.ui.comboBox_3.clear()
        cb_check=0
        for path, subdirs, files in os.walk("Models"):
            #print("Dosyalar: ", files)
            for name in files:
                #print(os.path.join(path, name))
                if(cb_check==0):
                    self.test_model_path=name
                    self.ui.pushButton_7.setEnabled(True)
                    cb_check=cb_check+1
                self.ui.comboBox_3.addItem(name)

    
    #Test
    def model_select(self):
        self.test_model_path=self.ui.comboBox_3.currentText()
        self.ui.label_27.setPixmap(QtGui.QPixmap("Plts/"+self.test_model_path.replace(".h5","")+".png"))
      
    def tensorboard_log(self):
        if(self.sub_process_check==False):
            webbrowser.open('http://localhost:6006/', new=1, autoraise=True)
            self.sub_process=subprocess.Popen('tensorboard --logdir=TensorBoard_logs\\'+self.test_model_path.replace(".h5","")+"-log", creationflags=subprocess.CREATE_NEW_CONSOLE)
            self.sub_process_check=True
        
        if(self.sub_process_check==True):
            self.sub_process.terminate()
            self.sub_process=subprocess.Popen('tensorboard --logdir=TensorBoard_logs\\'+self.test_model_path.replace(".h5","")+"-log", creationflags=subprocess.CREATE_NEW_CONSOLE)

    def test(self):
        model = tf.keras.models.load_model("Models\\"+self.test_model_path)
        
        chosed_class= random.choice(["men", "women"])
        #chosed_img=random.choice(os.listdir(self.dataset+"/"+chosed_class))
        chosed_img=random.choice(os.listdir("C:/Users/DORUK57_2/Desktop/Test/"+chosed_class))

        #random_img_patch=self.dataset+"/"+chosed_class+"/"+chosed_img
        random_img_patch="C:/Users/DORUK57_2/Desktop/Test/"+chosed_class+"/"+chosed_img
        
        self.ui.label_18.setPixmap(QtGui.QPixmap(random_img_patch))
        
        s = self.test_model_path
        find = ['AlexNet', 'VGG16', 'VGG19']
        results = [item for item in find if item in s]
        print(results)
        if(results[0]=="AlexNet"):
            image = load_img(random_img_patch, target_size=(227, 227))
        else:
            image = load_img(random_img_patch, target_size=(224, 224))
        
        image = img_to_array(image)
        
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
       
        pred = model.predict(image)
        
        self.get_pred(pred)

    def get_pred(self, pred):
        res_arr=pred[0]

        if(res_arr[0]>res_arr[1]):
            self.ui.label_10.setText("Erkek "+str("%.2f" %(res_arr[0]*100))+"%")
        else:
            self.ui.label_10.setText("Kadın "+str("%.2f" %(res_arr[1]*100))+"%")
    
    def model_bilgi_getir(self):
        model = tf.keras.models.load_model("Models\\"+self.test_model_path)
        t_g = ImageDataGenerator(rescale=1/255, validation_split=20/100)
        
        s = self.test_model_path
        find = ['AlexNet', 'VGG16', 'VGG19']
        results = [item for item in find if item in s]
        print(results)
        if(results[0]=="AlexNet"):
            IMAGE_SHAPE = (227, 227, 3)
        else:
            IMAGE_SHAPE = (224, 224, 3)
        
        CLASS_NAMES = np.array(["men","women"])
        test_gen = t_g.flow_from_directory(directory=self.aktif_dataset, batch_size=32, classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]), shuffle=True, subset="validation")
        
        Y_pred = model.predict(test_gen, test_gen.samples // test_gen.batch_size+1)
        y_pred = np.argmax(Y_pred, axis=1)
        print('Classification Report')
        print("--------------------------------------")
        print(classification_report(test_gen.classes, y_pred, target_names=self.class_lar))
        self.ui.plainTextEdit.setPlainText(classification_report(test_gen.classes, y_pred, target_names=self.class_lar))
    
    
    def plot_confusion_matrix(self, cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = cm * 100
            print("\nNormalized confusion matrix")
        else:
            print('\nConfusion matrix, without normalization')
        
        print(cm)
        print ()

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.0f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #plt.show()
        plt.savefig("Plts/"+self.model_adi+".png")

       














#Pencereyi göstermek için
def pencere():
    pencere=QtWidgets.QApplication(sys.argv)
    win=App()
    win.show()
    sys.exit(pencere.exec_())

pencere()        