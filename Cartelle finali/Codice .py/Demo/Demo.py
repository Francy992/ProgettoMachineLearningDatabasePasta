import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
import os, pathlib, sys

from ImportFile import *

dictionary = {  'Bio': 4,
                'Cereali': 5,
                'Farina_semola_pizza': 8,
                'Legumi': 6,
                'Pasta_5_Cereali': 3,
                'Pasta_all\'uovo_ripiena': 6,
                'Pasta_Base': 18,
                'Pasta_box': 47,
                'Pasta_emiliane_chef': 2,
                'Pasta_integrale': 8,
                'Pasta_integrale_voiello': 2,
                'Pasta_piccolini': 6,
                'Pasta_special_pack': 4,
                'Pasta_specialita': 15,
                'PASTA_UOVO_5_CEREALI': 2,
                'senza_glutine': 9,
                'Sughi': 7,
             }

class TreeNet:
    def __init__(self, dictionary, data_augmentation=""):
        self.package_net = models.alexnet(pretrained=True)
        self.package_net.classifier[0] = nn.Dropout(p=0.0)
        self.package_net.classifier[3] = nn.Dropout(p=0.0)
        self.package_net.classifier[6] = nn.Linear(4096, 17)  # Numero esatto di classi nel nostro dataset.

        self.softmax = nn.Softmax(dim=1)
        self.load_dict(self.package_net, data_augmentation + "retraining_alexnet_17classes_newDataset")
        print("Parametri caricati")
        self.dictionary = dictionary

    def load_dict(self, net, name):
        net.load_state_dict(torch.load("./pth/" + name + ".pth", map_location='cpu'))

    def get_class(self, net, img, n_classes, key="", data_augmentation=""):
        net.eval()
        net.cpu()

        if key == "":
            mean_pre_trained = [0.485, 0.456, 0.406]
            std_pre_trained = [0.229, 0.224, 0.225]
        else:
            path = data_augmentation + 'single_dataset/' + key
            txt_file = path + '/train.txt'

        mean_pre_trained = [0.485, 0.456, 0.406]
        std_pre_trained = [0.229, 0.224, 0.225]

        img = img.resize((256, 256))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_pre_trained, std_pre_trained)])
        img = transform(img)
        x = Variable(img).cpu()
        x = x.unsqueeze(0)

        # Carico la giusta net in base al parametro key. Se diverso da vuoto allora devo caricare i giusti parametri e crearne
        # una al volo
        if key != "":
            net = models.squeezenet1_0(pretrained=True)

            net.classifier[0] = nn.Dropout(p=0.0)
            net.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))

            name_save = data_augmentation + "retraining_squeezenet_" + str(n_classes) + "classes_dataset_" + key
            # Alleno o carico lo stato attuale se train è false.
            path = "./" + name_save + ".pth"
            net.load_state_dict(torch.load('./pth/'+path, map_location='cpu'))

        pred = self.softmax(net(x)).data.cpu().numpy().copy()
        # print("Prima --> ", pred)
        pred = pred.argmax(1)
        # print(pred , key)
        # Caso di get_pastabox
        if key == "":
            filepath = "./classes_first_level.txt"
        else:
            filepath = data_augmentation + 'single_dataset/' + key + '/classes.txt'
        with open(filepath, "r+") as fp:
            for i in range(0, n_classes):
                line = fp.readline()
                n, c = line.strip().split(",")

                if int(c) == int(pred):
                    # print("name: ", n, " c: ", c)
                    name = n

        return name, pred



def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Demo(QMainWindow, QWidget):
    def __init__(self):
        super().__init__()
        self.net = TreeNet(dictionary, "Data_augmentation_")


        self.setWindowTitle('Demo')

        self.lbl = QLabel(self)
        self.package_lbl = QLabel(self)
        self.class_lbl = QLabel(self)

        self.rsize_x = 160 * 2
        self.rsize_y = 120 * 2
        self.canvas = None

        self.knn = True

        self.setGeometry(100, 100, 800, 600)

        self.import_img_btn()

        self.show()

    def show_image(self, img):
        rgba_img = img.convert("RGBA")
        qim = ImageQt(rgba_img)
        pix = QPixmap.fromImage(qim)
        self.lbl.deleteLater()
        self.lbl = QLabel(self)
        self.lbl.setPixmap(pix)
        self.lbl.resize(pix.width(), pix.height())
        width = self.geometry().width()
        height = self.geometry().height()
        self.lbl.move(width / 2 - pix.width() / 2, height / 2 - pix.height() / 2)
        self.lbl.updateGeometry()
        self.lbl.update()
        self.update()
        self.lbl.show()

    def import_img_btn(self):
        importAct = self.button(self.process_image,"import image")
        return importAct

    def button(self, function, text):
        btn = QPushButton(text, self)
        btn.clicked.connect(function)
        return btn

    def import_file(self,):
        imp = ImportFile()

        self.path = imp.openFileNameDialog()
        if self.path == "wrong path": print("Nessun file caricato")
        print("path: ", self.path)

    def process_image(self):
        self.import_file()

        if self.path != "wrong path":
            img = Image.open(self.path).convert(
                "RGB")

            name, pred = self.net.get_class(self.net.package_net, img, 17)
            print(name, pred)
            if name == "Pasta_Integrale":
                name = "Pasta_integrale"
            if name == "Pasta_Integrale_voiello":
                name = "Pasta_integrale_voiello"
            if name == "Pasta_specialitÃ\xa0":
                name = "Pasta_specialita"

            name_product, _ = self.net.get_class(self.net.package_net, img, dictionary[name], name)

            width = self.geometry().width()
            height = self.geometry().height()

            basewidth = self.rsize_x
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)

            self.package_lbl.setText("Package: " + name)
            self.class_lbl.setText("Product: " + name_product)
            self.package_lbl.setGeometry(int(width/2 + width/4), int(height/10), 300, 30)
            self.class_lbl.setGeometry(int(width/2 + width/4), int(height/7), 300, 30)

            #img = img.resize(self.rsize_x)
            self.show_image(img)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Demo()
    sys.exit(app.exec_())