import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from Utility import *

class Multilayer_percetron(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(1000, 256, bias=True),
            nn.Linear(256, 184, bias=True),
            nn.ReLU(inplace = True),
            nn.Linear(184, 153, bias=True)
        )
        self.classifier =  nn.Sequential(
            nn.Softmax(dim=1)
        )
    def forward(self, xb, train = True):
        x = xb.view(-1,1000) 
        x = self.features(x)
        if train == False:
            x = self.classifier(x)
            pred = x.data.cpu().numpy().copy()
            print(pred)        
        return x


class OurNet():
    def __init__(self):
        self.Alexnet =  models.alexnet(pretrained=True)
        if torch.cuda.is_available():
            self.Alexnet.cuda()
        self.OurNet = Multilayer_percetron()

        '''
        Creo il dataset delle immagini senza data augmentation
        '''
        transformss = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_pre_trained,std_pre_trained)])
        self.barilla_train = ScenesDataset('../Immagini/Dataset_base','../train.txt',transform=transformss)
        self.barilla_test = ScenesDataset('../Immagini/Dataset_base','../test.txt',transform=transformss)
        self.barilla_train_loader = torch.utils.data.DataLoader(self.barilla_train, batch_size=10, num_workers=0, shuffle=True)
        self.barilla_test_loader = torch.utils.data.DataLoader(self.barilla_test, batch_size=10, num_workers=0)

        '''
            Creo il dataset  delle immagini CON data augmentation
        '''

        transformss = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_pre_trained,std_pre_trained)])
        self.barilla_train_da = ScenesDataset('../Immagini/Dataset_data_augmentation','../train_data_augmentation.txt',transform=transformss)
        self.barilla_test_da = ScenesDataset('../Immagini/Dataset_data_augmentation','../test_data_augmentation.txt',transform=transformss)
        self.barilla_train_loader_da = torch.utils.data.DataLoader(self.barilla_train_da, batch_size=10, num_workers=0, shuffle=True)
        self.barilla_test_loader_da = torch.utils.data.DataLoader(self.barilla_test_da, batch_size=10, num_workers=0)
        print("OurNet costruttore terminato.")

    '''
        Funzione per allenare il nostro modello o nel caso di train = False precaricare i parametri allenati e stampare le accuracy.

    '''
    def train_our_model(self, data_augmentation = False, epochs = 150, name_train = "our_net", train = False):
        print("Avviato OurNet.train_our_model")
        if data_augmentation == True: 
            name_train = "Data_augmentation_" + name_train  
            barilla_train_loader_OB = torch.utils.data.DataLoader(self.barilla_train_da, batch_size=1, num_workers=0, shuffle=True)
            barilla_test_loader_OB = torch.utils.data.DataLoader(self.barilla_test_da, batch_size=1, num_workers=0)
        else:
            barilla_train_loader_OB = torch.utils.data.DataLoader(self.barilla_train, batch_size=1, num_workers=0, shuffle=True)
            barilla_test_loader_OB = torch.utils.data.DataLoader(self.barilla_test, batch_size=1, num_workers=0)
        
        path = "../Parametri/" + name_train + ".pth"
        my_file = Path(path)
        if train == False and my_file.is_file():#Carico lo state altrimenti rialleno
            self.OurNet.load_state_dict(torch.load(path))
            print("Caricato lo stato per il classificatore " + name_train + ".")
        else:
            feature_train = extract_features(barilla_train_loader_OB, self.Alexnet)
            feature_test = extract_features(barilla_test_loader_OB, self.Alexnet)
            train_classification_our_network(self.OurNet, epochs, train_loader = feature_train, test_loader = feature_test, lr = 0.001, exp_name = name_train)
        
        if data_augmentation == True:
            get_our_accuracy(self.Alexnet, self.barilla_train_loader_da, self.OurNet)
            get_our_accuracy(self.Alexnet, self.barilla_test_loader_da, self.OurNet)
        else:
            get_our_accuracy(self.Alexnet, self.barilla_train_loader, self.OurNet)
            get_our_accuracy(self.Alexnet, self.barilla_test_loader, self.OurNet)
