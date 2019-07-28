import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from Utility import *


class AlexNetRetrain():
    def __init__(self, num_classes = 153):
        #Per riallenare l'ultimo layer
        self.Alex_net_last_layer =  models.alexnet(pretrained=True)
        for param in self.Alex_net_last_layer.parameters():
            param.requires_grad = False
        self.Alex_net_last_layer.classifier[6] = nn.Linear(4096, num_classes) 

        #Per riallenare completamente alexnet.
        self.Alexnet_no_pretrained = models.alexnet()
        self.Alexnet_no_pretrained.classifier[6] = nn.Linear(4096, num_classes) 

        #Per riallenare gli ultimi due layer
        self.Alexnet_last_two_layer = models.alexnet(pretrained=True)
        for param in self.Alexnet_last_two_layer.parameters():
            param.requires_grad = False            
        self.Alexnet_last_two_layer.classifier[4] = nn.Linear(4096, 4096) 
        self.Alexnet_last_two_layer.classifier[6] = nn.Linear(4096, num_classes) 


        if torch.cuda.is_available():
            self.Alex_net_last_layer.cuda()
            self.Alexnet_last_two_layer.cuda()
            self.Alexnet_no_pretrained.cuda()

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
        print("AlexNetRetrain costruttore terminato.")



    '''
        Funzione per riallenare i vari modelli nei vari casi possibili..
        choose:
            1 --> Allenamento ultimo layer.
            2 --> Allenamento ultimi due layer.
            3 --> Allenamento di tutta la rete.
        train: Se train = False lui prova a precaricare i parametri e a stampare a schermo gli indicatori di qualità.
        all_indicator: Se == True stampa anche i knn con le relative accuracy.
    '''
    def retrained(self, choose = 1, name_save = "last_layer_training_153classes_150epoche", data_augmentation = False, train = False, all_indicator = False, my_epochs = 150, my_lr = 0.001, my_momentum = 0.8):
        if data_augmentation == True:
            path = "../Parametri/Data_augmentation_" + name_save + ".pth"
            name_save = "Data_augmentation_" + name_save
            barilla_train_loader = self.barilla_train_loader_da
            barilla_test_loader = self.barilla_test_loader_da

        else:
            path = "../Parametri/" + name_save + ".pth"
            barilla_train_loader = self.barilla_train_loader
            barilla_test_loader = self.barilla_test_loader
        
        my_file = Path(path)

        if choose == 1:    
            net = self.Alex_net_last_layer
        elif choose == 2:
            net = self.Alexnet_last_two_layer
        elif choose == 3:
            net = self.Alexnet_no_pretrained
        else:
            print("Scelta errata.")
            return

        if train == False and my_file.is_file(): #Carico lo state altrimenti rialleno
            net.load_state_dict(torch.load(path))
            print("Caricato " + path)
        else:
            lenet_mnist, lenet_mnist_logs = train_classification(net, epochs = my_epochs, train_loader = barilla_train_loader,
                                                                test_loader = barilla_test_loader, 
                                                                exp_name = name_save, lr = my_lr, momentum = my_momentum)
            torch.save(net.state_dict(), "./" + name_save + ".pth")

        if data_augmentation == True:
            print("*************Train************")
            get_our_accuracy(net, self.barilla_train_loader_da, None)
            print("*************Test************")
            get_our_accuracy(net, self.barilla_test_loader_da, None)
        else:
            print("*************Train************")
            get_our_accuracy(net, self.barilla_train_loader, None)
            print("*************Test************")
            get_our_accuracy(net, self.barilla_test_loader, None)
