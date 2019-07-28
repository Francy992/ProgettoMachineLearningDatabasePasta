import torchvision.models as models
from torch.utils.data.dataset import Dataset
from PIL import Image
from os import path as pathFunction
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from torch.optim import SGD
from torch.autograd import Variable
from torchnet.logger import VisdomPlotLogger, VisdomSaver
from torchnet.meter import AverageValueMeter
from pathlib import Path
from ScenesDataset import *




'''
    Restituisce media e varianza dato un oggetto di tipo ScenesDataset
'''
def get_mean_devst(dataset):
    m = np.zeros(3)
    for sample in dataset:
        m+= np.array(sample['image'].sum(1).sum(1)) #accumuliamo la somma dei pixel canale per canale
    #dividiamo per il numero di immagini moltiplicato per il numero di pixel
    m=m/(len(dataset)*width*height)
    #procedura simile per calcolare la deviazione standard
    s = np.zeros(3)
    for sample in dataset:
        s+= np.array(((sample['image']-torch.Tensor(m).view(3,1,1))**2).sum(1).sum(1))
    s=np.sqrt(s/(len(dataset)*width*height))
    #print("Medie",m)
    #print("Dev.Std.",s)
    return m, s


'''
     Prende in input l'array di samples e il classificatore(knn.)
''' 
def accuracy(classifier, samples):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    TOT = len(samples)
    for i in range(len(samples)):
        pred_label = classifier.predict(samples[i]["feature"].cpu().numpy().reshape(1, -1))
        if pred_label[0] == samples[i]["label"]:
            TP += 1
        else:
            TN += 1
        
    return float(TP)/len(samples)

'''
     Prende in input l'array di samples e il classificatore(knn.)
''' 
def get_pred_label_and_target_label(classifier, samples):
    pred_label = []
    target_label = []
    for i in range(len(samples)):
        pred_label.append(classifier.predict(samples[i]["feature"].cpu().numpy().reshape(1, -1))[0])
        target_label.append(samples[i]["label"])
    return pred_label, target_label

'''
     Prende in input l'array di samples e il classificatore(knn.) e restituisce le feature
''' 
def extract_features(dataset, net):
    #Presa ogni riga del dataloader li passa alla net senza attivare il layer di classificazione
    feature_dataset = []
    print("Avviato extract_feature.")
    for i, dataset_train in enumerate(dataset):
        print(i, end= " ")
        x=Variable(dataset_train['image'], requires_grad=False)
        y=Variable(dataset_train['label'])
        x, y = x.cpu(), y.cpu()
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
            #print("Con cuda")
        output = net(x)
        #print(output.grad, type(output.grad))

        #print("output len-->", len(output[0]), type(output), output[0], " - ", output[0].detach())
        feature_dataset.append({"label": dataset_train['label'], "feature":output[0].detach(), "name": dataset_train['name']})

    return feature_dataset

'''
     Prende in input un dataset, una rete ed eventualmente un array di feature.
     Crea un array che è possibile dare come input per creare un oggetto di tipo dataframe.
''' 
def get_dataframe(dataset, net, feature_dataset = None):
    print("Avviato get_dataframe.")
    if feature_dataset == None:
        feature_dataset = extract_features(dataset, net) 
    print("\nHo concluso extract_features, sto creare feature_dataset_matrix")
    feature_dataset_matrix = np.zeros((len(feature_dataset), len(feature_dataset[0]["feature"])))    
    label_array = np.zeros(len(feature_dataset))
    for i in range(0, len(feature_dataset)):#302
        print(i, end= " ")
        for j in range(0, len(feature_dataset[0]["feature"])):#153
            if j == 0:#salviamo la y finale nell'array label_array
                label_array[i] = feature_dataset[i]['label']
                #print(i, end= " ")
            feature_dataset_matrix[i][j] =feature_dataset[i]["feature"][j] 
    return feature_dataset_matrix, label_array


'''
     Funzione che gestisce la classificazione.
     Model: Rete.
     train_loader: oggetto di tipo dataloader
     test_loader: oggetto di tipo dataloader.
     lr: learning_rate
     epochs: Numero di epoche
     momentum: Valore del momentum
     exp_name: Nome dell'esperimento, verrà visualizzato su visdom e verrà utilizzato come nome dello stato del modello che viene salvato ad ogni epoca.
''' 
def train_classification(model, train_loader, test_loader, lr=0.001, epochs=20, momentum=0.8, exp_name = 'experiment' ):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),lr, momentum=momentum)
    loaders = {'train':train_loader, 'test':test_loader}
    losses = {'train':[], 'test':[]}
    accuracies = {'train':[], 'test':[]}
    
    
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    loss_logger = VisdomPlotLogger('line', env=exp_name, opts={'title': 'Loss', 'legend':['train','test']})
    acc_logger = VisdomPlotLogger('line', env=exp_name, opts={'title': 'Accuracy','legend':['train','test']})
    visdom_saver = VisdomSaver(envs=[exp_name])

    if torch.cuda.is_available():
        model=model.cuda()

    for e in range(epochs):
        #print("Primo ciclo for.")
        for mode in ['train', 'test']:
            #print("Secondo ciclo for.")
            
            loss_meter.reset()
            acc_meter.reset()
            
            if mode=='train':
                model.train()
            else:
                model.eval()
            epoch_loss = 0
            epoch_acc = 0
            samples = 0
            #print("Mode-->",mode)
            #print("Enumerate-->", loaders[mode])
            for i, batch in enumerate(loaders[mode]):
                #trasformiamo i tensori in variabili
                x=Variable(batch['image'], requires_grad=(mode=='train'))
                y=Variable(batch['label'])
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                    #print("Con cuda")
                #else:
                    #print("Senza cuda")
                output = model(x)
                #print(type(output))
                l = criterion(output,y)
                if mode=='train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                #print("L-->",l.item())
                acc = accuracy_score(y.cpu().data,output.cpu().max(1)[1].data)
                epoch_loss+=l.data.item()*x.shape[0]
                epoch_acc+=acc*x.shape[0]
                samples+=x.shape[0]
                '''print ("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % \
                (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss/samples, epoch_acc/samples),
                epoch_loss/samples,
                epoch_acc/samples,
                losses[mode].append(epoch_loss))'''
                accuracies[mode].append(epoch_acc)
                n = batch['image'].shape[0]
                loss_meter.add(l.item()*n,n)
                acc_meter.add(acc*n,n)
                #loss_logger.log(e+(i+1)/len(loaders[mode]), loss_meter.value()[0], name=mode)
                #acc_logger.log(e+(i+1)/len(loaders[mode]), acc_meter.value()[0], name=mode)


            loss_logger.log(e+1, loss_meter.value()[0], name=mode)
            acc_logger.log(e+1, acc_meter.value()[0], name=mode)
            if mode == "train":
                print(e, end = " ")
            #print("Fine secondo ciclo for")
        '''print("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % \
        (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss, epoch_acc))'''
        torch.save(model.state_dict(), "./" + exp_name + ".pth")
        print("Pesi aggiornati.")
    #print("Ho finito.")
    #restituiamo il modello e i vari log
    return model, (losses, accuracies)


'''
    Prende in input un modello e un dataloader di test e restituisce l'arreay di predizioni + l'array di valori reali.
''' 
def test_model_classification(model, test_loader):
    softmax = nn.Softmax(dim=1)
    model.eval()
    model.cpu()
    preds = []
    gts = []
    for batch in test_loader:
        x=Variable(batch["image"])
        x = x.cpu()
        #applichiamo la funzione softmax per avere delle probabilità
        if torch.cuda.is_available():
            x = x.cuda()
            model.cuda()
        pred = softmax(model(x)).data.cpu().numpy().copy()
        gt = batch["label"].cpu().numpy().copy()
        #print("Pred-->", pred, ", gt-->", gt)
        preds.append(pred)
        gts.append(gt)
        #print(len(preds), len(gts))
    return np.concatenate(preds),np.concatenate(gts)

'''
    Utilizzata per le predizioni del tipo output di alexnet preallenata dati come input ad un nostro modello
    Prende in input un modello e un dataloader di test e restituisce l'arreay di predizioni + l'array di valori reali.
''' 
def test_model_classification_our_network(model, our_model, test_loader):
    softmax = nn.Softmax(dim=1)
    model.eval()
    model.cpu()
    our_model.eval()
    our_model.cpu()
    preds = []
    gts = []
    for batch in test_loader:
        x=Variable(batch["image"])
        x = x.cpu()
        #applichiamo la funzione softmax per avere delle probabilità
        if torch.cuda.is_available():
            x = x.cuda()
            model.cuda()
            our_model.cuda()
            
        pred = softmax(our_model(model(x))).data.cpu().numpy().copy()
        gt = batch["label"].cpu().numpy().copy()
        #print("Pred-->", pred, ", gt-->", gt)
        preds.append(pred)
        gts.append(gt)
        #print(len(preds), len(gts))
    return np.concatenate(preds),np.concatenate(gts)

'''
    Utilizzata per le predizioni del tipo output di alexnet preallenata dati come input ad un nostro modello
    Prende in input l'array di samples e il classificatore(knn.) e restituisce le feature
''' 
def extract_features_our_network(dataset, net):
    #Presa ogni riga del dataloader li passa alla net senza attivare il layer di classificazione
    feature_dataset = []
    print("Avviato extract_feature.")
    for i, dataset_train in enumerate(dataset):
        print(i, end= " ")
        x=Variable(dataset_train["feature"], requires_grad=False)
        x = x.cpu()
        if torch.cuda.is_available():
            x = x.cuda()
        output = net(x)
        feature_dataset.append({"feature": output[0].detach(), "label":dataset_train["label"]})
    return feature_dataset


'''
    Prende una rete, una rete nostra e un dataloader.
'''
def get_our_accuracy(net, barilla_train_loader_da, our_net = None):
    #In questo caso e' l'accuracy per la classe AlexNetRetrain
    if our_net == None:
        lenet_mnist_predictions, lenet_mnist_gt = test_model_classification(net, barilla_train_loader_da)
    else:
        lenet_mnist_predictions, lenet_mnist_gt = test_model_classification_our_network(net, our_net, barilla_train_loader_da)
    print ("Accuracy: %0.2f" % \
    accuracy_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1)))
    print("Precision_score-->")
    print(precision_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1), average = 'macro'))
    print("Recall_score-->")
    print(recall_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1), average = 'macro'))
    print("F1_Score-->")
    print(f1_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1), average = 'macro'))
    print("Matrice di confusione-->\n")
    print(confusion_matrix(lenet_mnist_gt,lenet_mnist_predictions.argmax(1)))

def train_classification_our_network(model, train_loader, test_loader, lr=0.001, epochs=20, momentum=0.8, exp_name = 'experiment' ):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),lr, momentum=momentum)
    loaders = {'train':train_loader, 'test':test_loader}
    losses = {'train':[], 'test':[]}
    accuracies = {'train':[], 'test':[]}
    
    
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    loss_logger = VisdomPlotLogger('line', env=exp_name, opts={'title': 'Loss', 'legend':['train','test']})
    acc_logger = VisdomPlotLogger('line', env=exp_name, opts={'title': 'Accuracy','legend':['train','test']})
    visdom_saver = VisdomSaver(envs=[exp_name])

    if torch.cuda.is_available():
        model=model.cuda()
    for e in range(epochs):
        #print("Primo ciclo for.")
        for mode in ['train', 'test']:
            #print("Secondo ciclo for.")
            
            loss_meter.reset()
            acc_meter.reset()
            
            if mode=='train':
                model.train()
            else:
                model.eval()
            epoch_loss = 0
            epoch_acc = 0
            samples = 0
            #print("Mode-->",mode)
            #print("Enumerate-->", loaders[mode])
            for i, batch in enumerate(loaders[mode]):
                #trasformiamo i tensori in variabili
                x=Variable(batch['feature'], requires_grad=(mode=='train'))
                y=Variable(batch['label'])
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                    print("Con cuda")
                #else:
                    #print("Senza cuda")
                output = model(x)
                #print(type(output))
                #print(output)
                l = criterion(output,y)
                if mode=='train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                #print("L-->",l.item())
                acc = accuracy_score(y.cpu().data,output.cpu().max(1)[1].data)
                epoch_loss+=l.data.item()*x.shape[0]
                epoch_acc+=acc*x.shape[0]
                samples+=x.shape[0]
                print ("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % \
                (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss/samples, epoch_acc/samples),
                epoch_loss/samples,
                epoch_acc/samples,
                losses[mode].append(epoch_loss))
                accuracies[mode].append(epoch_acc)
                n = batch['feature'].shape[0]
                loss_meter.add(l.item()*n,n)
                acc_meter.add(acc*n,n)
                #loss_logger.log(e+(i+1)/len(loaders[mode]), loss_meter.value()[0], name=mode)
                #acc_logger.log(e+(i+1)/len(loaders[mode]), acc_meter.value()[0], name=mode)


            loss_logger.log(e+1, loss_meter.value()[0], name=mode)
            acc_logger.log(e+1, acc_meter.value()[0], name=mode)
            
            #print("Fine secondo ciclo for")
        print("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % \
        (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss, epoch_acc))
        torch.save(model.state_dict(), "./" + exp_name + ".pth")
        print("Pesi aggiornati.")
    print("Ho finito.")
    #restituiamo il modello e i vari log
    return model, (losses, accuracies)




'''
   Dato il nome di un'immagine ci restituisce la classe della stessa.
''' 
def get_original_class(data_augmentation, name_img):    
    name_list = name_img.split("_")
    #print(name_list[-2], name_list[-3])
    return name_list[-2], name_list[-3]

    
def get_dataloader(path, txt_file, train = True, batch_size = 10):
    mean_pre_trained =[0.485, 0.456, 0.406]
    std_pre_trained =[0.229, 0.224, 0.225]
    transformss = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_pre_trained,std_pre_trained)])
    barilla_dataset = ScenesDataset(path, txt_file, transform=transformss)
    if train == True:
        dataloader = torch.utils.data.DataLoader(barilla_dataset, batch_size, num_workers=0, shuffle=True)
    else: 
        dataloader = torch.utils.data.DataLoader(barilla_dataset, batch_size, num_workers=0)
    return dataloader

def get_scenedataset(path, txt_file, train = True, batch_size = 1):
    mean_pre_trained =[0.485, 0.456, 0.406]
    std_pre_trained =[0.229, 0.224, 0.225]
    transformss = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_pre_trained,std_pre_trained)])
    barilla_dataset = ScenesDataset(path, txt_file, transform=transformss)
    return barilla_dataset
