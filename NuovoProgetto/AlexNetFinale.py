#Variabilil globali
width = 256
height = 256
mean_pre_trained =[0.485, 0.456, 0.406]
std_pre_trained =[0.229, 0.224, 0.225]

#Import
import torchvision.models as models
from torch.utils.data.dataset import Dataset
from PIL import Image
from os import path
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from torch.autograd import Variable

np.random.seed(1234)
torch.random.manual_seed(1234)

class ScenesDataset(Dataset):
    def __init__(self,base_path,txt_list,transform=None):
        #conserviamo il path alla cartella contenente le immagini
        self.base_path=base_path
        #carichiamo la lista dei file
        #sarà una matrice con n righe (numero di immagini) e 2 colonne (path, etichetta)
        self.images = np.loadtxt(txt_list,dtype=str,delimiter=',')
        #print("self.images ha i seguenti elementi:", len(self.images))
        #conserviamo il riferimento alla trasformazione da applicare
        self.transform = transform
    def __getitem__(self, index):
        #print("Get item numero -->", index)
        #recuperiamo il path dell'immagine di indice index e la relativa etichetta
        f,c = self.images[index]
        #carichiamo l'immagine utilizzando PIL e facciamo il resize a 3 canali.
        im = Image.open(path.join(self.base_path, f)).convert("RGB")
        
        #Resize:
        im = im.resize((width,height))
        #se la trasfromazione è definita, applichiamola all'immagine
        if self.transform is not None:
            im = self.transform(im)       
        
        #convertiamo l'etichetta in un intero
        label = int(c)
        #restituiamo un dizionario contenente immagine etichetta
        #print("Mentre creo il tutto, label vale-->", label, ", name vale -->", f)
        return {'image' : im, 'label':label, 'name': f}
    #restituisce il numero di campioni: la lunghezza della lista "images"
    def __len__(self):
        #print("Ho invocato len, vale-->", len(self.images))
        return len(self.images)

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
    print("Medie",m)
    print("Dev.Std.",s)
    return m, s

# Prende in input l'array di feature e il classificatore(knn.) 
def accuracy(classifier, samples):
    right_pred = 0
    for i in range(len(samples)):
        pred_label = classifier.predict(samples[i]["feature"].cpu().detach().numpy().reshape(1, -1))
        if pred_label[0] == samples[i]["label"]:
            right_pred += 1
            
    return float(right_pred)/len(samples)

def extract_features(dataset, net):
    #Presa ogni riga del dataloader li passa alla net senza attivare il layer di classificazione
    feature_dataset = []
    print("Avviato extract_feature.")
    for i, dataset_train in enumerate(dataset):
        x=Variable(dataset_train['image'], requires_grad=False)
        y=Variable(dataset_train['label'])
        x, y = x.cpu(), y.cpu()
        #if torch.cuda.is_available():
            #x, y = x.cuda(), y.cuda()
            #print("Con cuda")
        output = net(x)
        print(i)
        feature_dataset.append({"label": dataset_train['label'], "feature":output, "name": dataset_train['name']})
    return feature_dataset

def get_dataframe(dataset, net):
    print("Avviato get_dataframe.")
    feature_dataset = extract_features(dataset, net)  
    feature_dataset_matrix = np.zeros((len(feature_dataset), len(feature_dataset[0]["feature"][0])))    
    #Qui abbiamo nelle righe tutte le immagini, nella lable feature tutte le 9000 colonne, ossia le feature.
    label_array = np.zeros(len(feature_dataset))
    print("Ho finito l'extract feature.")
    for i in range(0, len(feature_dataset)):#302
        for j in range(0, len(feature_dataset[0]["feature"][0])):#9206
            if j == 0:#salviamo la y finale nell'array label_array
                label_array[i] = feature_dataset[i]['label'][0]
                print(i, end= " ")
            feature_dataset_matrix[i][j] =feature_dataset[i]["feature"][0][j] 

    return feature_dataset_matrix, label_array


net = models.alexnet(pretrained=True)
classifier = nn.Sequential(*list(net.classifier.children())[:-1])
net.classifier = classifier
print(sum([p.numel() for p in net.parameters()]))
transformss = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_pre_trained,std_pre_trained)])
barilla_train_da = ScenesDataset('Dataset_data_augmentation','train_data_augmentation.txt',transform=transformss)
barilla_test_da = ScenesDataset('Dataset_data_augmentation','test_data_augmentation.txt',transform=transformss)
barilla_train_loader_da = torch.utils.data.DataLoader(barilla_train_da, batch_size=1, num_workers=0, shuffle=True)
barilla_test_loader_da = torch.utils.data.DataLoader(barilla_test_da, batch_size=1, num_workers=0)
knn_1_da = KNN(n_neighbors=5)

#if torch.cuda.is_available():
    #net = net.cuda()
    #torch.cuda.empty_cache()
net.cpu()
net.eval()

input_for_datafram_train_da, label_array_train_da = get_dataframe(barilla_train_loader_da, net)
df_da = pd.DataFrame(input_for_datafram_train_da)
knn_1_da.fit(df_da, label_array_train_da)
feature_test_da = extract_features(barilla_test_loader_da, net)
print("Accuracy con rete preallenata e dataset base.")
print(accuracy(knn_1_da, feature_test_da))