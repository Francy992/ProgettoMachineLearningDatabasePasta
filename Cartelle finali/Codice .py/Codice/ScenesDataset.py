import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
'''
    Variabili globali
'''
width = 256
height = 256
mean_pre_trained = [0.485, 0.456, 0.406]
std_pre_trained = [0.229, 0.224, 0.225]
num_classes = 153
training = True
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
        im = Image.open(self.base_path + "/" + f).convert("RGB")
        
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