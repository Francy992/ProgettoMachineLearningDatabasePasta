from ScenesDataset import *
from TreeNet import *
from Utility import *
from TwoLevelClassify import *
from OurNet import *
from AlexNetRetrain import *

'''
    Oggettoo che gestisce classificazione a due livelli, ossia quella finale
'''
#twoLevelClassify = TwoLevelClassify()
''' Stampa l accuracy di train e di test finale di data augmentation e senza data augmentation. '''
#twoLevelClassify.get_accuracy_all_dataset(True)
''' Allena tutti i classificatori di secondo livello se come primo parametro viene passato True.
    Altrimenti carica i modelli già allenati e stampa tutti gli indicatori. ''' 
#twoLevelClassify.training_all_classificator(False, False, True, "Data_augmentation_")


'''
    Oggetto che gestisce l'allenamento/stampa delle informazioni per la nostra rete
'''
#ourNet = OurNet()
#ourNet.train_our_model(True)




'''
    Funzione per riallenare i vari modelli nei vari casi possibili..
    choose:
        1 --> Allenamento ultimo layer.
        2 --> Allenamento ultimi due layer.
        3 --> Allenamento di tutta la rete.
    train: Se train = False lui prova a precaricare i parametri e a stampare a schermo gli indicatori di qualità.
    all_indicator: Se == True stampa anche i knn con le relative accuracy.
'''
        
alexNetPretrained = AlexNetRetrain()
alexNetPretrained.retrained(name_save = "last_layer_training_153classes_150epoche", train = False, choose= 1)