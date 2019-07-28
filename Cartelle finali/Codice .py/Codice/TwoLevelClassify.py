from Utility import *
from TreeNet import * 
from ScenesDataset import *
import os 
from pathlib import Path

class TwoLevelClassify:

    def __init__(self):        
        transformss = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_pre_trained,std_pre_trained)])
        barilla_train = ScenesDataset('../Immagini/Data_augmentation_new_image_first_level','../Data_augmentation_train_first_level.txt',transform=transformss)
        barilla_test = ScenesDataset('../Immagini/Data_augmentation_new_image_first_level','../Data_augmentation_test_first_level.txt',transform=transformss)
        self.barilla_train_loader_fl = torch.utils.data.DataLoader(barilla_train, batch_size=10, num_workers=0, shuffle=True)
        self.barilla_test_loader_fl = torch.utils.data.DataLoader(barilla_test, batch_size=10, num_workers=0)
        self.dictionary = {  'Bio': 4,
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
        self.treeNetDataAugmentation = TreeNet(self.dictionary, "Data_augmentation_")
        self.treeNet = TreeNet(self.dictionary)

    def get_new_name_for_classname(self, name):
        if name == "Pasta_Integrale":
            name = "Pasta_integrale"
        if name == "Pasta_Integrale_voiello":
            name = "Pasta_integrale_voiello"
        if name == "Pasta_specialitÃ\xa0":
            name = "Pasta_specialita"
        return name

    def classify_img(self, path_img):
        img = Image.open(path_img).convert("RGB")
        #img = Image.open("./pasta_specialita_2.jpg").convert("RGB")
        name, pred = self.treeNet.get_class(self.treeNet.package_net, img, 17)
        #print(name, pred)
        name = self.get_new_name_for_classname(name)
        self.treeNet.get_class(self.treeNet.package_net, img, self.dictionary[name], name)

    def classify_img_data_augmentation(self, path_img):
        img = Image.open(path_img).convert("RGB")
        #img = Image.open("./pasta_specialita_2.jpg").convert("RGB")
        name, pred = self.treeNetDataAugmentation.get_class(self.treeNetDataAugmentation.package_net, img, 17)
        #print(name, pred)
        name = self.get_new_name_for_classname(name)
        self.treeNetDataAugmentation.get_class(self.treeNetDataAugmentation.package_net, img, self.dictionary[name], name, data_augmentation ="Data_augmentation_")

    '''
        Restituisce l'accuracy di tutto il dataset di train e di test (se data_augmentation = True restituisce quella di quest'ultimo, altrimenti quella base)

    '''
    def get_accuracy_all_dataset(self, data_augmentation = False):
        if data_augmentation == True:
            path = '../Immagini/Data_augmentation_new_image_first_level/'
        else: 
            path = '../Immagini/new_image_first_level/'
            
        entries = os.listdir(path)
        print("Avviato get_accuracy_all_dataset.")
        TP = 0
        error = 0
        total = 0
        accuracy = []
        t = []

        if data_augmentation == False:
            filepath = "../train_first_level.txt"
        else:
            filepath = "../Data_augmentation_train_first_level.txt"
        #Scorre il file, per ogni riga calcola:
        #La classe predetta del pastabox, la classe predetta del tipo specifico, la classe originale del tipo specifico
        # Poi confronta se le due classi sono uguali incrementa TP. 
        with open(filepath, "r+") as fp:
            for i in range(len(open(filepath, "r+").readlines())):
                line = fp.readline()
                entry, c = line.strip().split(",")
                total += 1
                #Apro l'immagine
                #print("\nImmagine-->", entry)
                img = Image.open(path + entry).convert("RGB")
                if data_augmentation == True:
                    name, pred = self.treeNetDataAugmentation.get_class(self.treeNetDataAugmentation.package_net, img, 17, data_augmentation ="Data_augmentation_")
                else:
                    name, pred = self.treeNet.get_class(self.treeNet.package_net, img, 17)
                #print(name, dictionary[name])
                name = self.get_new_name_for_classname(name)

                if data_augmentation == True:
                    final_name, final_pred = self.treeNetDataAugmentation.get_class(self.treeNetDataAugmentation.package_net, img, self.dictionary[name], name, data_augmentation ="Data_augmentation_")
                else:
                    final_name, final_pred = self.treeNet.get_class(self.treeNet.package_net, img, self.dictionary[name], name)

                original_name, original_pred = get_original_class(data_augmentation, entry)
                #print("Predizione package: ", name, ", predizione classe finale: ", final_name, ", classe finale reale: ", original_name)
                if final_name == original_name:
                    TP += 1
                else:
                    error += 1
                try:
                    accuracy.append(TP/total)
                    t.append(total)
                    print("Ciclo n°: ", total, ", accuracy: ", TP/total)
                except:
                    pass
            print("Totale accuracy di train-->", str(total), ", accuracy: ", str(TP/total))



        TP = 0
        error = 0
        total = 0
        accuracy = []
        t = []

            
        if data_augmentation == False:
            filepath = "../test_first_level.txt"
        else:
            filepath = "../Data_augmentation_test_first_level.txt"
            
        with open(filepath, "r+") as fp:
            for i in range(len(open(filepath, "r+").readlines())):
                line = fp.readline()
                entry, c = line.strip().split(",")
                total += 1
                #Apro l'immagine
                #print("\nImmagine-->", entry)
                img = Image.open(path + entry).convert("RGB")
                if data_augmentation == True:
                    name, pred = self.treeNetDataAugmentation.get_class(self.treeNetDataAugmentation.package_net, img, 17, data_augmentation ="Data_augmentation_")
                else:
                    name, pred = self.treeNet.get_class(self.treeNet.package_net, img, 17)
                #print(dictionary[name])
                name = self.get_new_name_for_classname(name)

                if data_augmentation == True:
                    final_name, final_pred = self.treeNetDataAugmentation.get_class(self.treeNetDataAugmentation.package_net, img, self.dictionary[name], name, data_augmentation ="Data_augmentation_")
                else:
                    final_name, final_pred = self.treeNet.get_class(self.treeNet.package_net, img, self.dictionary[name], name)

                original_name, original_pred = get_original_class(data_augmentation, entry)
                #print("Predizione package: ", name, ", predizione classe finale: ", final_name, ", classe finale reale: ", original_name)
                
                if final_name == original_name:
                    TP += 1
                else:
                    error += 1
                try:
                    accuracy.append(TP/total)
                    t.append(total)
                    print("Ciclo n°: ", total, ", accuracy: ", TP/total)
                except:
                    pass
            print("Totale accuracy di test-->", str(total), ", accuracy: ", str(TP/total))



    '''
        Preso un dictionary allena dei classificatori per le classi all'interno del dictionary
        train: Se true allena, altrimenti carica lo stato che ha già allenato. Se non lo trova allena.
        knn_indicator: Se tru restituisce le accuracy anche dei knn.
        all_indicator: Se true ci restituisce anche altri indicatori oltre all'accuracy.
        pre_name: Se passiamo Data_augmentation_ lui carica i giusti dataset per allenare per data augmentation.
        net_name: Nome della rete.
    '''

    def training_all_classificator(self, train = True, knn_indicator = False, all_indicator = False, pre_name = "", net_name = "retraining_squeezenet_"):    
        for key, num_classes in self.dictionary.items():
            print()
            print()
            print("Inizio nuovo allenamento sulla cartella " + key + " con un numero di classi totali: " + str(num_classes))
            #Creo i dataloader
            barilla_train_loader = get_dataloader("../Immagini/" + pre_name + 'single_dataset/' + key, "../Immagini/" + pre_name + 'single_dataset/'+ key +'/train.txt')
            barilla_test_loader = get_dataloader("../Immagini/" + pre_name + 'single_dataset/' + key, "../Immagini/" + pre_name + 'single_dataset/'+ key +'/test.txt', train = False)
            
            #Creo la squeeznet con i parametri preallenati, cambio l'ultimo layer e creo il nome per poi salvare lo stato di allenamento.
            net = models.squeezenet1_0(pretrained=True)
            #Update last layer for squeezenet1_0
            net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            
            #net = models.alexnet(pretrained=True)
            #net.classifier[6] = nn.Linear(4096, num_classes) #Numero esatto di classi nel nostro dataset.
            
            
            print("TODO: Da togliere, numero di parametri: " + str(sum([p.numel() for p in net.parameters()])))
            name_save = pre_name + net_name + str(num_classes) +"classes_dataset_"+key

            #Alleno o carico lo stato attuale se train è false.
            path = "../Parametri/" + name_save + ".pth"
            my_file = Path(path)
            if train == False and my_file.is_file():#Carico lo state altrimenti rialleno
                net.load_state_dict(torch.load(path))
                print("Caricato lo stato per il classificatore " + name_save + ".")
            else:
                #net.load_state_dict(torch.load(path)) #Buoni parametri: lr = 0.0009, momentum = 0.8
                lenet_mnist, lenet_mnist_logs = train_classification(net, epochs=80, train_loader = barilla_train_loader,
                                                                    test_loader = barilla_test_loader,
                                                                    exp_name = name_save, lr = 0.0001, momentum = 0.8)
                print("Allenamento finito, stato salvato correttamente.")
                torch.save(net.state_dict(), "./" + name_save + ".pth")
                
                
            #Stampo accuracy ed eventualmente anche gli altri indicatori se richiesto.
            lenet_mnist_predictions, lenet_mnist_gt = test_model_classification(net, barilla_train_loader)
            print ("Accuracy " + net_name +" di train su "+ name_save +": %0.2f" % \
                accuracy_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1)))
            if all_indicator: 
                print("Precision_score-->")
                print(precision_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1), average = 'macro'))
                print("Recall_score-->")
                print(recall_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1), average = 'macro'))
                print("F1_Score-->")
                print(f1_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1), average = 'macro'))
                print("Matrice di confusione-->\n")
                print(confusion_matrix(lenet_mnist_gt,lenet_mnist_predictions.argmax(1)))
            
            lenet_mnist_predictions, lenet_mnist_gt = test_model_classification(net, barilla_test_loader)
            print ("Accuracy " + net_name +" di test su "+ name_save +": %0.2f" % \
                accuracy_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1)))
            if all_indicator: 
                print("Precision_score-->")
                print(precision_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1), average = 'macro'))
                print("Recall_score-->")
                print(recall_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1), average = 'macro'))
                print("F1_Score-->")
                print(f1_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1), average = 'macro'))
                print("Matrice di confusione-->\n")
                print(confusion_matrix(lenet_mnist_gt,lenet_mnist_predictions.argmax(1)))
            
            #Parte relativa al knn.
            if knn_indicator:
                #Creo i vari knn passandogli le feature estratte dalla rete allenata.
                knn_1 = KNN(n_neighbors=1)
                knn_3 = KNN(n_neighbors=3)
                knn_5 = KNN(n_neighbors=5)

                barilla_train = get_scenedataset("../Immagini/" + pre_name +'single_dataset/' + key, "../Immagini/" +pre_name +'single_dataset/'+ key +'/train.txt')
                barilla_test = get_scenedataset("../Immagini/" + pre_name +'single_dataset/' + key, "../Immagini/" +pre_name +'single_dataset/'+ key +'/test.txt')

                barilla_train_loader_OB = torch.utils.data.DataLoader(barilla_train, batch_size=1, num_workers=0, shuffle=True)
                barilla_test_loader_OB = torch.utils.data.DataLoader(barilla_test, batch_size=1, num_workers=0)


                if torch.cuda.is_available():
                    net = net.cuda()
                    torch.cuda.empty_cache()
                net.eval()

                input_for_datafram_train, label_array_train = get_dataframe(barilla_train_loader_OB, net)
                df = pd.DataFrame(input_for_datafram_train)
                knn_1.fit(df, label_array_train)
                knn_3.fit(df, label_array_train)
                knn_5.fit(df, label_array_train)
                
                #Estraggo le feature del test loader e calcolo accuracy e vari indicatori su tutto.
                feature_test = extract_features(barilla_test_loader_OB, net)
                print("**************************1NN**************************")
                pred_label, target_label = get_pred_label_and_target_label(knn_1, feature_test)
                print("Accuracy-->", accuracy_score(target_label,pred_label))
                if all_indicator:
                    print("Precision_score-->", precision_score(target_label,pred_label, average = 'macro'))
                    print("Recall_score-->", recall_score(target_label,pred_label, average = 'macro'))
                    print("F1_Score-->", f1_score(target_label,pred_label, average = 'macro'))
                    print("Matrice di confusione-->\n")
                    print(confusion_matrix(target_label,pred_label))
                
                print("**************************3NN**************************")
                pred_label, target_label = get_pred_label_and_target_label(knn_3, feature_test)
                print("Accuracy-->", accuracy_score(target_label,pred_label))
                if all_indicator:
                    print("Precision_score-->", precision_score(target_label,pred_label, average = 'macro'))
                    print("Recall_score-->", recall_score(target_label,pred_label, average = 'macro'))
                    print("F1_Score-->", f1_score(target_label,pred_label, average = 'macro'))
                    print("Matrice di confusione-->\n")
                    print(confusion_matrix(target_label,pred_label))
                
                if pre_name != "" and key != "Pasta_emiliane_chef": #Si rompe in questo caso perchè non abbiamo 5 esempi.
                    print("**************************5NN**************************")
                    pred_label, target_label = get_pred_label_and_target_label(knn_5, feature_test)
                    print("Accuracy-->", accuracy_score(target_label,pred_label))
                    if all_indicator:
                        print("Precision_score-->", precision_score(target_label,pred_label, average = 'macro'))
                        print("Recall_score-->", recall_score(target_label,pred_label, average = 'macro'))
                        print("F1_Score-->", f1_score(target_label,pred_label, average = 'macro'))
                        print("Matrice di confusione-->\n")
                        print(confusion_matrix(target_label,pred_label))