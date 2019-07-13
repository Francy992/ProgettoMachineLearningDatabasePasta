import bs4
import requests
import os  
import numpy as np
import random
from PIL import Image
started_path = os.getcwd() + "\\pack\\Meal solution"

num_class = 0
num_only_one_image = 0
train = open("train.txt","w+")
test = open("test.txt", "w+")
classes_file = open("classes.txt", "w+")

path_new_image = "C:\Git\progetto_machine_learning_pasta\\new_image"

def get_name(path):
    path_list = path.split(os.sep)
    index = path_list.index("pack")
    #Il nuovo path sarà costituito dal nome del percorso a partire dal pack in modo da non avere duplicati.
    name = ""
    for i in range(index+1, len(path_list)):
        prov_name = path_list[i].replace(" ", "")
        name += prov_name + "_"    
    #Tolgo l'ultimo trattino
    #print(name)
    name = name[:-1]
    return name

def get_classes(path):
    path_list = path.split(os.sep)
    index_class = len(path_list) - 2 #L'ultimo sarà il nome dell'immagine, quindi prendo il penultimo.
    classes = path_list[index_class].replace(" ", "")
    return classes


def save_image(original_path, name):
    #Apro l'immagine e la salvo in un'altro path.
    img = Image.open(original_path)#Copio l'immagine e la rinomino. Forse.
    print("Dopo")
    global path_new_image
    print("Final name: ", path_new_image+"\\"+name)
    img.save(path_new_image+"\\"+name)


def start(path): #path è sempre un path globale.
    actually_dir = os.listdir(path)
    #Caso base, sono in una cartella dove ci sono solo immagini.
    if(len(actually_dir) == 0): #Per evitare che si rompa nel caso in cui non vi sia neanche un'immagine (C:\Git\progetto_machine_learning_pasta\pack\Meal solution\Barilla\Pasta di semola\Pasta cello\Mezzi  canneroni 48 cello 1Kg)
        return
    new_path = path+"\\"+actually_dir[0]
    if os.path.isdir(new_path) == False:
        #print("Sono alla base, la cartella sopra di me è: ", path)
        global num_class
        global num_only_one_image
        global train
        global test
        provv = 0
        print(len(actually_dir))
        if len(actually_dir) == 1:
            num_only_one_image += 1   #Nel caso in cui abbia solamente una foto bypasso perchè non avrei come creare test e train set.
        elif len(actually_dir) == 2: #Uno per train e uno per test scelti in modo casuale.
            rand = random.randint(0, 1)
            path_0 = path+"\\"+actually_dir[0]
            path_1 = path+"\\"+actually_dir[1]
            name_0 = get_name(path_0)
            name_1 = get_name(path_1)
            #Copio le immagini e le rinomino.
            save_image(path_0, name_0)
            save_image(path_1, name_1)
            print("Ho finito di salvare")
            #Aggiungo la riga nel file.
            str_0 = name_0 + ", " + str(num_class) + "\n"
            str_1 = name_1 + ", " + str(num_class) + "\n"
            if rand == 0:
                test.write(str_0)
                train.write(str_1)
            else:
                train.write(str_0)
                test.write(str_1)
            #Salvo la classe:
            classes = get_classes(path_0)
            classes_file.write(classes + "\n")
            num_class += 1

        else: #Numero massimo di immagini dovrebbe essere 3, quindi sempre random due per train e uno per test.
            rand = random.randint(0, 1)
            path_0 = path+"\\"+actually_dir[0]
            path_1 = path+"\\"+actually_dir[1]
            path_2 = path+"\\"+actually_dir[2]
            #Mi prendo i nomi a partire dal path(dove l'ultima parte è, appunto, il nome.)
            name_0 = get_name(path_0)
            name_1 = get_name(path_1)
            name_2 = get_name(path_2)

            #Copio le immagini e le rinomino.
            save_image(path_0, name_0)
            save_image(path_1, name_1)
            save_image(path_2, name_2)
            print("Ho finito di salvare")
            #Aggiungo la riga nel file.
            str_0 = name_0 + ", " + str(num_class) + "\n"
            str_1 = name_1 + ", " + str(num_class) + "\n"
            str_2 = name_2 + ", " + str(num_class) + "\n"
            num_class += 1
            #Scrivo il file
            if rand == 0:
                test.write(str_0)
                train.write(str_1)
                train.write(str_2)
            else:
                train.write(str_0)
                train.write(str_2)
                test.write(str_1)
            #Salvo la classe:
            classes = get_classes(path_0)
            classes_file.write(classes + "\n")
        
    else:
        for _dir in actually_dir: 
            #Procedura ricorsiva per arrivare alla fine dell'albero.
            start(path+"\\"+_dir)

start(started_path)
print("Il numero di classi totali sono: ", num_class, ", il numero di classi con solamente una foto sono: ", num_only_one_image)