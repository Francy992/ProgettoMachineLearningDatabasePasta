import bs4
import requests
import os  
from Level import Level
import urllib.request as req
import numpy as np



def estrapola_sorgente(url):
    if 'http://' in url:
        sorgente = requests.get(url).text
        #print("Type di sorgente: ", type(sorgente))
        return(sorgente)
    else:
        return("L'url non è valido")
    
def get_link_from_level(soup):
    soup_div = bs4.BeautifulSoup(str(soup.findAll("div", {"class": "grid__item"})), features="html.parser")
    link_list = soup_div.findAll("a")
    level = Level()
    #level.set_path(get_path_for_level(soup))
    get_path_for_level(soup, level)
    cont = 0
    for a in link_list:
        href = a.get("href")
        title = a.get("title")
        if title == "":
            title = get_title_on_level_pack(soup, cont)

        level.add_dict_on_list(title, href)
        
        if title == "Pack":
            return level
        
        cont = cont + 1

    return level    

def get_title_on_level_pack(soup, index):
    soup_div = bs4.BeautifulSoup(str(soup.findAll("span", {"class": "grid__item__name"})), features="html.parser")
    span_list = soup_div.findAll("span")
    return span_list[index].text

def get_path_for_level(soup, level):
    soup_div = bs4.BeautifulSoup(str(soup.findAll("div", {"class": "breadcrumb"})), features="html.parser")
    soup_div = bs4.BeautifulSoup(str(soup_div.findAll("div", {"class": "centered"})), features="html.parser")
    link_list = soup_div.findAll("a")
    path =""
    for a in link_list:
        #path += a.text + "/"
        level.set_path(a.text)

    #path = path[:len(path)-1]
    return path    

def scraping(end_path):
    sorgente = estrapola_sorgente(sito+end_path)
    soup = bs4.BeautifulSoup(sorgente, features="html.parser")
    new_level = get_link_from_level(soup)
    array_level.append(new_level) # append level object.
    #print(new_level.path)
    #print(new_level.list_dict)

    if(new_level.path[-1] == "Pack"):
        return

    for key,val in new_level.list_dict.items():
        scraping(val)
    #if new_level.path == "finisce con pack":
        #return
    #dictionary = new_level.list_dict
    #ciclo dictionary e scraping di value
        #scraping(path.)        
    #get_path_for_level(soup)
    #print(array_level[0].list_dict)

def save_image():
    for level in array_level:
        if level.path[-1] == "Pack":
            #Create path
            print("Level list dict len == --> ", len(level.list_dict))
            for key,val in level.list_dict.items():
                path = path_project + level.get_string_path() + "\\"+key
                try:  
                    os.makedirs(path)
                except OSError:  
                    print ("Creation of the directory %s failed" % path)
                else:  
                    print ("Successfully created the directory %s " % path)  
                try:          
                    sorgente = estrapola_sorgente(sito+val)
                    soup = bs4.BeautifulSoup(sorgente, features="html.parser")
                    img_list = get_image_from_soup(soup)                
                    if(len(img_list)==1):
                        img_url_front = sito + img_list[0]
                        req.urlretrieve(img_url_front, path + "\Front.png")
                    elif(len(img_list) == 2):
                        img_url_left = sito + img_list[0]
                        req.urlretrieve(img_url_left, path + "\Left.png")
                        img_url_right = sito + img_list[1]
                        req.urlretrieve(img_url_right, path + "\Right.png")
                    elif(len(img_list) == 3) :
                        img_url_left = sito + img_list[0]
                        req.urlretrieve(img_url_left, path + "\Left.png")
                        img_url_right = sito + img_list[1]
                        req.urlretrieve(img_url_right, path + "\Right.png")
                        img_url_front = sito + img_list[2]
                        req.urlretrieve(img_url_front, path + "\Front.png")
                except:
                    print("C'è stato un errore --> ")

def get_image_from_soup(soup):
    soup_div = bs4.BeautifulSoup(str(soup.findAll("div", {"class": "grid__item"})), features="html.parser")
    img_tag_list = soup_div.findAll("img")
    #print(img_tag_list)
    img_list = []
    for img in img_tag_list:
        img_list.append(img['src'])
    return img_list

def scraping_test(end_path, path_project):
    sorgente = estrapola_sorgente(sito+end_path)
    soup = bs4.BeautifulSoup(sorgente, features="html.parser")
    new_level = get_link_from_level(soup)
    array_level.append(new_level) # append level object.
    #print(new_level.path)
    #print(new_level.list_dict)
    print(new_level.list_dict)
    sorgente = estrapola_sorgente(sito+new_level.list_dict['Barilla'])
    soup = bs4.BeautifulSoup(sorgente, features="html.parser")
    new_level = get_link_from_level(soup)
    array_level.append(new_level) # append level object.
    print(new_level.list_dict)
    sorgente = estrapola_sorgente(sito+new_level.list_dict['Pasta di semola'])
    soup = bs4.BeautifulSoup(sorgente, features="html.parser")
    new_level = get_link_from_level(soup)
    array_level.append(new_level) # append level object.
    print(new_level.list_dict)
    sorgente = estrapola_sorgente(sito+new_level.list_dict['Pasta 5 Cereali'])
    soup = bs4.BeautifulSoup(sorgente, features="html.parser")
    new_level = get_link_from_level(soup)
    array_level.append(new_level) # append level object.
    print(new_level.list_dict)
    sorgente = estrapola_sorgente(sito+new_level.list_dict['Pack'])
    soup = bs4.BeautifulSoup(sorgente, features="html.parser")
    new_level = get_link_from_level(soup)
    array_level.append(new_level) # append level object.
    print(new_level.list_dict)
    print("Qui ci fermiamo, l'ultimo livello di path vale 'Path'")
    print(new_level.path[-1])
    sorgente = estrapola_sorgente(sito+new_level.list_dict['Fusilli 5 cereali'])
    soup = bs4.BeautifulSoup(sorgente, features="html.parser")
    img_list = get_image_from_soup(soup)
    path = path_project + new_level.get_string_path()
    '''try:  
        os.makedirs(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)'''
    

    imgurl = sito + img_list[0]
    req.urlretrieve(imgurl, path + "\\Left.png")
    



sito = 'http://www.barillastudio.it/'
sito_inizale = 'products.php?idf=12'

array_level = []

#sorgente = estrapola_sorgente(sito_inizale)
#scraping(sito_inizale) 
path_project = os.getcwd()  

#scraping_test(sito_inizale, path_project)
scraping(sito_inizale)

save_image()
#print("Array_level vale: ", array_level)
