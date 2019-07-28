from Utility import *
from ScenesDataset import *

class TreeNet:
    def __init__(self, dictionary, data_augmentation = ""):
        self.package_net = models.alexnet(pretrained=True)
        self.package_net.classifier[6] = nn.Linear(4096, 17) #Numero esatto di classi nel nostro dataset.
        #Spegnamo il dropout per l'evaluation.
        self.package_net.classifier[0] = nn.Dropout(p=0.0)
        self.package_net.classifier[3] = nn.Dropout(p=0.0)
        #self.package_net = models.alexnet(num_classes = 17)
        self.softmax = nn.Softmax(dim=1)
        self.load_dict(self.package_net, "../Parametri/" + data_augmentation + "retraining_alexnet_17classes_newDataset")
        print("Parametri caricati")
        self.dictionary = dictionary
        
    def load_dict(self, net, name):
        net.load_state_dict(torch.load("./" + name + ".pth"))
        
    def print_accuracy(self):
        mean_pre_trained =[0.485, 0.456, 0.406]
        std_pre_trained =[0.229, 0.224, 0.225]
        transformss = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_pre_trained,std_pre_trained)])
        barilla_test = ScenesDataset('../Immagini/new_image_first_level','../test_first_level.txt',transform=transformss)
        barilla_test_loader_fl = torch.utils.data.DataLoader(barilla_test, batch_size=10, num_workers=0)
        lenet_mnist_predictions, lenet_mnist_gt = test_model_classification(self.package_net, barilla_test_loader_fl)
        print ("Accuracy aLexNet di train su barillatestloader: %0.2f" % \
        accuracy_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1)))
    
    
    def get_class(self, net, img, n_classes, key = "", data_augmentation = ""):
        #print(n_classes, key)
        net.eval()
        net.cpu()

        if key == "":
            mean_pre_trained =[0.485, 0.456, 0.406]
            std_pre_trained =[0.229, 0.224, 0.225]
        else:
            path = "../Immagini/" +data_augmentation + 'single_dataset/' + key
            txt_file = path + '/train.txt'
            #mean_pre_trained =[0.485, 0.456, 0.406]
            #std_pre_trained =[0.229, 0.224, 0.225]
            #transformss = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_pre_trained, std_pre_trained)])
            #dataset = ScenesDataset(path, txt_file, transform=transformss)
            #mean_pre_trained,std_pre_trained = get_mean_devst(dataset)
        
        mean_pre_trained =[0.485, 0.456, 0.406]
        std_pre_trained =[0.229, 0.224, 0.225]
        
        img = img.resize((256,256))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_pre_trained, std_pre_trained)])
        img = transform(img)
        x = Variable(img).cpu()
        x = x.unsqueeze(0)
        
        #Carico la giusta net in base al parametro key. Se diverso da vuoto allora devo caricare i giusti parametri e crearne
        #una al volo
        if key != "":
            net = models.squeezenet1_0(pretrained=True)
            #Update last layer for squeezenet1_0
            #Spegnamo il dropout.
            net.classifier[0] = nn.Dropout(p=0.0)

            net.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1,1), stride=(1,1))
            name_save = "../Parametri/" + data_augmentation + "retraining_squeezenet_"+ str(n_classes) +"classes_dataset_" + key
            #Alleno o carico lo stato attuale se train è false.
            path = "./" + name_save + ".pth"
            net.load_state_dict(torch.load(path))
        
        pred = self.softmax(net(x)).data.cpu().numpy().copy()
        #print("Prima --> ", pred)
        pred = pred.argmax(1)
        #print(pred , key)
        #Caso di get_pastabox
        if key == "":
            filepath = "../classes_first_level.txt"
        else:
            filepath = "../Immagini/" + data_augmentation + 'single_dataset/' + key + '/classes.txt'
        with open(filepath, "r+") as fp:
            for i in range(0, n_classes):
                line = fp.readline()
                n, c = line.strip().split(",")

                if int(c) == int(pred):
                    #print("name: ", n, " c: ", c)
                    name = n
        
        
        return name, pred

    