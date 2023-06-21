import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class SSearch :
    def __init__(self, model_file, layer_name, k):
        self.k = k
        #loading the model
        model = tf.keras.models.load_model(model_file)         
        model.summary()
        #defining the submodel (embedding layer)                                        
        output = model.get_layer(layer_name).output
        self.sim_model = tf.keras.Model(model.input, output)        
        self.sim_model.summary()
        self.mu = np.load('mean.npy')
        #print('mu {}'.format(self.mu))
        #loading data        
        
    def load_catalog(self, data_file, label_file):
        self.data_catalog = np.load(data_file)
        self.data_labels = np.load(label_file)
        print(self.data_catalog.shape)    

    def prepare_data(self, data):
        prepared_data= np.expand_dims(data, axis = -1)
        prepared_data= prepared_data - self.mu
        return prepared_data
        
    def compute_features(self, data):
        data = self.prepare_data(data)                        
        self.fv = self.sim_model.predict(data)         
        print('FV-shape {}'.format(self.fv.shape))   
        return self.fv
#
    def compute_features_on_catalog(self):
        return self.compute_features(self.data_catalog)
    
    def ssearch_all(self):
        _ = self.compute_features_on_catalog()
        fv = self.fv
        normfv = np.linalg.norm(fv, ord = 2, axis = 1, keepdims = True)        
        fv = fv / normfv
        sim = np.matmul(fv, np.transpose(fv))
        self.sorted_sim = np.argsort(-sim, axis = 1)
        #tomamos un idx aleatorio
        idxq = np.random.randint(self.fv.shape[0]);        
        print('label {}'.format(self.data_labels[idxq]))
        k = 10        
        sorted_idx = self.sorted_sim[idxq, : k]
        #tomamos los 10 más cercanos 
        print(self.data_labels[sorted_idx])
        # self.visualize(sorted_idx)
    """    
    def getClass(self, idxq):
        k = 1
        nn = self.data_labels[self.sorted_sim[idxq,1:k+1]]
        print(nn)
        cl = nn[0]
        print(cl)
        print('clase correcta/inferida {}/{}'.format(self.data_labels[idxq], cl))       
        return cl
    """
    def getClass(self, idxq):
        nn = self.data_labels[self.sorted_sim[idxq, 1:self.k+1]]
        # retornamos la clase con mayor cantidad de votos
        # print((nn))
        # print(set(nn))
        # print(max(set(nn), key = nn.tolist().count))
        return max(set(nn), key = nn.tolist().count)
   
    def visualize(self, sort_idx):    
        size = 28
        n = 10
        image = np.ones((size, n*size), dtype = np.uint8)*255                        
        i = 0
        for i in np.arange(n) :
            image[:, i * size:(i + 1) * size] = self.data_catalog[sort_idx[i], : , : ]
            i = i + 1   
        plt.axis('off')     
        plt.imshow(image)
        plt.show()

    def calculate_accuracy(self, label):
        correct = 0
        total = 0
        for i in range(len(self.data_labels)):
            if self.data_labels[i]==label:
                total+=1
                if self.getClass(i) == label:
                    correct+=1
        if total == 0:
            accuracy = 0
        else:
            accuracy = correct/ total
        print("Accuracy for label {}: {:-2f}%".format(label, accuracy*100))
        return accuracy

    def calculate_accuracy_total(self):
        correct = 0
        for i in range(len(self.data_labels)):
            if self.getClass(i) == self.data_labels[i]:
                correct += 1
        accuracy = correct / len(self.data_labels)
        return accuracy
    #Necesito obtener los resultados del programa
    #y los resultados correctos, en formato lista
    #En teoria los rsultados correctos son el self.data_labels
    #y parece que los resultados del programa son self.sorted_sim

    #, y_true, y_pred
    def confusion_matrix(self, n_classes):
        y_true = self.data_labels
        y_pred = self.data_labels[self.sorted_sim]
        # for i in range(len(self.data_labels)):
        #     y_pred = np.append(y_pred, self.data_labels[self.sorted_sim])
        # print("aqui la lista",y_pred)
        y_pred = np.argmax(y_pred, axis = 1, keepdims = True)
        cm = np.zeros((n_classes, n_classes), dtype = np.int32)
        for cl_true in np.arange(1,n_classes+1) :
            y = y_pred[y_true == cl_true]
            for cl_pred in np.arange(n_classes) :            
                cm[cl_true-1, cl_pred]= np.sum(y==cl_pred+1)
        return cm
            
         
