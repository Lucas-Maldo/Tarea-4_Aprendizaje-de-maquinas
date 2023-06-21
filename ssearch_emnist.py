import numpy as np
import ssearch.ssearch as ss
if __name__ == '__main__' :
    k = 5
    ssearch = ss.SSearch('emnist_model','embedding', k)
    ssearch.load_catalog('test_emnist_images.npy', 'test_emnist_labels.npy')
    #ssearch.compute_features_on_catalog()    
    ssearch.ssearch_all()


    indice_de_imagen =  2
    print(ssearch.getClass(indice_de_imagen)) #(knn k= 1)
             
    for label in range(1,27):
        accuracy = ssearch.calculate_accuracy(label)
    accuracy_total = ssearch.calculate_accuracy_total()
    matrix = ssearch.confusion_matrix(26)
    print(accuracy_total)
    print(matrix)
  
   # print(cm)
