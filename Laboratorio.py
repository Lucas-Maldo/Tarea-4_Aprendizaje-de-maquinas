import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos MNIST
mnist = fetch_openml('mnist_784')

# Obtener los datos y las etiquetas
X = mnist.data
y = mnist.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reductor de dimensionalidad UMAP
reducer = umap.UMAP()

# Dimensiones reducidas a evaluar
dimensions = [8, 16, 32, 64, 128]

# for dim in dimensions:
#     # Reducción de dimensionalidad a 'dim' dimensiones
#     reducer.n_components = dim
#     X_train_reduced = reducer.fit_transform(X_train)
#     X_test_reduced = reducer.transform(X_test)

# Entrenar el clasificador k-NN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Predecir las etiquetas del conjunto de prueba
y_pred = knn.predict(X_test)

# Calcular la precisión (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dimensión reducidaPrecisión (Accuracy): {accuracy}")

class SSearch:
    def __init__(self, model_file, layer_name):
        # Cargar el modelo
        model = tf.keras.models.load_model(model_file)
        model.summary()
        # Definir el submodelo (capa de embedding)
        output = model.get_layer(layer_name).output
        self.sim_model = tf.keras.Model(model.input, output)
        self.sim_model.summary()
        self.mu = np.load("mean.npy")
    
    def load_catalog(self, data_file, label_file):
        self.data_catalog = np.load(data_file)
        self.data_labels = np.load(label_file)
        print(self.data_catalog.shape)
    
    def prepare_data(self, data):
        prepared_data = np.expand_dims(data, axis=1)
        prepared_data = prepared_data - self.mu
        return prepared_data
    
    def compute_features(self, data):
        data = self.prepare_data(data)
        self.fv = self.sim_model.predict(data)
        print("FV-shape {}".format(self.fv.shape))
        return self.fv
    
    def compute_features_on_catalog(self):
        return self.compute_features(self.data_catalog)
    
    def ssearch_all(self):
        _ = self.compute_features_on_catalog()
        fv = self.fv
        normfv = np.linalg.norm(fv, ord=2, axis=1, keepdims=True)
        fv= fv/ normfv
        sim = np.matmul(fv, np.transpose(fv))
        idxq = np.random.randint(self.fv.shape[0])
        sim_q = sim[idxq,:]
        print("label {}".format(self.data_labels[idxq]))
        sort_idx = np.argsort(-sim_q)[:10]
        print(self.data_labels[sort_idx])
        self.visualize(sort_idx)
    
    def visualize(self, sort_idx):
        size = 28
        n = 10
        image = np.ones((size, n*size), dtype= np.uint8)*255
        i=0
        for i in np.arange(n):
            image[:, i*size:(i+1)*size] = self.data_catalog[sort_idx[i], :, :]
            i = i+1
        plt.axis("off")
        plt.imshow(image)
        plt.show()

if __name__ == "__main__":
    ssearch = SSearch("emnist_model", "embedding")
    ssearch.load_catalog("test_emnist_images.npy", "test_emnist_labels.npy")
    
    #Dividir los datos en entrenamiento y prueba
    # Datos de entrenamiento: 1000 muestras por clase
    train_samples_per_class= 1000
    train_indices =[]
    for i in range(1,27):
        class_indices = np.where(ssearch.data_labels == i)[0]
        train_indices.extend(class_indices[:train_samples_per_class])
        
    train_data = ssearch.data_catalog[train_indices]
    train_labels = ssearch.data_labels[train_indices]
    
    
    #Datos de prueba: 100 muestras por clase
    test_samples_per_class = 100
    test_indices =[]
    for i in range(1, 27):
        class_indices = np.where(ssearch.data_labels == i)[0]
        test_indices.extend(class_indices[-test_samples_per_class:])
    
    test_data = ssearch.data_catalog[test_indices]
    test_labels = ssearch.data_labels[test_indices]
# Reducción de dimensionalidad utilizando UMAP
# dimensions = [8,16,32,64,128]
# reduced_data = []
# acc = []
# red = []
# original_accuracy = []
# reduced_accuracy = [[] for _ in range(len(dimensions))]
# reduced_category_accuracy = np.zeros((26, len(dimensions)))
# for dim in range(len(dimensions)):
#     train_data_reshape = train_data.reshape(train_data.shape[0], -1)
#     test_data_reshape = test_data.reshape(test_data.shape[0], -1)
#     reducer = umap.UMAP(n_components=dimensions[dim])
#     reduced_train_data = reducer.fit_transform(train_data_reshape)
#     reduced_data.append(reduced_train_data)
#     reduced_test_data = reducer.transform(test_data_reshape)

#     # Entrenar el clasificador k-NN
#     knn = KNeighborsClassifier(n_neighbors=1)
#     knn.fit(reduced_train_data, train_labels)

#     # Predecir las etiquetas del conjunto de prueba
#     y_pred = knn.predict(reduced_test_data)

#     # Calcular la precisión (accuracy)
#     accuracy = accuracy_score(test_labels, y_pred)
#     print(f"Dimensión reducida: {dimensions[dim]}\tPrecisión (Accuracy): {accuracy}")
    
#     # Calcular el accuracy por categoría en el espacio reducido
#     for i in range(1,27):
#         category_indices = np.where(test_labels == i)[0]

#         if len(category_indices) > 0:
#             # for j in range(len(dimensions)):
#             reduced_category_pred = y_pred[category_indices]
#             reduced_category_accuracy[i-1, dim] = accuracy_score(test_labels[category_indices], reduced_category_pred)

# print("\nAccuracy por categoría (Espacio Reducido):")
# for i in range(26):
#     print("Categoría {}: {}".format(i, ", ".join("{:.2f}%".format(100 * acc) for acc in reduced_category_accuracy[i])))
# for dim in range(len(dimensions)):
#     temp = []
#     for i in range(26):
#         temp.append(reduced_category_accuracy[i][dim])
#     plt.bar(range(26), temp)
#     plt.show()


# Reducción de dimensionalidad utilizando PCA EXPERIMENTAL
dimensions = [8,16,32,64,128]
reduced_data = []
acc = []
red = []
original_accuracy = []
reduced_accuracy = [[] for _ in range(len(dimensions))]
reduced_category_accuracy = np.zeros((26, len(dimensions)))
for dim in range(len(dimensions)):
    train_data_reshape = train_data.reshape(train_data.shape[0], -1)
    test_data_reshape = test_data.reshape(test_data.shape[0], -1)
    reducer = PCA(n_components=dimensions[dim])
    reduced_train_data = reducer.fit_transform(train_data_reshape)
    reduced_data.append(reduced_train_data)
    reduced_test_data = reducer.transform(test_data_reshape)

    # Entrenar el clasificador k-NN
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(reduced_train_data, train_labels)

    # Predecir las etiquetas del conjunto de prueba
    y_pred = knn.predict(reduced_test_data)

    # Calcular la precisión (accuracy)
    accuracy = accuracy_score(test_labels, y_pred)
    print(f"Dimensión reducida: {dimensions[dim]}\tPrecisión (Accuracy): {accuracy}")
    
    # Calcular el accuracy por categoría en el espacio reducido
    for i in range(1,27):
        category_indices = np.where(test_labels == i)[0]

        if len(category_indices) > 0:
            # for j in range(len(dimensions)):
            reduced_category_pred = y_pred[category_indices]
            reduced_category_accuracy[i-1, dim] = accuracy_score(test_labels[category_indices], reduced_category_pred)

print("\nAccuracy por categoría (Espacio Reducido):")
for i in range(26):
    print("Categoría {}: {}".format(i, ", ".join("{:.2f}%".format(100 * acc) for acc in reduced_category_accuracy[i])))
for dim in range(len(dimensions)):
    temp = []
    for i in range(26):
        temp.append(reduced_category_accuracy[i][dim])
    plt.bar(range(26), temp)
    plt.show()
    