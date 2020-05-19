# import bibliotek
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ścieżka folderu
path = '/Users/ziole/Desktop/pro/'


# wczytanie danych - zbioru treningowego i labeli
train_dataset = pd.read_csv(path + "train_dataset.csv", sep = ';')
train_labels = pd.read_csv(path + "train_labels.csv", sep = ';')

# zakres liczby neronów w pierwszej warstwie 8-16
for i in range (8,16):

    # zakres liczby neuronów w drugiej warstwie 2-6
    for j in range (2,6):

        # zakres batch: [16,64,128,256]
        for k in [16,64,128,256]:
            
            # funkcja straty: mean absolute error lub huberloss
            for l in ['mae','huber_loss']:

                # budowa modelu
                model= Sequential()
                model.add(Dense(i, activation = 'relu', input_shape = [len(train_dataset.keys())]))
                model.add(Dense(j, activation = 'relu'))
                model.add(Dense(1, activation = 'sigmoid'))

                # kompilacja i trening sieci
                model.compile(optimizer = 'adam', loss = l, metrics = ['mae'])
                model.fit(train_dataset, train_labels, epochs = 64, validation_split = 0.2, verbose = 1, batch_size = k)

                # zapis modelu z odpowiednią nazwą
                first_layer_neurons_num = str(i)
                second_layer_neurons_num = str(j)
                batch_num = str(k)
                loss_func = str(l)
                name = path + 'wyniki/' + '1ln-' + first_layer_neurons_num + '_2ln-' + second_layer_neurons_num + '_bat-' + batch_num + '_loss-' + loss_func + '.h5'
                model.save(name)
