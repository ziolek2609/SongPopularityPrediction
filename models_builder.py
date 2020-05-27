# import bibliotek
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ścieżka folderu
path = '/Users/ziole/Desktop/pro/'


# wczytanie danych - zbioru treningowego i labeli
train_dataset = pd.read_csv(path + "train_dataset.csv", sep = ';')
train_labels = pd.read_csv(path + "train_labels.csv", sep = ';')

# wczytanie danych - zbioru testowego i labeli
test_dataset = pd.read_csv(path + "test_dataset.csv", sep = ';')
test_labels = pd.read_csv(path + "test_labels.csv", sep = ';')


# DataFrame dla błędów sieci
mae = []
labels = ['name', 'mae']
errors = pd.DataFrame.from_records(mae,columns = labels)


# zakres liczby neronów w pierwszej warstwie 8-16
for i in range (8,17):

    # zakres liczby neuronów w drugiej warstwie 2-6
    for j in range (2,7):

        # zakres batch: [16,64,128,256]
        for k in [16,64,128,256]:
            
            # funkcja straty: mean absolute error lub huberloss
            for l in ['mae','huber_loss']:

                for m in ['linear', 'sigmoid']:

                    # budowa modelu
                    model= Sequential()
                    model.add(Dense(i, activation = 'relu', input_shape = [len(train_dataset.keys())]))
                    model.add(Dense(j, activation = 'relu'))
                    model.add(Dense(1, activation = m))

                    # kompilacja i trening sieci
                    model.compile(optimizer = 'adam', loss = l, metrics = ['mae'])
                    model.fit(train_dataset, train_labels, epochs = 64, validation_split = 0.2, verbose = 1, batch_size = k)

                    # zapis modelu z odpowiednią nazwą
                    first_layer_neurons_num = str(i)
                    second_layer_neurons_num = str(j)
                    batch_num = str(k)
                    loss_func = str(l)
                    act = str(m)
                    name = path + 'wyniki/' + '1ln-' + first_layer_neurons_num + '_2ln-' + second_layer_neurons_num + '_act-' + act + '_bat-' + batch_num + '_loss-' + loss_func + '.h5'
                    model.save(name)

                    #sprawdzenie błędu sieci i dodanie do errors DF
                    loss = model.evaluate(test_dataset,  test_labels, verbose=2)
                    error = [(name, loss[1])]
                    err = pd.DataFrame.from_records(error, columns = labels)
                    errors = errors.append(err, ignore_index=True)
                    print(errors)




# wiersz z najmniejszym błędem
mask = errors['mae'] == min(errors.mae)
best = errors.loc[mask, 'name']
name = best.values[0]
print(best,"\n\n", name)
best_model = tf.keras.models.load_model(name)
print(best_model.summary())


# predykcja najlepszego modelu
test_predictions = best_model.predict(test_dataset)
prediction = pd.DataFrame(test_labels)
prediction.column = "song_popularity"
prediction["predicted"] = test_predictions
print(prediction)

#wykres predykcji
fig = px.scatter(prediction, 'predicted', 'song_popularity')
fig.add_trace(go.Scatter(x=[0,1], y=[0,1]))
fig.show()
