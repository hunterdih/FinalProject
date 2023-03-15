from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def simple_mlp_model(x_train, y_train, x_test, y_test, layer_1_nodes=1, activation_function='sigmoid', optimizer_function='adam', loss_function='sparse_categorical_crossentropy', data_metrics=['accuracy']):
    """
    :param x_train: Input training data, should be of shape ROWS=SAMPLE, COLUMNS=PARAMETERS
    :param y_train: Input training set labels, should be of shape ROWS=SAMPLE, COLUMNS=1(LABEL)
    :param x_test: Input test data, should be of shape ROWS=SAMPLE, COLUMNS=PARAMETERSr
    :param y_test: Input test set labels, should be of shape ROWS=SAMPLE, COLUMNS=1(LABEL)
    :param layer_1_nodes:
    :param activation_function: string containing the activation function
    :param optimizer_function:
    :param loss_function:
    :param data_metrics:
    :return y_results_train, y_results_test: Resulting labeled dataset
    """
    # Need to determine the number of parameters in the dataset to optimize (this will determine the size of the nn)
    numParameters = x_train.shape[1]
    label_num = int(np.max(y_test))
    # Create the model
    model = Sequential()
    model.add(Dense(units=numParameters, kernel_initializer='random_uniform', activation=activation_function, input_dim = numParameters))

    # If there were multiple hidden layers to the network, they would be added here
    # Create output layer
    model.add(Dense(units=numParameters, kernel_initializer='random_uniform', activation='softmax'))
    # Compile the model
    model.compile(optimizer=optimizer_function, loss=loss_function, metrics=data_metrics)

    history = model.fit(x_train, y_train, validation_split=0.35, batch_size=50, epochs=10, verbose=1)

    y_results_test = model.predict(x_test)

    print(history.history.keys())
    # summarize history for accuracy
    plt.figure(0)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.figure(1)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.figure(2)
    y_results_test = pd.DataFrame(y_results_test)
    y_results_test = y_results_test.idxmax(axis=1)
    cm = confusion_matrix(y_test, y_results_test, normalize='true')
    plt.imshow(cm, cmap='BuPu')
    for (i,j),label in np.ndenumerate(cm):
        plt.text(i, j, str(round(label, 4)), ha='center', va='center')
    plt.colorbar
    plt.show()
    return y_results_test


if __name__ == '__main__':
    print(f'Starting...')
    dataset = pd.read_csv(r'C:\Users\David Hunter\OneDrive\Northeastern Classes\Graduate\EECE5644MachineLearning\Homework1\HumanActivityData.csv', delimiter=',')
    y = dataset["Activity"]
    x = dataset.drop("Activity", axis=1)
    x = pd.DataFrame(MinMaxScaler().fit_transform(x.loc[:].values), columns=x.columns)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    y_vr = simple_mlp_model(np.asarray(x_train), y_train, np.asarray(x_test), y_test)
    print(f'Done...')
