from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import math
import time

random.seed(0)


def simple_mlp_model(x_train,
                     y_train,
                     x_test,
                     y_test,
                     layer_1_nodes=1,
                     activation_function='sigmoid',
                     optimizer_function='adam',
                     loss_function='sparse_categorical_crossentropy',
                     data_metrics=['accuracy'],
                     epoch_metric = 'val_loss',
                     time_limit = 20,
                     verbose = True):
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
    :param epoch_metric: string containing metric to base epoch generations on
    :param verbose: Display output confusion matrix, loss and accuracy metrics
    :return:
    """

    # Need to determine the number of parameters in the dataset to optimize (this will determine the size of the nn)
    numParameters = x_train.shape[1]
    label_num = int(np.max(y_test)) + 1
    # Create the model
    model = Sequential()
    model.add(Dense(units=numParameters, kernel_initializer='random_uniform', activation=activation_function, input_dim=numParameters))

    # If there were multiple hidden layers to the network, they would be added here
    # Create output layer
    model.add(Dense(units=label_num, kernel_initializer='random_uniform', activation='softmax'))
    # Compile the model
    model.compile(optimizer=optimizer_function, loss=loss_function, metrics=data_metrics)

    # Initial Train
    history = model.fit(x_train, y_train, validation_split=0.35, batch_size=150, epochs=10, verbose=1)
    start_time = time.time()
    accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    validation_loss = history.history['val_loss']
    epochs = 0
    difference = abs(history.history[epoch_metric][0] - history.history[epoch_metric][9])
    epochs += 10
    print(f'{epochs=}')
    elapsed_time = time.time() - start_time

    # Train until the accuracy difference between 10 epochs is less than 0.1% or n minutes has elapsed
    elapsed_time = 0

    while difference >= 0.0005 and elapsed_time < 60 * 5 or epochs < 50:
        history = model.fit(x_train, y_train, validation_split=0.35, batch_size=150, epochs=10, verbose=1)
        accuracy.extend(history.history['accuracy'])
        validation_accuracy.extend(history.history['val_accuracy'])
        loss.extend(history.history['loss'])
        validation_loss.extend(history.history['val_loss'])
        difference = abs(history.history[epoch_metric][0] - history.history[epoch_metric][9])
        elapsed_time = time.time() - start_time
        epochs += 10
        print(f'{epochs=}')

    y_results_test = model.predict(x_test)
    if verbose:
        print(history.history.keys())
        # summarize history for accuracy
        plt.figure(0)
        plt.plot(accuracy)
        plt.plot(validation_accuracy)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.figure(1)
        # summarize history for loss
        plt.plot(loss)
        plt.plot(validation_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.figure(2)
        y_results_test = pd.DataFrame(y_results_test)
        y_results_test = y_results_test.idxmax(axis=1)
        cm = confusion_matrix(y_test, y_results_test, normalize='true')
        plt.imshow(cm, cmap='BuPu')
        for (i, j), label in np.ndenumerate(cm):
            plt.text(j, i, str(round(label, 4)), ha='center', va='center')
        plt.colorbar
        error = round(np.sum(1-cm.diagonal())/cm.shape[0],4)
        plt.ylabel('True Label')
        plt.xlabel(f'Predicted Label, Error = {error}')
        plt.show()
    return max(accuracy), max(validation_accuracy), min(loss), min(validation_loss), epochs


def n_folds_calc(dataset, folds):
    '''

    :param dataset: Dataframe containing the entire dataset
    :param folds: Number of folds
    :return: Returns a list of indexes equal to the number of folds
    '''
    use_dataframe = dataset

    # Determine the number of entries in the dataset
    entries = use_dataframe.shape[0]
    samples_per_fold = math.floor(entries / folds)

    # Create index list
    entries_list = list(range(0, entries))

    # Shuffle the entries list to randomize sample selection in N-Fold
    random.shuffle(entries_list)
    return_entries = []
    for fold in range(folds):
        return_entries.append(entries_list[0:samples_per_fold])
        entries_list = entries_list[samples_per_fold:]

    return return_entries


def n_folds_split(data, indexes, fold):
    # Get test and train samples
    # Get test and train labels
    use_dataframe = data
    tr = use_dataframe.drop(indexes[fold])
    tst = use_dataframe.loc[use_dataframe.index[indexes[fold]]]

    return tr, tst


if __name__ == '__main__':
    print(f'Starting...')
    dataset = pd.read_csv(r'C:\Users\David Hunter\OneDrive\Northeastern Classes\Graduate\EECE5644MachineLearning\Project\archive\CAD_Dataset_Cleaned.csv', delimiter=',')
    target_parameter = "group_age"
    n_folds_crossvalidate = True
    n_folds = 10

    if n_folds_crossvalidate:
        index_list = n_folds_calc(dataset, n_folds)
        acc_list = []
        acc_val_list = []
        loss_list = []
        loss_val_list = []
        epoch_list = []
        for i in range(n_folds):
            train, test = n_folds_split(dataset, index_list, i)

            y_train = train[target_parameter]
            y_test = test[target_parameter]

            x_train = train.drop(target_parameter, axis=1)
            x_train = pd.DataFrame(MinMaxScaler().fit_transform(x_train.loc[:].values), columns=x_train.columns)
            x_test = test.drop(target_parameter, axis=1)
            x_test = pd.DataFrame(MinMaxScaler().fit_transform(x_test.loc[:].values), columns=x_test.columns)

            max_acc, max_acc_val, min_loss, min_loss_val, epoch_chosen = simple_mlp_model(np.asarray(x_train), y_train, np.asarray(x_test), y_test, verbose = False)
            acc_list.append(max_acc)
            acc_val_list.append(max_acc_val)
            loss_list.append(min_loss)
            loss_val_list.append(min_loss_val)
            epoch_list.append(epoch_chosen)

        use_fold = loss_val_list.index(min(loss_val_list))
        print(f'Fold Chosen: {use_fold}')
        train, test = n_folds_split(dataset, index_list, i)

        y_train = train[target_parameter]
        y_test = test[target_parameter]

        x_train = train.drop(target_parameter, axis=1)
        x_train = pd.DataFrame(MinMaxScaler().fit_transform(x_train.loc[:].values), columns=x_train.columns)
        x_test = test.drop(target_parameter, axis=1)
        x_test = pd.DataFrame(MinMaxScaler().fit_transform(x_test.loc[:].values), columns=x_test.columns)

        _1, _2, _3, _4, _5 = simple_mlp_model(np.asarray(x_train), y_train, np.asarray(x_test), y_test, verbose=True)

        plt.figure(3)
        plt.plot(acc_list)
        plt.plot(acc_val_list)
        plt.title('N-Folds Results (Accuracy)')
        plt.ylabel('Accuracy')
        plt.xlabel('Fold')
        plt.legend(['accuracy', 'validation accuracy'], loc='upper left')

        plt.figure(4)
        plt.plot(loss_list)
        plt.plot(loss_val_list)
        plt.title('N-Folds Results (Loss)')
        plt.ylabel('Loss')
        plt.xlabel('Fold')
        plt.legend(['loss', 'validation loss'], loc='upper left')
        plt.show()



    else:
        y = dataset[target_parameter]
        x = dataset.drop(target_parameter, axis=1)
        x = pd.DataFrame(MinMaxScaler().fit_transform(x.loc[:].values), columns=x.columns)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        y_vr = simple_mlp_model(np.asarray(x_train), y_train, np.asarray(x_test), y_test)


    print(f'Done...')
