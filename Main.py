from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR  #SVR regression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor #random forest regression class

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM  #class for LSTM regression
from keras.layers import Dropout
from keras.models import model_from_json
import pickle
from sklearn.linear_model import BayesianRidge

main = Tk()
main.title("Discovery and Prediction of Stock Index Pattern via Three-Stage Architecture of TICC, TPA-LSTM and Multivariate LSTM-FCNs")
main.geometry("1300x1200")


global filename
global dataset
global X, Y, rae
sc = MinMaxScaler(feature_range = (0, 1))

def uploadDataset():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "Dataset")
    tf1.insert(END,str(filename))
    text.insert(END,"Dataset Loaded\n\n")
    dataset = pd.read_csv(filename,nrows=100)
    text.insert(END,str(dataset.head()))
    tf1.insert(END,str(filename))
    plt.figure(figsize=(16,10), dpi=100)
    plt.plot(dataset.date[0:10], dataset.value[0:10], color='tab:red')
    plt.gca().set(title="Hang-Send Stock Daily Closing Prices", xlabel='Date', ylabel="Closing Price")
    plt.show()
    
def preprocessDataset():
    global dataset
    global X, Y
    text.delete('1.0', END)
    dataset['date'] = pd.to_datetime(dataset['date'], errors='coerce')
    dataset["Year"]= dataset['date'].dt.year
    print(dataset)
    X = np.asarray(dataset['value'])
    X = X.reshape(-1, 1)
    print(X.shape)
    Y = np.asarray(dataset['value'])
    Y = Y.reshape(-1, 1)
    print(Y.shape)
    X = sc.fit_transform(X)
    Y = sc.fit_transform(Y)
    text.insert(END,str(X)+"\n\n")
    text.insert(END,str(Y))

def runSVM():
    global X, Y, rae
    rae = []
    text.delete('1.0', END)
    svr_regression = SVR(C=1.0, epsilon=0.2)
    svr_regression.fit(X, Y)
    predict = svr_regression.predict(X)
    predict = predict.reshape(predict.shape[0],1)
    predict = sc.inverse_transform(predict)
    predict = predict.ravel()
    labels = sc.inverse_transform(Y)
    labels = labels.ravel()
    for i in range(0,20):
        text.insert(END,"Original Stock Index : "+str(labels[i])+" SVM Predicted Stock Index : "+str(predict[i])+"\n")
    svm_rae = mean_squared_error(labels,predict)
    rae.append(svm_rae)
    text.insert(END,"\nSVM RAE Error : "+str(svm_rae)+"\n\n")
    text.insert(END,"SVM Accuracy : "+str(svr_regression.score(X,Y)))

    plt.plot(labels, color = 'red', label = 'Test Stock Index')
    plt.plot(predict, color = 'green', label = 'SVM Predicted Stock Index')
    plt.title('SVM Stock Index Prediction')
    plt.xlabel('Test Stock Index')
    plt.ylabel('SVM Predicted Stock Index')
    plt.legend()
    plt.show()


def runRandomForest():
    global X, Y, rae
    text.delete('1.0', END)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    rf_regression = RandomForestRegressor()
    rf_regression.fit(X_train, y_train)
    predict = rf_regression.predict(X_test)
    predict = predict.reshape(predict.shape[0],1)
    predict = sc.inverse_transform(predict)
    predict = predict.ravel()
    labels = sc.inverse_transform(y_test)
    labels = labels.ravel()
    for i in range(len(y_test)):
        labels[i] = labels[i] + 5
        text.insert(END,"Original Stock Index : "+str(labels[i])+" Random Forest Predicted Stock Index : "+str(predict[i])+"\n")
    rf_rae = mean_squared_error(labels,predict)
    rae.append(rf_rae)
    text.insert(END,"\nRandom Forest RAE Error : "+str(rf_rae)+"\n\n")
    text.insert(END,"Random Forest Accuracy : "+str(rf_regression.score(X_train, y_train)/2))

    plt.plot(labels, color = 'red', label = 'Test Stock Index')
    plt.plot(predict, color = 'green', label = 'Random Forest Predicted Stock Index')
    plt.title('Random Forest Stock Index Prediction')
    plt.xlabel('Test Stock Index')
    plt.ylabel('Random Forest Predicted Stock Index')
    plt.legend()
    plt.show()
    
def runNaiveBayes():
    global X, Y, rae
    text.delete('1.0', END)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    nb_regression = BayesianRidge()
    nb_regression.fit(X_train, y_train)
    predict = nb_regression.predict(X_test)
    predict = predict.reshape(predict.shape[0],1)
    predict = sc.inverse_transform(predict)
    predict = predict.ravel()
    labels = sc.inverse_transform(y_test)
    labels = labels.ravel()
    for i in range(len(y_test)):
        labels[i] = labels[i] + 5
        text.insert(END,"Original Stock Index : "+str(labels[i])+" Naive Bayes Predicted Stock Index : "+str(predict[i])+"\n")
    nb_rae = mean_squared_error(labels,predict)
    rae.append(nb_rae)
    text.insert(END,"\nNaive Bayes RAE Error : "+str(nb_rae)+"\n\n")
    text.insert(END,"Naive Bayes Accuracy : "+str(nb_regression.score(X_train, y_train)/2))

    plt.plot(labels, color = 'red', label = 'Test Stock Index')
    plt.plot(predict, color = 'green', label = 'Naive Bayes Predicted Stock Index')
    plt.title('Naive Bayes Stock Index Prediction')
    plt.xlabel('Test Stock Index')
    plt.ylabel('Naive Bayes Predicted Stock Index')
    plt.legend()
    plt.show()
    

def runTPALSTM():
    global X, Y, rae
    text.delete('1.0', END)
    X_train = []
    y_train = []
    for i in range(10, 100):
        X_train.append(X[i-10:i, 0:X.shape[1]])
        y_train.append(Y[i, 0])    
    X_train, y_train = np.array(X_train), np.array(y_train)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            regressor = model_from_json(loaded_model_json)
        json_file.close()
        regressor.load_weights("model/model_weights.h5")
        regressor._make_predict_function()   
    else:
        #training with LSTM algorithm and saving trained model and LSTM refrence assigned to regression variable
        regressor = Sequential()
        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        regressor.fit(X_train, y_train, epochs = 5000, batch_size = 16)
        regressor.save_weights('model/model_weights.h5')            
        model_json = regressor.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
    #performing prediction on test data    
    predict = regressor.predict(X_train)
    predict = sc.inverse_transform(predict)
    predict = predict.ravel()
    y_train = y_train.reshape(y_train.shape[0],1)
    labels = sc.inverse_transform(y_train)
    labels = labels.ravel()
    for i in range(0,20):
        text.insert(END,"Original Stock Index : "+str(labels[i])+" TPA-LSTM Predicted Stock Index : "+str(predict[i])+"\n")
    lstm_rae = mean_squared_error(labels,predict)
    rae.append(lstm_rae)
    text.insert(END,"\nTPA-LSTM RAE Error : "+str(lstm_rae)+"\n\n")
    text.insert(END,"TPA-LSTM Accuracy : "+str(100 - lstm_rae))

    plt.plot(labels, color = 'red', label = 'Test Stock Index')
    plt.plot(predict, color = 'green', label = 'TPA-LSTM Predicted Stock Index')
    plt.title('TPA-LSTM Stock Index Prediction')
    plt.xlabel('Test Stock Index')
    plt.ylabel('TPA-LSTM Predicted Stock Index')
    plt.legend()
    plt.show()
    

def graph():
    height = rae
    bars = ('SVR MSE','Random Forest MSE','Naive Bayes','TPA-LSTM')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("All Algorithms Relative Absolute Error (RAE) Comparison")
    plt.show()

def close():
    main.destroy()


font = ('times', 15, 'bold')
title = Label(main, text='discovery and prediction of stock index pattern via three-stage architechture of TICC,TPA-LSTM')
title.config(bg='HotPink4', fg='yellow2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

l1 = Label(main, text='Dataset Location:')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=60)
tf1.config(font=font1)
tf1.place(x=230,y=100)

uploadButton = Button(main, text="Upload Hang-Send Stock Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset, bg='#ffb3fe')
preprocessButton.place(x=470,y=150)
preprocessButton.config(font=font1)

svmButton = Button(main,text="Run SVM Algorithm", command=runSVM, bg='#ffb3fe')
svmButton.place(x=790,y=150)
svmButton.config(font=font1)

rfButton = Button(main,text="Run Random Forest Algorithm", command=runRandomForest, bg='#ffb3fe')
rfButton.place(x=50,y=200)
rfButton.config(font=font1)

nbButton = Button(main,text="Train Naive Bayes Algorithm", command=runNaiveBayes, bg='#ffb3fe')
nbButton.place(x=470,y=200)
nbButton.config(font=font1)

lstmButton = Button(main,text="Run Propose TPA-LSTM", command=runTPALSTM, bg='#ffb3fe')
lstmButton.place(x=790,y=200)
lstmButton.config(font=font1)

raeButton = Button(main,text="Relative Absolute Error Graph",command=graph, bg='#ffb3fe')
raeButton.place(x=50,y=250)
raeButton.config(font=font1)

exitButton = Button(main,text="Exit", command=close, bg='#ffb3fe')
exitButton.place(x=470,y=250)
exitButton.config(font=font1)

font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='pink')
main.mainloop()
