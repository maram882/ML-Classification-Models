# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 01:58:35 2022

@author: EL Rowad
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import tree

from tkinter import *
from tkinter import messagebox
from tkinter.simpledialog import askstring
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg




def DecisionTree():
    
    titleLabel["text"]="Hello Decision Tree"
    if(var_chk.get()==1):
        data = pd.read_csv("diabetics.csv")
    elif(var_chk.get()==2):
        data = pd.read_csv("f:\\breast_cancer.csv")
    elif(var_chk.get()==3):
        messagebox.showerror("showerror","This Dataset for Regression, Select classification Dataset")
        return
    else:
        messagebox.showerror("showerror", "Select DataSet")
        return
    if(testValue.get()=='' or float(testValue.get()) < 0 or float(testValue.get()) > 1):
        messagebox.showerror("showerror", "Test Size Range = [0:1]")
        return
    name = askstring('DT', 'Max Depth')    

        
    #normalizing
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)
    
    
    #feature_selection
    x= df_scaled.iloc[:, :-1]
    y= df_scaled.iloc[:, -1]
    model=SVC(kernel='linear')
    rfe=RFE(model, n_features_to_select=10, step=1) 
    rfe=rfe.fit(x,y)
    filter=rfe.support_
    ranking=rfe.ranking_ 
    data=x[x.columns[filter]]
    data.iloc[ : , -1]=y
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = float(testValue.get()))
    
    #training + The Algorithm
    clf = DecisionTreeClassifier(max_depth= int(name))
    clf.fit(x_train,y_train)
    predict = clf.predict(x_test)
    acc=accuracy_score(y_test, predict)
    pre=precision_score(y_test, predict)
    recal=recall_score(y_test, predict)
    erorr = mean_squared_error(y_test,predict)

    figure=plt.figure(figsize=(11,11))
    
    def create_window():
     newwindow =Toplevel(root)
     root.geometry("600x950")
     return newwindow
     
    def plot(parent):
     chart_type = FigureCanvasTkAgg(figure,parent)
     chart_type.get_tk_widget().pack()
     
     
     

    tree.plot_tree(clf, fontsize=8)
   # plt.savefig("final decision tree" , dpi=100)
    #chart_type = FigureCanvasTkAgg(figure, root)
    """chart_type.get_tk_widget().pack()"""
    lbl.config(text = "Decision Tree Acc ="+ str(acc))
    prec.config(text = "Decision Tree Precision ="+ str(pre))
    reca.config(text = "Decision Tree Recall ="+ str(recal))
    err.config(text = "Decision Tree Error ="+ str(erorr))
    rr.config(text = "") 

    plot(create_window())




    

def SvM():

    
    titleLabel["text"]="Hello SVM"
    if(var_chk.get()==1):
        data = pd.read_csv("diabetics.csv")
    elif(var_chk.get()==2):
        data = pd.read_csv("f:\\breast_cancer.csv")
    elif(var_chk.get()==3):
        messagebox.showerror("showerror","This Dataset for Regression, Select classification Dataset")
        return
    else:
        messagebox.showerror("showerror", "Select DataSet")
        return
    if(testValue.get()=='' or float(testValue.get()) < 0 or float(testValue.get()) > 1):
        messagebox.showerror("showerror", "Test Size Range = [0:1]")
        return
    name = askstring('SVM', 'Kernel Type')    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)

    x= df_scaled.iloc[:, :-1]
    y= df_scaled.iloc[:, -1]
    model=SVC(kernel='linear')
    rfe=RFE(model, n_features_to_select=10, step=1) 
    rfe=rfe.fit(x,y)
    filter=rfe.support_
    ranking=rfe.ranking_
    pima=x[x.columns[filter]]
    pima.iloc[:,-1]=y
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=float(testValue.get()))
    
    
    clf= SVC(kernel=name) #use rbf for nonlinear
    clf = clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    acc= metrics.accuracy_score(y_test, y_pred)
    pre=precision_score(y_test, y_pred)
    recal=recall_score(y_test, y_pred)
    erorr = mean_squared_error(y_test,y_pred)
    lbl.config(text = "SVM Acc ="+ str(acc))
    prec.config(text = "SVM Precision ="+ str(pre))
    reca.config(text = "SVM Recall ="+ str(recal))
    err.config(text = "SVM Error ="+ str(erorr))
    rr.config(text = "") 
    
def Rfe():
    
    titleLabel["text"]="Hello RFC"
    
    
    if(var_chk.get()==1):
        data = pd.read_csv("diabetics.csv")
    elif(var_chk.get()==2):
        data = pd.read_csv("f:\\breast_cancer.csv")
    elif(var_chk.get()==3):
        messagebox.showerror("showerror","This Dataset for Regression, Select classification Dataset")
        return
    else:
        messagebox.showerror("showerror", "Select DataSet")
        return
    if(testValue.get()=='' or float(testValue.get()) < 0 or float(testValue.get()) > 1):
        messagebox.showerror("showerror", "Test Size Range = [0:1]")
        return
        
    name = askstring('RFC', 'n_estimators')
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)

    x= df_scaled.iloc[:, :-1]
    y= df_scaled.iloc[:, -1]
    
    model=SVC(kernel='linear')
    rfe=RFE(model, n_features_to_select=10, step=1) 
    rfe=rfe.fit(x,y)
    filter=rfe.support_
    ranking=rfe.ranking_
    data=x[x.columns[filter]]
    data.iloc[:,-1]=y
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =float(testValue.get()))
   
    
    clf = RandomForestClassifier(n_estimators = int(name))  
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc= metrics.accuracy_score(y_test, y_pred)
    pre=precision_score(y_test, y_pred)
    recal=recall_score(y_test, y_pred)
    erorr = mean_squared_error(y_test,y_pred)
    lbl.config(text = "RFC Acc ="+ str(acc))
    prec.config(text = "RFC Precision ="+ str(pre))
    reca.config(text = "RFC Recall ="+ str(recal))
    err.config(text = "RFC Error ="+ str(erorr))
    rr.config(text = "") 
    
def Knn():
    titleLabel["text"]="Hello Knn"
    
    if(var_chk.get()==1):
        data = pd.read_csv("diabetics.csv")
    elif(var_chk.get()==2):
        data = pd.read_csv("f:\\breast_cancer.csv")
    elif(var_chk.get()==3):
        messagebox.showerror("showerror","This Dataset for Regression, Select classification Dataset")
        return
    else:
        messagebox.showerror("showerror", "Select DataSet")
        return
    if(testValue.get()=='' or float(testValue.get()) < 0 or float(testValue.get()) > 1):
        messagebox.showerror("showerror", "Test Size Range = [0:1]")
        return
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)

    x= df_scaled.iloc[:, :-1]
    y= df_scaled.iloc[:, -1]
    
    
    model=SVC(kernel='linear')
    rfe=RFE(model, n_features_to_select=10, step=1) 
    rfe=rfe.fit(x,y)
    filter=rfe.support_
    ranking=rfe.ranking_
    df_scaled=x[x.columns[filter]]
    df_scaled.iloc[:,-1]=y
   
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =float(testValue.get()))
    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    acc=knn.score(x_test, y_test)
    pre=precision_score(y_test, y_pred)
    recal=recall_score(y_test, y_pred)
    erorr = mean_squared_error(y_test,y_pred)
    lbl.config(text = "Knn Acc ="+ str(acc))
    prec.config(text = "Knn Precision ="+ str(pre))
    reca.config(text = "Knn Recall ="+ str(recal))
    err.config(text = "Knn Error ="+ str(erorr))
    rr.config(text = "") 

def Lr():
    
    titleLabel["text"]="Hello Linear Regression"
    

    if(var_chk.get()==1):
        messagebox.showerror("showerror", "This Dataset for Classificatoin, Select Regression Dataset")
        return
    elif(var_chk.get()==2):
        messagebox.showerror("showerror", "This Dataset for Classificatoin, Select Regression Dataset")
        return
    elif(var_chk.get()==3):
        data = pd.read_csv("salary.csv")
    
    else:
        messagebox.showerror("showerror", "Select DataSet")
        return
    if(testValue.get()=='' or float(testValue.get()) < 0 or float(testValue.get()) > 1):
        messagebox.showerror("showerror", "Test Size Range = [0:1]")
        return
        
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)

    x= df_scaled.iloc[:, :-1]
    y= df_scaled.iloc[:, -1]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =float(testValue.get()))
    LR = LinearRegression() 
    LR.fit(x_train, y_train)  
    y_pred = LR.predict(x_test)
    erorr = mean_squared_error(y_test,y_pred)
    r2=LR.coef_
    lbl.config(text = "")
    prec.config(text = "")
    reca.config(text = "")
    err.config(text = "Linear Regression Error ="+ str(erorr)) 
    rr.config(text = "Linear Regression Coefficients ="+ str(r2)) 


    

"""/////////////////////////////////////////////////////////////////////"""

root = Tk()
root.geometry("600x950")
root.title("ML Project")
    
lbl2 = Label(root, text = "Choose the DataSet",width="20", height="1",
                     font="40",bg="purple",fg="white")
var_chk=IntVar()
r1=Radiobutton(root, text="Diabetics",value=1,variable=var_chk,font="30",fg="purple")
r2=Radiobutton(root, text="Breast Cancer",value=2,variable=var_chk,font="30",fg="purple")
r3=Radiobutton(root, text="Salary Data",value=3,variable=var_chk,font="30",fg="purple")

DT=Button(root,text="Decision Tree",command=DecisionTree,width="15",height="2",
              font="30",bg="white",fg="purple")
Svm=Button(root,text="SVM Algo",command=SvM,width="15",height="2",
              font="30",bg="white",fg="purple")
RFe=Button(root,text="RFC Algo",command=Rfe,width="15",height="2",
              font="30",bg="white",fg="purple")
knn=Button(root,text="Knn Algo",command=Knn,width="15",height="2",
              font="30",bg="white",fg="purple")
Lr=Button(root,text="LR Algo",command=Lr,width="15",height="2",
              font="30",bg="white",fg="purple")
titleLabel=Label(root,text="Hello Ml",width="40", height="2",
                     font="40",bg="purple",fg="white")
lbl3=Label(root,text="Enter the test size",width="20", height="1",
                     font="40",bg="purple",fg="white")
lbl4=Label(root,text="Choose your Algorithm",width="20", height="1",
                     font="40",bg="purple",fg="white")
testValue=Entry(root)
lbl=Label(root, text = "",font="15", fg="black")
prec=Label(root, text = "",font="15", fg="black")
reca=Label(root, text = "",font="15", fg="black")
err=Label(root, text = "",font="15", fg="black")
rr=Label(root, text = "",font="15", fg="black")


f1=Frame(root)




titleLabel.pack(pady=8)
lbl2.pack(pady=5)
r1.pack()
r2.pack()
r3.pack()
lbl3.pack(pady=5)
testValue.pack()
lbl4.pack(pady=5)
DT.pack()
Svm.pack(pady=8)
RFe.pack()
knn.pack(pady=5)
Lr.pack()
lbl.pack()
prec.pack()
reca.pack() 
err.pack()
rr.pack()

root.mainloop()


"""/////////////////////////////////////////////////////////////////////"""

"""It works in four steps: 
 
Select random samples from a given dataset. 
Construct a decision tree for each sample and get a prediction result from each decision tree. 
Perform a vote for each predicted result. 
Select the prediction result with the most votes as the final prediction.""" 

""" Advantages: 
Random forests is considered as a highly accurate and robust method because of the number of 
decision trees participating in the process. 
It does not suffer from the overfitting problem. 
The main reason is that it takes the average of all the predictions, which cancels out the biases. 
The algorithm can be used in both classification and regression problems. 
Random forests can also handle missing values. There are two ways to handle these: 
using median values to replace continuous variables, and computing the proximity-weighted average of missing values. 
You can get the relative feature importance, which helps in selecting the most contributing features for the classifier. """

"""Disadvantages: 
Random forests is slow in generating predictions because it has multiple decision trees. 
Whenever it makes a prediction, all the trees in the forest have to make a prediction for the same given input 
and then perform voting on it. This whole process is time-consuming. 
The model is difficult to interpret compared to a decision tree, 
where you can easily make a decision by following the path in the tree."""

"""Random Forests vs Decision Trees 
Random forests is a set of multiple decision trees. 
Deep decision trees may suffer from overfitting, 
but random forests prevents overfitting by creating trees on random subsets. 
Decision trees are computationally faster. 
Random forests is difficult to interpret, 
while a decision tree is easily interpretable and can be converted to rules."""

"""/////////////////////////////////////////////////////////////////////"""


"""Knn 
depend on distance , eclidean distance between the new point and all rows , 
then give the smallest distance depend on k value

 Advantages :
 Lazy learner , easy to add data , easy to implement , no training.
 
 Disadvantages:
 not well with big data  / high dimension data , sensitive to noise and missing values.
 need normalization and scaling.
"""