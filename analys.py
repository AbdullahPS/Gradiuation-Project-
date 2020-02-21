from tkinter import *
# Importing the required packages
import PIL
import cv2
import pandas as pd
import easygui
from PIL import ImageTk as tk
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import pickle
import os
from tkinter.filedialog import askopenfilename

import maaiin
from ImageStone import ImageStone
from PIL import Image, ImageTk
import main

from random import randint

def cvToTkinter(image):

    imgtk = PIL.ImageTk.PhotoImage(file=image)


    return imgtk

def openimage():
    # Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    image_path = filename
    # img.src = filename
    # img.src = cv2.imread(image_path)




    # Rearrang the color channel

    return image_path


def importdata():
    balance_data = pd.read_csv(r'Preprocessed.csv', sep=',',
                               header=0)

    # Printing the dataswet shape
    print("\n**************************************************************************\n")

    print("Our Dataset Lenght:> ", len(balance_data))
    print("Our Dataset Shape:> ", balance_data.shape)

    # Printing the dataset obseravtions
    return balance_data


# Function to split the dataset
def splitdataset(balance_data):
    # Seperating the target variable
    X = balance_data.values[:, 0:24]
    Y = balance_data.values[:, -1]

    # print(X)

    # print(Y)

    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.35, random_state=1)

    return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                         max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: \n",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : \n",
          classification_report(y_test, y_pred))


def load_database():  # response function of button train dataset
    string = ''
    if (not (os.path.isfile("./gini_model.pkl")) and not (os.path.isfile("./entropy_model.pkl"))):
        data = importdata()
        X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
        clf_gini = train_using_gini(X_train, X_test, y_train)
        clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
        f = open("gini_model.pkl", "wb")
        f1 = open("entropy_model.pkl", "wb")
        pickle.dump(clf_gini, f)
        pickle.dump(clf_entropy, f1)
        # Prediction using gini
        # Operational Phase
        print("\n**************************************************************************\n")
        print("Results Using Gini Index:")
        y_pred_gini = prediction(X_test, clf_gini)
        cal_accuracy(y_test, y_pred_gini)
        print("\n**************************************************************************\n")
        print("Results Using Entropy:")
        # Prediction using entropy
        y_pred_entropy = prediction(X_test, clf_entropy)
        cal_accuracy(y_test, y_pred_entropy)
    else:
        with open("entropy_model.pkl", "rb") as f1:
            clf_entropy = pickle.load(f1)
        clf_entropy = pickle.load(open("entropy_model.pkl", "rb"))
        with open("gini_model.pkl", "rb") as f1:
            clf_gini = pickle.load(f1)
        clf_gini = pickle.load(open("gini_model.pkl", "rb"))
        messagebox.showinfo('Message title', 'The Database is already loaded you can enter new patient now ')

#
# def savepatient():  # function to save the patient in the variable new_patient when savebutton is clicked return the new patient  (response of button save and called for button predict
#     new_patient = [
#         [entry_age.get(), entry_bloodpressure.get(), entry_spec_grav.get(), entry_albumin.get(), entry_sugar.get(),
#          entry_red_blood.get(), entry_puscell.get(), entry_puscellcl.get(), entry_bacteria.get(),
#          entry_bloodglucose.get(), entry_bloodurea.get(), entry_creatinine.get(), entry_sodium.get(),
#          entry_potassium.get(), entry_hemoglobin.get(), entry_packedcell.get(), entry_wbcc.get(), entry_rbcc.get(),
#          entry_hypertensiom.get(), entry_diaMel.get(), entry_carArtDis.get(), entry_appet.get(), entry_pedEdem.get(),
#          entry_anemia.get()]]
#     print(new_patient)
#     return new_patient


# new_patient = [sc, sod, pot, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet,pe, ane]]
def predictpatient(gini,ent,savedpatient):  # function responf of button predict
    if ent.get():  # checks if gini checkbox is checked
        with open("entropy_model.pkl", "rb") as f1:  # imports entropy learn modeule
            clf_entropy = pickle.load(f1)
        try:
            qasem1 = prediction(savedpatient, clf_entropy)
            print("\n**************************************************************************\n")
            print("this patient based on entropy model have >> : " + " > >  " + qasem1)
            messagebox.showinfo("Succesfully entropy checked",
                                "this patient based on entropy model have >> : " + " > >  " + qasem1)  # popups result of check
        except Exception as e :
            print(e)
            easygui.msgbox("Fields Cannot be empty", title="Output")

    elif gini.get():  # checks if entropy checkbox is checked
        with open("gini_model.pkl", "rb") as f1:
            clf_gini = pickle.load(f1)

        try:
            qasem2 = prediction(savedpatient, clf_gini)
            print("this patient based on gini model have >> : " + " > >  " + qasem2)
            messagebox.showinfo("Succesfully gini checked",
                                "this patient based on gini model have >> : " + " > >  " + qasem2)
        except Exception as e:
            print(e)
            easygui.msgbox("Fields can't be emoty ", 'error')


# print("this patient based on gini model have >> : " + " > >" + qasem2)
if __name__ == "__analys__":
            #
            # # BEGINNING OF GUI
            # img = ImageStone()
            root = Tk()
            # root.title("Kidney Disease predictor")
            #
            #
            #
            # background_image=tk.PhotoImage(file='guidis.png')
            #
            # background_label = Label(root, image=background_image)
            # background_label.pack()
            # background_label.image = background_image
            # background_label.photo=background_image
            # background_label.image = background_image
            #
            # tabControl = ttk.Notebook(root)  # getting tag imports
            # tab1 = ttk.Frame(tabControl)  # first tag for kidney disease prediction
            # tabControl.add(tab1, text='Disease')
            #
            # tab2 = ttk.Frame(tabControl)  # second tag for kidney stone detection
            # tabControl.add(tab2, text='Stone')
            #
            # tabControl.pack(expand=1, fill='both')
            #
            # # frame for first layer
            # frame1 = Frame(tab1)
            # frame1.pack()
            # button_load_database = Button(frame1, text="Load database and train ", command=load_database)
            # button_load_database.grid(row=0)
            #
            # # frame for third layer
            # frame3 = Frame(tab1)
            # frame3.pack()
            #
            # label_age = Label(frame3, text="\n 1.enter the Age of the patient > > \n ")
            # entry_age = Entry(frame3, width=20)
            # label_age.grid(column=0, row=0)
            # entry_age.grid(column=1, row=0)
            #
            # label_bloodpressure = Label(frame3, text="\n 2.enter the blodd pressure of the patient > > \n ")
            # entry_bloodpressure = Entry(frame3, width=20)
            # label_bloodpressure.grid(column=0, row=1)
            # entry_bloodpressure.grid(column=1, row=1)
            #
            # label_spec_grav = Label(frame3, text="\n 3.enter the specific gravity  of the patient > > \n ")
            # entry_spec_grav = Entry(frame3, width=20)
            # label_spec_grav.grid(column=0, row=2)
            # entry_spec_grav.grid(column=1, row=2)
            #
            # label_almbumin = Label(frame3, text="\n 4.enter the albumin percentage of the patient > > \n ")
            # entry_albumin = Entry(frame3, width=20)
            # label_almbumin.grid(column=0, row=3)
            # entry_albumin.grid(column=1, row=3)
            #
            # label_sugar = Label(frame3, text="\n 5.enter the sugar percentage of the patient > > \n ")
            # entry_sugar = Entry(frame3, width=20)
            # label_sugar.grid(column=0, row=4)
            # entry_sugar.grid(column=1, row=4)
            #
            # label_red_blood = Label(frame3, text="\n 5.enter the red blood cells of the patient > > \n ")
            # entry_red_blood = Entry(frame3, width=20)
            # label_red_blood.grid(column=0, row=5)
            # entry_red_blood.grid(column=1, row=5)
            #
            # label_puscellcl = Label(frame3, text="\n 7.enter the pus cell clumps of the patient > > \n ")
            # entry_puscellcl = Entry(frame3, width=20)
            # label_puscellcl.grid(column=0, row=6)
            # entry_puscellcl.grid(column=1, row=6)
            #
            # label_bacteria = Label(frame3, text="\n 8/enter the bacteria of the patient > > \n ")
            # entry_bacteria = Entry(frame3, width=20)
            # label_bacteria.grid(column=0, row=7)
            # entry_bacteria.grid(column=1, row=7)
            #
            # label_bloodglucose = Label(frame3, text="\n 9.enter the blood glucose  of the patient > > \n ")
            # entry_bloodglucose = Entry(frame3, width=20)
            # label_bloodglucose.grid(column=0, row=8)
            # entry_bloodglucose.grid(column=1, row=8)
            #
            # label_bloodurea = Label(frame3, text="\n 10.enter the blood urea  of the patient > > \n ")
            # entry_bloodurea = Entry(frame3, width=20)
            # label_bloodurea.grid(column=0, row=9)
            # entry_bloodurea.grid(column=1, row=9)
            #
            # label_creatinine = Label(frame3, text="\n 11.enter the  creatinine of the patient > > \n ")
            # entry_creatinine = Entry(frame3, width=20)
            # label_creatinine.grid(column=0, row=10)
            # entry_creatinine.grid(column=1, row=10)
            #
            # label_sodium = Label(frame3, text="\n 12.enter the  sodium of the patient > > \n ")
            # entry_sodium = Entry(frame3, width=20)
            # label_sodium.grid(column=0, row=11)
            # entry_sodium.grid(column=1, row=11)
            #
            # label_potassium = Label(frame3, text="\n 13.enter the  potassium of the patient > > \n ")
            # entry_potassium = Entry(frame3, width=20)
            # label_potassium.grid(column=2, row=0)
            # entry_potassium.grid(column=3, row=0)
            #
            # label_hemoglobine = Label(frame3, text="\n 14.enter the  homoglobine of the patient > > \n ")
            # entry_hemoglobin = Entry(frame3, width=20)
            # label_hemoglobine.grid(column=2, row=1)
            # entry_hemoglobin.grid(column=3, row=1)
            #
            # label_packedcell = Label(frame3, text="\n 15. enter the packed cell of the patient > > \n ")
            # entry_packedcell = Entry(frame3, width=20)
            # label_packedcell.grid(column=2, row=2)
            # entry_packedcell.grid(column=3, row=2)
            #
            # label_wbcc = Label(frame3, text="\n 16.enter the  white blood cell count of the patient > > \n ")
            # entry_wbcc = Entry(frame3, width=20)
            # label_wbcc.grid(column=2, row=3)
            # entry_wbcc.grid(column=3, row=3)
            #
            # label_rbcc = Label(frame3, text="\n 17.enter the  red blood cell count of the patient > > \n ")
            # entry_rbcc = Entry(frame3, width=20)
            # label_rbcc.grid(column=2, row=4)
            # entry_rbcc.grid(column=3, row=4)
            #
            # label_hypertensiom = Label(frame3, text="\n 18. enter the  hypertension count of the patient > > \n ")
            # entry_hypertensiom = Entry(frame3, width=20)
            # label_hypertensiom.grid(column=2, row=5)
            # entry_hypertensiom.grid(column=3, row=5)
            #
            # label_diaMel = Label(frame3, text="\n 19.enter Diabetes mellitus count of the patient > > \n ")
            # entry_diaMel = Entry(frame3, width=20)
            # label_diaMel.grid(column=2, row=6)
            # entry_diaMel.grid(column=3, row=6)
            #
            # label_carArtDis = Label(frame3, text="\n 20.enter the coronary artery disease of the patient > > \n  ")
            # entry_carArtDis = Entry(frame3, width=20)
            # label_carArtDis.grid(column=2, row=7)
            # entry_carArtDis.grid(column=3, row=7)
            #
            # label_appet = Label(frame3, text="\n 21.enter the Appet of the patient > > \n  ")
            # entry_appet = Entry(frame3, width=20)
            # label_appet.grid(column=2, row=8)
            # entry_appet.grid(column=3, row=8)
            #
            # label_pedEdem = Label(frame3, text="\n  22.enter the pedal edema of the patient > > \n  ")
            # entry_pedEdem = Entry(frame3, width=20)
            # label_pedEdem.grid(column=2, row=9)
            # entry_pedEdem.grid(column=3, row=9)
            #
            # label_anemia = Label(frame3, text="\n  23.enter the anemia of the patient > > \n  ")
            # entry_anemia = Entry(frame3, width=20)
            # label_anemia.grid(column=2, row=10)
            # entry_anemia.grid(column=3, row=10)
            #
            # label_puscell = Label(frame3, text="\n 24. enter the pus cell of the patient > > \n ")
            # entry_puscell = Entry(frame3, width=20)
            # label_puscell.grid(column=2, row=11)
            # entry_puscell.grid(column=3, row=11)
            #
            # Button_new_patient = Button(frame3, text='save data', command=savepatient)
            # Button_new_patient.grid(row=12)
            #
            # # frame for second layer
            # frame2 = Frame(tab1)
            # frame2.pack()
            # chk_state_gini = BooleanVar()
            # chk_state_entropy = BooleanVar()
            #
            # chk_gini = Checkbutton(frame2, text='Geni-Index', var=chk_state_gini)
            # chk_entropy = Checkbutton(frame2, text='Entropy', var=chk_state_entropy)
            #
            # button_train = Button(frame2, text="Predict", command=predictpatient)
            # chk_entropy.grid(column=0, row=0)
            # chk_gini.grid(column=1, row=0)
            # button_train.grid(row=1)
            #
            # stoneButFrame = Frame(tab2)
            # stoneButFrame.pack()
            # canvas = Canvas(stoneButFrame,width =704, height= 492)
            #
            # # image_path=main.openimage()
            # # image = cv2.imread(image_path)
            #
            # buttonStoneImgLoad = Button(stoneButFrame, text="Load Stone Image", command=main.openimage)
            # buttonStoneImgLoad.grid(column=0,row=0)
            # image= tk.PhotoImage(file='treshed.jpg')
            # canvas.create_image(0, 0, anchor=NW, image=image)
            # canvas.grid(column=1,row=0)
            #
            # buttonStoneImgLoad = Button(stoneButFrame, text="threshold image", command=main.thresholding)
            # buttonStoneImgLoad.grid(column=0,row=1)
            # buttonStoneImgLoad = Button(stoneButFrame, text="Fill holes", command=main.fillholes)
            # buttonStoneImgLoad.grid(column=0,row=2)
            # buttonStoneImgLoad = Button(stoneButFrame, text="clean", command=main.removetext)
            # buttonStoneImgLoad.grid(column=0,row=3)
            # buttonStoneImgLoad = Button(stoneButFrame, text="adjust", command=main.adjust)
            # buttonStoneImgLoad.grid(column=0,row=4)
            # buttonStoneImgLoad = Button(stoneButFrame, text="Load Stone Image", command=main.showfigureGUI)
            # buttonStoneImgLoad.grid(column=0,row=5)
            # image= tk.PhotoImage(file='cleaned.jpg')
            # canvas.create_image(0, 0, anchor=NW, image=image)
            #
            #
            # buttonStoneImgLoad = Button(stoneButFrame, text="check", command=main.checkstone)
            # buttonStoneImgLoad.grid(column=0,row=6)
            # buttonStoneImgLoad = Button(stoneButFrame, text="check", command=main.makeall)
            # buttonStoneImgLoad.grid(column=1,row=6)
            # # cv2.imshow('hi',img.src)
            # # cv2.waitKey(0)
            # # cv2.imshow('hi',buttonStoneImgLoad)
            #
            #
            # root.mainloop()
            # # Function importing Dataset
