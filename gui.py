
from PIL import ImageTk as tk
from tkinter import *
from tkinter.ttk import *
from tkinter import ttk, messagebox
import analys
import main

class GUI:
    def __init__(self, master):

        self.master = master
        master.title("Kidney Disease predictor")

        self.chk_state_gini = BooleanVar()
        self.chk_state_entropy = BooleanVar()
        self.background_image=tk.PhotoImage(file='guidis.png')
        # self.background_label = Label(master, image=self.background_image)
        # self.background_label.pack()
        # self.background_label.image = self.background_image
        # self.background_label.photo=self.background_image
        # self.background_label.image = self.background_image

        self.tabControl = ttk.Notebook(master)  # getting tag imports
        self.tab1 = ttk.Frame(self.tabControl)  # first tag for kidney disease prediction
        self.tabControl.add(self.tab1, text='Disease')

        self.tab2 = ttk.Frame(self.tabControl)  # second tag for kidney stone detection
        self.tabControl.add(self.tab2, text='Stone')

        self.tabControl.pack(expand=1, fill='both')

        # frame for first layer
        self.frame1 = Frame(self.tab1)
        self.frame1.pack()
        self.button_load_database = Button(self.frame1, text="Load database and train ", command=analys.load_database)
        self.button_load_database.grid(row=0)

        # frame for third layer
        self.frame3 = Frame(self.tab1)
        self.frame3.pack()

        self.label_age = Label(self.frame3, text="\n 1.Enter the Age of the patient > > \n ",style = 'TLabel')
        self.entry_age = Entry(self.frame3, width=20)
        self.label_age.grid(column=0, row=0)
        self.entry_age.grid(column=1, row=0)

        self.label_bloodpressure = Label(self.frame3, text="\n 2.Enter the blood pressure of the patient > > \n ")
        self.entry_bloodpressure = Entry(self.frame3, width=20)
        self.label_bloodpressure.grid(column=0, row=1)
        self.entry_bloodpressure.grid(column=1, row=1)

        self.label_spec_grav = Label(self.frame3, text="\n 3.Enter the specific gravity  of the patient > > \n ")
        self.entry_spec_grav = Entry(self.frame3, width=20)
        self.label_spec_grav.grid(column=0, row=2)
        self.entry_spec_grav.grid(column=1, row=2)

        self.label_almbumin = Label(self.frame3, text="\n 4.Enter the albumin percentage of the patient > > \n ")
        self.entry_albumin = Entry(self.frame3, width=20)
        self.label_almbumin.grid(column=0, row=3)
        self.entry_albumin.grid(column=1, row=3)

        self.label_sugar = Label(self.frame3, text="\n 5.Enter the sugar percentage of the patient > > \n ")
        self.entry_sugar = Entry(self.frame3, width=20)
        self.label_sugar.grid(column=0, row=4)
        self.entry_sugar.grid(column=1, row=4)

        self.label_red_blood = Label(self.frame3, text="\n 6.Enter the red blood cells of the patient > > \n ")
        self.entry_red_blood = Entry(self.frame3, width=20)
        self.label_red_blood.grid(column=0, row=5)
        self.entry_red_blood.grid(column=1, row=5)

        self.label_puscellcl = Label(self.frame3, text="\n 7.Enter the pus cell clumps of the patient > > \n ")
        self.entry_puscellcl = Entry(self.frame3, width=20)
        self.label_puscellcl.grid(column=0, row=6)
        self.entry_puscellcl.grid(column=1, row=6)

        self.label_bacteria = Label(self.frame3, text="\n 8.Enter the bacteria of the patient > > \n ")
        self.entry_bacteria = Entry(self.frame3, width=20)
        self.label_bacteria.grid(column=0, row=7)
        self.entry_bacteria.grid(column=1, row=7)

        self.label_bloodglucose = Label(self.frame3, text="\n 9.Enter the blood glucose  of the patient > > \n ")
        self.entry_bloodglucose = Entry(self.frame3, width=20)
        self.label_bloodglucose.grid(column=0, row=8)
        self.entry_bloodglucose.grid(column=1, row=8)

        self.label_bloodurea = Label(self.frame3, text="\n 10.Enter the blood urea  of the patient > > \n ")
        self.entry_bloodurea = Entry(self.frame3, width=20)
        self.label_bloodurea.grid(column=0, row=9)
        self.entry_bloodurea.grid(column=1, row=9)

        self.label_creatinine = Label(self.frame3, text="\n 11.Enter the  creatinine of the patient > > \n ")
        self.entry_creatinine = Entry(self.frame3, width=20)
        self.label_creatinine.grid(column=0, row=10)
        self.entry_creatinine.grid(column=1, row=10)

        self.label_sodium = Label(self.frame3, text="\n 12.Enter the  sodium of the patient > > \n ")
        self.entry_sodium = Entry(self.frame3, width=20)
        self.label_sodium.grid(column=0, row=11)
        self.entry_sodium.grid(column=1, row=11)

        self.label_potassium = Label(self.frame3, text="\n 13.Enter the  potassium of the patient > > \n ")
        self.entry_potassium = Entry(self.frame3, width=20)
        self.label_potassium.grid(column=2, row=0)
        self.entry_potassium.grid(column=3, row=0)

        self.label_hemoglobine = Label(self.frame3, text="\n 14.Enter the  homoglobine of the patient > > \n ")
        self.entry_hemoglobin = Entry(self.frame3, width=20)
        self.label_hemoglobine.grid(column=2, row=1)
        self.entry_hemoglobin.grid(column=3, row=1)

        self.label_packedcell = Label(self.frame3, text="\n 15. Enter the packed cell of the patient > > \n ")
        self.entry_packedcell = Entry(self.frame3, width=20)
        self.label_packedcell.grid(column=2, row=2)
        self.entry_packedcell.grid(column=3, row=2)
        frame3=self.frame3
        self.label_wbcc = Label(frame3, text="\n 16.Enter the  white blood cell count of the patient > > \n ")
        self.entry_wbcc = Entry(frame3, width=20)
        self.label_wbcc.grid(column=2, row=3)
        self.entry_wbcc.grid(column=3, row=3)

        self.label_rbcc = Label(frame3, text="\n 17.Enter the  red blood cell count of the patient > > \n ")
        self.entry_rbcc = Entry(frame3, width=20)
        self.label_rbcc.grid(column=2, row=4)
        self.entry_rbcc.grid(column=3, row=4)

        self.label_hypertensiom = Label(frame3, text="\n 18. Enter the  hypertension count of the patient > > \n ")
        self.entry_hypertensiom = Entry(frame3, width=20)
        self.label_hypertensiom.grid(column=2, row=5)
        self.entry_hypertensiom.grid(column=3, row=5)

        self.label_diaMel = Label(frame3, text="\n 19.Enter Diabetes mellitus count of the patient > > \n ")
        self.entry_diaMel = Entry(frame3, width=20)
        self.label_diaMel.grid(column=2, row=6)
        self.entry_diaMel.grid(column=3, row=6)

        self.label_carArtDis = Label(frame3, text="\n 20.Enter the coronary artery disease of the patient > > \n  ")
        self.entry_carArtDis = Entry(frame3, width=20)
        self.label_carArtDis.grid(column=2, row=7)
        self.entry_carArtDis.grid(column=3, row=7)

        self.label_appet = Label(frame3, text="\n 21.Enter the Appet of the patient > > \n  ")
        self.entry_appet = Entry(frame3, width=20)
        self.label_appet.grid(column=2, row=8)
        self.entry_appet.grid(column=3, row=8)

        self.label_pedEdem = Label(frame3, text="\n  22.Enter the pedal edema of the patient > > \n  ")
        self.entry_pedEdem = Entry(frame3, width=20)
        self.label_pedEdem.grid(column=2, row=9)
        self.entry_pedEdem.grid(column=3, row=9)

        self.label_anemia = Label(frame3, text="\n  23.Enter the anemia of the patient > > \n  ")
        self.entry_anemia = Entry(frame3, width=20)
        self.label_anemia.grid(column=2, row=10)
        self.entry_anemia.grid(column=3, row=10)

        self.label_puscell = Label(frame3, text="\n 24. Enter the pus cell of the patient > > \n ")
        self.entry_puscell = Entry(frame3, width=20)
        self.label_puscell.grid(column=2, row=11)
        self.entry_puscell.grid(column=3, row=11)

        self.Button_new_patient = Button(frame3, text='Save data', command=self.savepatient)
        self.Button_new_patient.grid(row=12)

        # frame for second layer
        self.frame2 = Frame(self.tab1)
        self.frame2.pack()
        self.chk_state_gini = BooleanVar()
        self.chk_state_entropy = BooleanVar()

        self.chk_gini = Checkbutton(self.frame2, text='Geni-Index', var=self.chk_state_gini)
        self.chk_entropy = Checkbutton(self.frame2, text='Entropy', var=self.chk_state_entropy)

        self.button_train = Button(self.frame3, text="Predict", command=lambda: analys.predictpatient(self.chk_state_gini,self.chk_state_entropy,self.savepatient()))
        self.chk_entropy.grid(column=0, row=0)
        self.chk_gini.grid(column=1, row=0)
        self.button_train.grid(column=2,row=12)

        self.stoneButFrame = Frame(self.tab2)
        self.stoneButFrame.pack()
        self.canvas = Canvas(self.stoneButFrame,width =704, height= 492)


        # image_path=main.openimage()
        # image = cv2.imread(image_path)

        self.buttonStoneImgLoad = Button(self.stoneButFrame, text="Load Stone Image", command=lambda:[main.openimage(),self.changeimage('original.jpg')])
        self.canvas.grid(column=0,row=0)

        style = ttk.Style()
        style.configure('TButton', font=
        ('calibri', 20, 'bold'),
                        borderwidth='4')
        style.configure('TLabel', font=
        ('calibri', 10, 'bold'),
                        borderwidth='5')
        ttk.Style().configure('TLabelframe', font=
        "15", foreground="red")

        # Changes will be reflected
        # by the movement of mouse.
        # style.map('TButton', foreground=[('active', '! disabled', 'green')],background=[('active', 'black')])



        self.buttonStoneImgLoad.grid(column=0,row=1)
        self.buttonStoneImgLoad = Button(self.stoneButFrame, text=" threshold image ",style = 'TButton',  command=lambda:[main.thresholding(),self.changeimage('treshed.jpg')])
        self.buttonStoneImgLoad.grid(column=0,row=3)
        self.buttonStoneImgLoad = Button(self.stoneButFrame, text="   Fill  holes   ",  command=lambda:[main.fillholes(),self.changeimage('holesFilled.jpg')])
        self.buttonStoneImgLoad.grid(column=0,row=5)
        self.buttonStoneImgLoad = Button(self.stoneButFrame, text="      Clean      ",  command=lambda:[main.removetext(),self.changeimage('cleaned.jpg')])
        self.buttonStoneImgLoad.grid(column=1,row=1)
        self.buttonStoneImgLoad = Button(self.stoneButFrame, text="      Adjust      ", command=lambda:[main.adjust(),self.changeimage('adjusted2.jpg')])
        self.buttonStoneImgLoad.grid(column=1,row=3)
        self.buttonStoneImgLoad = Button(self.stoneButFrame, text="Show All Togeteher", command=main.showfigureGUI)
        self.buttonStoneImgLoad.grid(column=2,row=1)
        self.label_tvalue = Label(self.tab2, text="\n Enter threshed value \n ")
        self.entry_tvalue = Entry(self.tab2, width=20)

        self.label_tvalue.pack()
        self.entry_tvalue.pack()



        self.buttonStoneImgLoad = Button(self.stoneButFrame, text="Check for stone", command=lambda : [main.checkstone(self.entry_tvalue.get())])
        self.buttonStoneImgLoad.grid(column=1,row=5)
        self.buttonStoneMakeAll = Button(self.stoneButFrame, text="Perform check only",command=lambda : [main.makeall(self.entry_tvalue.get())])
        self.buttonStoneMakeAll.grid(column=2,row=3)
        self.buttonStoneOnebOne = Button(self.stoneButFrame, text="Show Stones One by One ", command=main.showstones1by1)
        self.buttonStoneOnebOne.grid(column=2, row=5)

        self.image = tk.PhotoImage(file='emptyBack.png')
        self.canvas.create_image(0, 0, anchor=NW, image=self.image)
        self.canvas.grid(column=1, row=0)



    def savepatient(self):  # function to save the patient in the variable new_patient when savebutton is clicked return the new patient  (response of button save and called for button predict
            new_patient = [
                [self.entry_age.get(), self.entry_bloodpressure.get(), self.entry_spec_grav.get(), self.entry_albumin.get(),
                 self.entry_sugar.get(),
                 self.entry_red_blood.get(), self.entry_puscell.get(), self.entry_puscellcl.get(), self.entry_bacteria.get(),
                 self.entry_bloodglucose.get(), self.entry_bloodurea.get(), self.entry_creatinine.get(), self.entry_sodium.get(),
                 self.entry_potassium.get(), self.entry_hemoglobin.get(), self.entry_packedcell.get(), self.entry_wbcc.get(),
                 self.entry_rbcc.get(),
                 self.entry_hypertensiom.get(), self.entry_diaMel.get(), self.entry_carArtDis.get(), self.entry_appet.get(),
                 self.entry_pedEdem.get(),
                 self.entry_anemia.get()]]
            print(new_patient)
            return new_patient
    def changeimage(self,filename):
        self.image = tk.PhotoImage(file=filename)
        self.canvas.create_image(0, 0, anchor=NW, image=self.image)
        self.canvas.grid(column=1, row=0)





