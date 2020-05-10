# 10/03/20
# Kevin Marroquin
# Code using two different machine learning models (Decision Tree and Random 
# Forest) on Raman data for pulp.
# %% Libraries setup
import pandas as pd # Read data files (.csv or .xlsx)
#import tkinter as tk #Filedialog window
#import sys #sys.exit() to programatically stop code 
import numpy as np #Arrays
from sklearn.model_selection import train_test_split #Split data into training and validation 
from sklearn.tree import DecisionTreeRegressor #DT model
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.ensemble import RandomForestRegressor #RF model
import matplotlib.pyplot as plt #Plots
# %% Functions setup 
# Decision Tree
def decision_tree(X,target_data,ii,target_name,dt_optimization_leafs,y):
    y = np.array(target_data.iloc[:,ii])
    #Split data
    train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 1)
    # Model selection, fitting and MAE calculation 
    raman_dtmodel = DecisionTreeRegressor(random_state=1) #Model selection 
    raman_dtmodel.fit(train_X,train_y) #Training
    val_dtpredictions = raman_dtmodel.predict(val_X) #Predictions
    val_dtmae = mean_absolute_error(val_dtpredictions,val_y) #MAE
    print('MAE for Decision Tree validation of {}: {}'.format(target_name,val_dtmae))
    # Optimization
    n = 0 
    dtmae = []
    for jj in dt_optimization_leafs:
        raman_dtmodel = DecisionTreeRegressor(max_leaf_nodes = jj, random_state=1) #Model
        raman_dtmodel.fit(train_X,train_y) #Training
        dtpredictions = raman_dtmodel.predict(val_X) #Predictions
        dtmae.append(mean_absolute_error(dtpredictions,val_y)) #MAE
        print('MAE for Decision Tree of {} with {} nodes: {}'.format(target_name,dt_optimization_leafs[n],dtmae[n]))
        n+=1
    dtmae_min_idx = dtmae.index(min(dtmae))
    dt_optimum_leafs = dt_optimization_leafs[dtmae_min_idx]
    print("Optimum number of leafs for Decision Tree model of {}: {}".format(target_name,dt_optimum_leafs))
    # Plotting 
    plt.figure(ii+1)
    plt.plot(dt_optimization_leafs,dtmae,'.') 
    plt.xlabel('Number of leafs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Decision Tree Model for {}'.format(target_name))
    
# Random Forest 
def random_forest(X,target_data,ii,target_name,rf_optimization_leafs,y):
    # Split data
    train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 1)
    raman_rfmodel = RandomForestRegressor(random_state=1) #Model selection
    raman_rfmodel.fit(train_X,train_y) #Training
    val_rfpredictions = raman_rfmodel.predict(val_X) #Predictions
    val_rfmae = mean_absolute_error(val_rfpredictions,val_y)
    print('MAE for Random Forest validation of {}: {}'.format(target_name,val_rfmae))
    # Optimization
    rfmae = [] 
    n = 0   
    for jj in rf_optimization_leafs:
        raman_rfmodel = RandomForestRegressor(max_leaf_nodes = jj, random_state=1)
        raman_rfmodel.fit(train_X,train_y)
        rfpredictions = raman_rfmodel.predict(val_X)
        rfmae.append(mean_absolute_error(rfpredictions,val_y))
        print('MAE for Random Forest of {} with {} nodes: {}'.format(target_name,rf_optimization_leafs[n],rfmae[n]))
        n+=1
    rfmae_min_idx = rfmae.index(min(rfmae))
    rf_optimum_leafs = rf_optimization_leafs[rfmae_min_idx]
    print('Optimum number of leafs for Random Forest model: {}'.format(rf_optimum_leafs))
    # Plotting
    plt.figure(ii+1+target_data.shape[1])
    plt.plot(rf_optimization_leafs,rfmae,'.')
    plt.xlabel('Number of leafs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Random Forest Model for {}'.format(target_name))
# %% Load and read data (either programatically or manually)
# Select data manually from filedialog pop-up window  
    # root = tk.Tk()
    # root.withdraw() #Prevent root window from appearing 
    # file = tk.filedialog.askopenfilename() #Invoke dialog window to open file
 
# Load file data programatically
data_filepath = 'C:/Kevin/Python/ML_Raman/Neural Network data/Raman20200218.csv'
data = pd.read_csv(data_filepath) #Read data in Raman file
target_filepath = 'C:/Kevin/Python/ML_Raman/Neural Network data/Targets20200218.csv'
target = pd.read_csv(target_filepath)  #Read data in target file
# %% Select data  of interest
# Create list with features of interest (Spectra_049 to Spectra_1024 since first 50 pixels empty)
spectra_number = range(49,1025)
spectra_interest = []
for ii in spectra_number:
    if ii<100:
        spectra_interest.append('Spectrum_0'+ str(ii))
    else:
        spectra_interest.append('Spectrum_'+ str(ii))   
total_data = data[spectra_interest]

# Target data has 760 values in comparison to the 699 values of the spectra
# data, so we select target data with that same range
total_datalen = len(total_data)
tear = target.Tear[0:total_datalen] 
BL = target.BL[0:total_datalen]
burst = target.Burst[0:total_datalen]
#Insert target at the end of data 
total_data.insert(total_data.shape[1],'Tear',tear) 
total_data.insert(total_data.shape[1],'BL',BL)
total_data.insert(total_data.shape[1],'burst',burst)
# Filter rows with missing target data
filter_data = total_data.dropna(axis=0)
#%% Set features and target data
# Features
X = filter_data[spectra_interest]
# Target
target_data = filter_data[['Tear','BL','burst']]
#%% Machine learning models 
dt_optimization_leafs = [2,4,6,8,10,12,14,16,18] #Number of optimization tree leafs to try in dt model
rf_optimization_leafs = [20,22,24,28,30,32,34] #Number of optimization tree leafs to try in rf model
for ii, target_name in enumerate(target_data):
    y = np.array(target_data.iloc[:,ii])
    decision_tree(X,target_data,ii,target_name,dt_optimization_leafs,y)
    random_forest(X,target_data,ii,target_name,rf_optimization_leafs,y)
   






