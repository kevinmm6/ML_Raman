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
def decision_tree(X,target_data,ii,target_name,dt_optimization_leafs,y,mae_dt):
    #Split data
    train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 1)
    # Model selection, fitting and MAE calculation 
    raman_dtmodel = DecisionTreeRegressor(random_state=1) #Model selection 
    raman_dtmodel.fit(train_X,train_y) #Training
    val_dtpredictions = raman_dtmodel.predict(val_X) #Predictions
    val_dtmae = mean_absolute_error(val_dtpredictions,val_y) #MAE
    print('MAE for Decision Tree validation of {}: {}'.format(target_name,val_dtmae))
    # Optimization
    dtmae = []
    for jj in dt_optimization_leafs:
        raman_dtmodel = DecisionTreeRegressor(max_leaf_nodes = jj, random_state=1) #Model
        raman_dtmodel.fit(train_X,train_y) #Training
        dtpredictions = raman_dtmodel.predict(val_X) #Predictions
        dtmae.append(mean_absolute_error(dtpredictions,val_y)) #MAE
    dtmae_min_idx = dtmae.index(min(dtmae))
    if target_name == 'Tear':
        mae_dt['Tear'] = min(dtmae)
        mae_dt['Tear_nodes'] = dt_optimization_leafs[dtmae_min_idx]
    elif target_name == 'BL':
        mae_dt['BL'] = min(dtmae)
        mae_dt['BL_nodes'] = dt_optimization_leafs[dtmae_min_idx]
    else:
        mae_dt['Burst']= min(dtmae)
        mae_dt['Burst_nodes'] = dt_optimization_leafs[dtmae_min_idx]
    # # Plotting 
    # plt.figure(ii+1)
    # plt.plot(dt_optimization_leafs,dtmae,'.') 
    # plt.xlabel('Number of leafs')
    # plt.ylabel('Mean Absolute Error')
    # plt.title('Decision Tree Model for {}'.format(target_name))
    
# Random Forest 
def random_forest(X,target_data,ii,target_name,rf_optimization_leafs,y,mae_rf):
    # Split data
    train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 1)
    raman_rfmodel = RandomForestRegressor(random_state=1) #Model selection
    raman_rfmodel.fit(train_X,train_y) #Training
    val_rfpredictions = raman_rfmodel.predict(val_X) #Predictions
    val_rfmae = mean_absolute_error(val_rfpredictions,val_y)
    print('MAE for Random Forest validation of {}: {}'.format(target_name,val_rfmae))
    # Optimization
    rfmae = []   
    for jj in rf_optimization_leafs:
        raman_rfmodel = RandomForestRegressor(max_leaf_nodes = jj, random_state=1)
        raman_rfmodel.fit(train_X,train_y)
        rfpredictions = raman_rfmodel.predict(val_X)
        rfmae.append(mean_absolute_error(rfpredictions,val_y))
    rfmae_min_idx = rfmae.index(min(rfmae))
    if target_name == 'Tear':
        mae_rf['Tear'] = min(rfmae)
        mae_rf['Tear_nodes'] = rf_optimization_leafs[rfmae_min_idx]
    elif target_name == 'BL':
        mae_rf['BL'] = min(rfmae)
        mae_rf['BL_nodes'] = rf_optimization_leafs[rfmae_min_idx]
    else:
        mae_rf['Burst']= min(rfmae)
        mae_rf['Burst_nodes'] = rf_optimization_leafs[rfmae_min_idx]
    # # Plotting
    # plt.figure(ii+1+target_data.shape[1])
    # plt.plot(rf_optimization_leafs,rfmae,'.')
    # plt.xlabel('Number of leafs')
    # plt.ylabel('Mean Absolute Error')
    # plt.title('Random Forest Model for {}'.format(target_name))

# Comparison
def comparison(target_name,mae_dt,mae_rf):
    if target_name == 'BL':
        if mae_dt['BL'] < mae_rf['BL']:
            bestmodel = 'Decision Tree'
            mae = mae_dt['BL']
            nodes = mae_dt['BL_nodes']
        else:
            bestmodel = 'Random Forest'
            mae = mae_rf['BL']
            nodes = mae_rf['BL_nodes']
        print('Best model for BL is {} with {} nodes and a MAE of {}'.format(bestmodel,nodes,mae))
    elif target_name == 'Tear':
        if mae_dt['Tear'] < mae_rf['Tear']:
            bestmodel = 'Decision Tree'
            mae = mae_dt['Tear']
            nodes = mae_dt['Tear_nodes']
        else:
            bestmodel = 'Random Forest'
            mae = mae_rf['Tear']
            nodes = mae_rf['Tear_nodes']
        print('Best model for Tear is {} with {} nodes and a MAE of {}'.format(bestmodel,nodes,mae))
    else:
        if mae_dt['Burst'] < mae_rf['Burst']:
            bestmodel = 'Decision Tree'
            mae = mae_dt['Burst']
            nodes = mae_dt['Burst_nodes']
        else:
            bestmodel = 'Random Forest'
            mae = mae_rf['Burst']
            nodes = mae_rf['Burst_nodes']
        print('Best model for Burst is {} with {} nodes and a MAE of {}'.format(bestmodel,nodes,mae))
# %% Load and read data (either programatically or manually)
# Select data manually from filedialog pop-up window  
    # root = tk.Tk()
    # root.withdraw() #Prevent root window from appearing 
    # file = tk.filedialog.askopenfilename() #Invoke dialog window to open file
 
# Load file data programatically
data_filepath = 'C:/Kevin/Python/ML_Raman/Neural Network data/Raman 20200218.csv'
data = pd.read_csv(data_filepath) #Read data in Raman file
target_filepath = 'C:/Kevin/Python/ML_Raman/Neural Network data/Targets 20200218.csv'
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
mae_dt = {'Tear': 0,'Burst': 0,'BL': 0,'Tear_nodes' : 0,'BL_nodes' : 0,'Burst_nodes' : 0}
mae_rf = {'Tear': 0,'Burst': 0,'BL': 0,'Tear_nodes' : 0,'BL_nodes' : 0,'Burst_nodes' : 0}
for ii, target_name in enumerate(target_data):
    y = np.array(target_data.iloc[:,ii]) #Target data
    decision_tree(X,target_data,ii,target_name,dt_optimization_leafs,y,mae_dt)
    random_forest(X,target_data,ii,target_name,rf_optimization_leafs,y,mae_rf)
    comparison(target_name,mae_dt,mae_rf)
    





