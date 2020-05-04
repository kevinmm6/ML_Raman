# 10/03/20
# Kevin Marroquin
# Code using two different machine learning models (Decision Tree and Random 
# Forest) on Raman data for pulp.
# %% Libraries setup
import pandas as pd # Read data files (.csv or .xlsx)
#import tkinter as tk #Filedialog window
import sys #sys.exit() to programatically stop code 
from sklearn.model_selection import train_test_split #Split data into training and validation 
from sklearn.tree import DecisionTreeRegressor #DT model
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.ensemble import RandomForestRegressor #RF model
import matplotlib.pyplot as plt #Plots
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
for ii in spectra_number :
    if ii<100:
        spectra_interest.append('Spectrum_0'+ str(ii))
    else:
        spectra_interest.append('Spectrum_'+ str(ii))   
total_data = data[spectra_interest]

# Target data has 760 values in comparison to the 699 values of the spectra
# data, so we select target data with that same range
total_datalen = len(total_data)
tear = target.Tear[0:total_datalen] 
total_data.insert(total_data.shape[1],'Tear',tear) #Insert target at the end of data 

# Filter rows with missing target data
filter_data = total_data.dropna(axis=0)
#%% Set features and target data and split them into training and validation data
# Features
X = filter_data[spectra_interest]
# Target
y = filter_data.Tear
# Split data
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 1)
#%% Model selection (Decision Tree), fitting and MAE calculation
raman_dtmodel = DecisionTreeRegressor(random_state=1) #Model selection 
raman_dtmodel.fit(train_X,train_y) #Training
val_dtpredictions = raman_dtmodel.predict(val_X) #Predictions
val_dtmae = mean_absolute_error(val_dtpredictions,val_y) #MAE
print('MAE for Decision Tree validation: {}'.format(val_dtmae))
# %% Optimization 
dt_optimization_leafs = [2,4,5,8,9,10,12] #Number of optimization tree leafs to try 
dtmae = []
n = 0
for ii in dt_optimization_leafs:
    raman_dtmodel = DecisionTreeRegressor(max_leaf_nodes = ii, random_state=1) #Model
    raman_dtmodel.fit(train_X,train_y) #Training
    dtpredictions = raman_dtmodel.predict(val_X) #Predictions
    dtmae.append(mean_absolute_error(dtpredictions,val_y)) #MAE
    print('MAE for Decision Tree with {} nodes: {}'.format(dt_optimization_leafs[n],dtmae[n]))
    n+=1
dtmae_min_idx = dtmae.index(min(dtmae))
dt_optimum_leafs = dt_optimization_leafs[dtmae_min_idx]
print("Optimum number of leafs for Decision Tree model: {}".format(dt_optimum_leafs))
#%% Plotting
plt.figure(1)
plt.plot(dt_optimization_leafs,dtmae,'.') 
plt.xlabel('Number of leafs')
plt.ylabel('Mean Absolute Error')
plt.title('Decision Tree Model')
#%% Random Forest model, fitting and MAE calculation
raman_rfmodel = RandomForestRegressor(random_state=1) #Model selection
raman_rfmodel.fit(train_X,train_y) #Training
val_rfpredictions = raman_rfmodel.predict(val_X) #Predictions
val_rfmae = mean_absolute_error(val_rfpredictions,val_y)
print('MAE for Random Forest validation: {}'.format(val_rfmae))
#%% Random Forest Optimization 
rf_optimization_leafs = [2,5,10,20,22,25,28,30]
rfmae = [] 
n = 0
for ii in rf_optimization_leafs:
    raman_rfmodel = RandomForestRegressor(max_leaf_nodes = ii, random_state=1)
    raman_rfmodel.fit(train_X,train_y)
    rfpredictions = raman_rfmodel.predict(val_X)
    rfmae.append(mean_absolute_error(rfpredictions,val_y))
    print('MAE for Random Forest with {} nodes: {}'.format(rf_optimization_leafs[n],rfmae[n]))
    n+=1
rfmae_min_idx = rfmae.index(min(rfmae))
rf_optimum_leafs = rf_optimization_leafs[rfmae_min_idx]
print('Optimum number of leafs for Random Forest model: {}'.format(rf_optimum_leafs))
#%% Plotting
plt.figure(2)
plt.plot(rf_optimization_leafs,rfmae,'.')
plt.xlabel('Number of leafs')
plt.ylabel('Mean Absolute Error')
plt.title('Random Forest Model')

   






