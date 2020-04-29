import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, Embedding
from keras.layers import Flatten,GlobalMaxPooling1D, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

warnings.filterwarnings('ignore')

# Features needed
feats = ['DateTime','GG_SPEED_NGG_KRPM',
         'PT_INLET_TEMP_T54', 'GG_COMP_DIS_PRES_CDP',
         'LO_SPLY_PRES','FO_MFLD_PRES_FMP',
         'GG_@_NGG_VIBS','PT_@_NGG_VIBS','LO_SCAV_PRES',
         'FMV/A_POSITION'  ]

var = ['GG_SPEED_NGG_KRPM',
         'PT_INLET_TEMP_T54', 'GG_COMP_DIS_PRES_CDP',
         'LO_SPLY_PRES','FO_MFLD_PRES_FMP',
         'GG_@_NGG_VIBS','PT_@_NGG_VIBS','LO_SCAV_PRES',
         'FMV/A_POSITION' ]

def data_loader():
    dataframes = []
    path = "data/*.csv"
    for fname in glob.glob(path):
        df = pd.read_csv(fname)
        dataframes.append(df)

    for df in dataframes:
        df.columns = df.columns.str.replace('1A ', '')
        df.columns = df.columns.str.replace('1B ', '')
        df.columns = df.columns.str.replace('2A ', '')
        df.columns = df.columns.str.replace('2B ', '')
        df.columns = df.columns.str.replace('ER1 ', '')
        df.columns = df.columns.str.replace('ER2 ', '')
        df.columns = df.columns.str.replace('STBD ', '')
        df.columns = df.columns.str.replace('PORT ', '')
        df.columns = df.columns.str.replace('-', ' ')
        df.columns = df.columns.str.replace('   ', '_')
        df.columns = df.columns.str.replace('  ', '_')
        df.columns = df.columns.str.replace(' ', '_')
        df = df[feats]

    return dataframes


###################################################
def indexer(dataframes):
    #all index with unique start time
    master_idx = []


    for df in dataframes:
        df.DateTime = pd.to_datetime(df.DateTime) # converting to datetime parameter
        df['Date'] = df.DateTime.shift(1) # creating a shifted datetime for computing time delta
        df['Delta'] = df.DateTime - df.Date # time delta
        df['New_Start'] = df.Delta > pd.Timedelta(1, unit = 'h') # identifying if readings has a gap of at least 1hr, regular readings are logged at 1Hz
        df.New_Start = np.where(df.New_Start == True, 1, 0) # replacing 1 for true ,0 for false for the identified start up
        start_idx = [0]
        start_idx.extend(list(df[df.New_Start == 1].index)) # identifying the indices for the all unique start up

        #appending start index of df
        master_idx.append(start_idx)
    return master_idx
#####################################################################

def labeler(dataframe,index):
        
    for df,start_indx in zip(dataframe, index):
        df['Label'] = pd.Series()
        for i in start_indx:
            if ((df.GG_SPEED_NGG_KRPM.iloc[i:i+150].any() < 0)):
                for k in range(0,150):
                    if df[Label].iloc[i+k] < 0:
                        df[Label].iloc[i+k] = 'DataError'
                        
            elif (max(df.GG_SPEED_NGG_KRPM.iloc[i:i+59]) > 1.2):
                df['Label'].iloc[i:i+59] = 'DataError'

            elif ((max(df.GG_SPEED_NGG_KRPM.iloc[i+60:i+150]) > 4.5) & (max(df['PT_INLET_TEMP_T54'][i:i+59]) <= 150) &
                 (df['PT_INLET_TEMP_T54'][i] <= 150)  & (df['GG_COMP_DIS_PRES_CDP'][i] < 20)):
                df['Label'].iloc[i:i+150] = 'Cold_Start'

            elif ((max(df.GG_SPEED_NGG_KRPM[i+60:i+150]) > 4.5) & (max(df['PT_INLET_TEMP_T54'][i:i+60]) > 150)):
                df['Label'].iloc[i+60:i+150] = 'Hot_Start'

            elif ((max(df.GG_SPEED_NGG_KRPM[i+60:i+150]) > 2.2) & (max(df['LO_SPLY_PRES'][i+60:i+150]) > 6) &
                  (max(df['FO_MFLD_PRES_FMP'][i+60:i+150]) > 25) & (max(df['PT_INLET_TEMP_T54'][i+60:i+150])>300)):
                df['Label'].iloc[i:i+150] = 'Failed_Start'
                
            elif ( (max(df['FO_MFLD_PRES_FMP'][i+60:i+150]) < 25)& ((max(df['LO_SPLY_PRES'][i+60:i+150]) > 6) or
                 (max(df.PT_INLET_TEMP_T54[i+60:i+150])<300))):
                df['Label'].iloc[i+60:i+150] = 'Motoring'
                
            else:
                df['Label'].iloc[i:i+150] = 'Unknown'                          

        
    return dataframe

################################################
def toll_tensors_150(dataframe, index):
    df_startup = pd.DataFrame()
    df_label = []
    for df,start_indx in zip(dataframe, index):
        for i in start_indx:
            if df[var].iloc[i:i+150].notna().values.all() & pd.notnull(df['Label'][i+149]):
                df_startup = df_startup.append(df[var].iloc[i:i+150])
                df_label.append(df['Label'].iloc[i+149])               
                
    df_label = pd.Series(df_label)        
    df_tensor = df_startup.to_numpy()
    df_tensor = np.split(df_tensor,len(df_label))
    df_tensor=np.array(df_tensor)
    
    return df_tensor, df_label

################################################
def toll_tensors_120(dataframe, index):
    df_startup = pd.DataFrame()
    df_label = []
    for df,start_indx in zip(dataframe, index):
        for i in start_indx:
            if df[var].iloc[i:i+120].notna().values.all() & pd.notnull(df['Label'][i+119]):
                df_startup = df_startup.append(df[var].iloc[i:i+120])
                df_label.append(df['Label'].iloc[i+119])               
                
    df_label = pd.Series(df_label)        
    df_tensor = df_startup.to_numpy()
    df_tensor = np.split(df_tensor,len(df_label))
    df_tensor=np.array(df_tensor)
    
    return df_tensor, df_label
################################################
def toll_tensors_90(dataframe, index):
    df_startup = pd.DataFrame()
    df_label = []
    for df,start_indx in zip(dataframe, index):
        for i in start_indx:
            if df[var].iloc[i:i+90].notna().values.all() & pd.notnull(df['Label'][i+89]):
                df_startup = df_startup.append(df[var].iloc[i:i+90])
                df_label.append(df['Label'].iloc[i+89])               
                
    df_label = pd.Series(df_label)        
    df_tensor = df_startup.to_numpy()
    df_tensor = np.split(df_tensor,len(df_label))
    df_tensor=np.array(df_tensor)
    
    return df_tensor, df_label
################################################
def label_encoder(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc
        

##################################################
def build_classifier_90():
    #ANN classifier
    model = Sequential()
    model.add(LSTM(100, input_shape=(90,9)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
##################################################
def build_classifier_120():
    #ANN classifier
    model = Sequential()
    model.add(LSTM(100, input_shape=(120,9)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
##################################################
def build_classifier_150():
    #ANN classifier
    model = Sequential()
    model.add(LSTM(100, input_shape=(150,9)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



