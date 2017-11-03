import numpy as np
import sys
import pandas as pd

if __name__=="__main__":
    if(sys.argv.__len__())>1:
        inputFilePath = sys.argv[1]
    else:
       print("Please enter Filepath of data")

    print("Processing...")
    ### -----------read file------------###
    df=pd.read_csv(inputFilePath,header=None)

 ### -----------Finding null values------------###
    indexStr=np.where(pd.isnull(df))[0]

    ### -----------droping null values------------###
    df=df.drop(df.index[indexStr])

    ### -----------Transfer Categorical data to numeric------------###
    for i in range(len(df.columns)):
        df[i]=(pd.Categorical(df[i])).codes

   ###--------------Standardization of data----------------------###
    df_result=df.copy()
    mean_val=df.mean()
    std_val=df.std()
    for i in range(len(df.columns)):
        df_result[i]=(df[i]-mean_val[i])/std_val[i]
    df_result.to_csv(sys.argv[2])
    print("PreProcessed data is written at given filePath")



