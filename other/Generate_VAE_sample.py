from VAE_transform_feature import VAE_transform
import pandas as pd
import numpy as np

data_dir = "/mnt/binf/eric/Mercury_Dec2023/Feature_all_Mar2024.pkl"
feature_list = ["Frag", "Arm", "Cnv", "Griffin", "MCMS", "Ma","Gemini"]
input_size_list = [1100,950,2500,2600,180,25,60]

if(data_dir.endswith('.csv')):
    data = pd.read_csv(data_dir)
elif(data_dir.endswith('.pkl')):
    data = pd.read_pickle(data_dir)
    
data_idonly = data.loc[:,['SampleID','Train_Group','train','Project','R01B_label']]
times_to_duplicate = 10

data_VAE = None

for i in range(len(feature_list)):
    
    if input_size_list[i] >= 500:
        
        # use "No" in the control option; set beta to 0.0 to skip the task classifier
        
        data_tmp = VAE_transform(data_dir=data_dir,input_size=input_size_list[i],feature_type=feature_list[i], encoding_size=64, control="No", alpha=0.5, beta=0.0, num_epochs=200, num_repeats=times_to_duplicate)
        
        if i == 0:
            data_VAE = data_tmp.copy()
        else:
            data_tmp = data_tmp.drop(columns=['SampleID','Train_Group','train','Project','R01B_label'])
            data_VAE = pd.concat([data_VAE, data_tmp], axis=1)
        
    else:
        data_tmp = data.filter(regex = feature_list[i], axis=1).add_prefix("VAE_raw_")
        data_R01Brep_tmp = data.loc[data["R01B_label"] == "R01B_match"].filter(regex = feature_list[i], axis=1).add_prefix("VAE_raw_")
        data_repeat_tmp=None
        
        for k in range(times_to_duplicate):
            if k == 0:
                data_repeat_tmp=data_tmp.copy()
            else:
                data_repeat_tmp=pd.concat([data_repeat_tmp,data_R01Brep_tmp],axis=0,ignore_index=True) # simply duplicate R01B_match samples    

        data_VAE = pd.concat([data_VAE, data_repeat_tmp], axis=1)    

        
data_full = data_VAE.copy()
print(f"------------- the shape of the final data is {data_full.shape} -------------")    
   
data_full.to_pickle("/mnt/binf/eric/Mercury_Dec2023/Feature_all_Mar2024_VAE_simu700.pkl") 

### drop AE_raw columns

# AE_raw_columns = [colname for colname in data_full.columns if 'AE_raw_' in colname]
# data_full_norep = data_full.drop(columns=AE_raw_columns)

# data_full_norep.to_pickle("/mnt/binf/eric/Mercury_Dec2023/Feature_all_Dec2023_withAE_norep.pkl") 



    


