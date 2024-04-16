#!/bin/bash
feature_type_list=(CCSA VAE AE_Frag AE_Arm AE_Cnv AE_Griffin Frag Arm Griffin Cnv MCMS Gemini Ma)
input_size=(800 550 64 64 64 64 1200 1000 2650 2550 350 120 50)
output_path=/mnt/binf/eric/DANN_Mar2024Results/DANN_0410_VAE_1cluster_noKAG9v2
data_dir=/mnt/binf/eric/Mercury_Dec2023/Feature_all_Apr2024_VAE_noKAG9.pkl
R01BTune=No
cluster_method_list=(None kmeans DBSCAN GMM MeanShift)
nfold=5
for i in {1..1}
do
    for j in {0..4}
    do
        python ./DANN_run.py ${feature_type_list[i]} 1D ${input_size[i]} 100 1000 ${output_path} ${data_dir} ${R01BTune} ${cluster_method_list[j]} ${nfold}
    done
done