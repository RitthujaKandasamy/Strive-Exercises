from timeseries2 import x
import numpy as np
import pandas as pd




# Extract features
def getfeatures(feature):

    new_feature = []
    
    # loop only single column in sequence
    for i in range(feature.shape[0]):              #row(60078)

        seq_feature_new = []   
        
        seq_feature_new.append(np.std(feature[i, :, 4]))  
        seq_feature_new.append(np.min(feature[i, :, 0]))
        seq_feature_new.append(np.std(feature[i, :, 3]))
        seq_feature_new.append(np.max(feature[i, :, 5]))
        seq_feature_new.append(np.min(feature[i, :, 6]))
        seq_feature_new.append(np.median(feature[i, :, 11]))
        seq_feature_new.append(np.std(feature[i, :, 1]))
        seq_feature_new.append(np.median(feature[i, :, 2]))
        seq_feature_new.append(np.max(feature[i, :, 13]))
        seq_feature_new.append(np.min(feature[i, :, 7]))
        seq_feature_new.append(np.std(feature[i, :, 12]))
        seq_feature_new.append(np.median(feature[i, :, 9]))
        seq_feature_new.append(np.min(feature[i, :, 10]))
        seq_feature_new.append(np.std(feature[i, :, 8]))
        seq_feature_new.append(np.max(feature[i, :, 12]))


        
        # loop each columns in sequence
        for j in range(feature.shape[2]):               # columns(14)

           seq_feature_new.append(np.mean(feature[i][:, j]))   # getting mean for each columns(14)
           seq_feature_new.append(np.max(feature[i][:, j]) - np.min(feature[i][:, j]))

        new_feature.append(seq_feature_new)

    return np.array(new_feature)


new_x = getfeatures(x)
#print(new_x.shape)



# create dataframe
new_data = pd.DataFrame(new_x)
#print(new_data.head())




# dataframe result

"""

(60078, 43)
         0       1         2     3     4      5         6        7      8     9         10     11       12  ...    30        31    32        33    34           35    36        37    38        39    40          41      42
0  0.405860  996.50  0.189444  3.33  3.01  0.330  0.177615  265.135  214.3  0.19  0.491867  3.085  1307.75  ...  0.03  1.918333  0.08  3.081667  0.13  1308.973333  2.49  0.468333  0.84  0.940000  1.25  177.500000   78.20
1  0.466964  996.50  0.534010  3.44  2.90  0.265  0.485023  264.825  190.3  0.19  0.158955  2.980  1305.69  ...  0.03  1.890000  0.22  3.036667  0.35  1309.785000  6.56  0.320000  0.43  0.690000  0.38  170.650000   71.70
2  0.180278  996.81  0.123243  3.17  2.89  0.140  0.103669  264.645  272.4  0.19  0.158298  2.950  1311.37  ...  0.02  1.838333  0.05  2.950000  0.09  1311.991667  1.64  0.178333  0.26  0.565000  0.50  176.550000  157.10
3  0.349603  996.99  0.283279  3.12  2.73  0.205  0.243630  264.245  240.0  0.22  0.189861  2.825  1312.78  ...  0.02  1.755000  0.10  2.818333  0.17  1314.430000  3.38  0.230000  0.33  0.628333  0.50  175.193333  173.84
4  0.296742  997.37  0.173469  2.97  2.64  0.365  0.154164  263.675  157.0  0.22  0.173365  2.705  1316.25  ...  0.02  1.680000  0.08  2.696667  0.11  1317.613333  2.56  0.335000  0.41  0.813333  0.50  115.286667   96.28

[5 rows x 43 columns]

"""