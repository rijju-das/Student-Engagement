import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt

def scalingDF(df):
  # scaler = QuantileTransformer(output_distribution='normal')
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(df)
  df_s = pd.DataFrame(scaled,index=df.index, columns=df.columns)
  return df_s

data_path = "/Users/rijju/Documents/GitHub/Student-Engagement/WACV data" #replace it according to your data path
result_base_path = "/Users/rijju/Documents/GitHub/Student-Engagement"

df0 = pd.read_csv(os.path.join(data_path,"merged_data0.csv"))
df1 = pd.read_csv(os.path.join(data_path,"merged_data1.csv"))
df2 = pd.read_csv(os.path.join(data_path,"merged_data2.csv"))

df00 = df0.loc[df0['confidence'] >= 0.7]
df11 = df1.loc[df1['confidence'] >= 0.7]
df22 = df2.loc[df2['confidence'] >= 0.7]

#concatenate all the data files (disengaged[0], partially engaged[1] and engaged[2])
df = pd.concat([df00,df11,df22])

df = df.sample(frac=1)
df_x = df.loc[:,"x0":"AU45_c"] #extract the AU columns from the dataframe
df_y = df.loc[:,"Label_y"]

df = pd.concat([scalingDF(df_x),df_y],axis=1)

#Conditional Probability Mapping of AUs with Engagement Labels. 
from AU_mapping import AU_mapping
map = AU_mapping()
fig, df_mapAU = map.au_heatmap(df)
fig.savefig(os.path.join(result_base_path,"Results/CondProb_AU_mapping.pdf"))

#Relative Activation Ratios for AUs across Engagement Labels.
from AU_mapping_relative import AU_mapping
map = AU_mapping()
fig, df_mapR = map.au_heatmap(df)
fig.savefig(os.path.join(result_base_path,"Results/Relative_AU_mapping.pdf"))


#Log-Ratio Statistical Discriminative Coefficient (SDC) Scores for AUs.
from AU_mapping_SDC import AU_mapping
map = AU_mapping()
fig, df_mapC = map.au_heatmap(df)
fig.savefig(os.path.join(result_base_path,"Results/SDC_Scores.pdf"))



#Conditional Probability Mapping of AUs with Engagement Labels. 
plt.figure(figsize=(12,5))
for index, row in df_mapAU.T.iterrows():
    plt.plot(row, label=index)
plt.legend()
plt.savefig(os.path.join(result_base_path,"Results/CondProb_AU_mappingLinePlot.pdf"))
plt.show()
