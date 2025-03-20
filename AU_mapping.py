class AU_mapping():
  def prob_au(self,df,label,column):
      """The function performs returns the conditional probabilities 
         between each AU with engagement labels.

            Parameters
            ----------
            df: pandas dataframe
                the features dataframe
            label: the specific engagement labels [0,1 or 2]
            column: the AU names [AU01 to AU45]

            Returns
            -------
            dic: dictionary with conditional probabilities values
            """
      dic = {}
      d = df.loc[df["Label_y"]==label].shape[0]
      
      for c in column:
          df1 = df.loc[(df[c] >=0.1) & (df['Label_y'] == label)]
          dic[c]=round((df1.shape[0]/d),2)
      return dic
  def au_heatmap(self,df):
     
      """The function performs mapping among the AUs and engagement labels
        by calculating their conditional probabilities. And plot the 
        mapping values via a heatmap.

            Parameters
            ----------
            df: pandas dataframe
                the features dataframe

            Returns
            -------
            df_map: pandas dataframe
                The dataframe contains mapping values between each AU and engagement labels
            fig: the heatmap plot of df_map
            """

      import seaborn as sns
      import pandas as pd
      import matplotlib.pyplot as plt
      df_au = df.loc[:,"AU01_r":"AU45_r"]
      columns = df_au.columns
      df_y = df.loc[:,"Label_y"]
      df_au = pd.concat([df_au,df_y], axis=1)
      df_map = pd.DataFrame([self.prob_au(df_au,0,columns),self.prob_au(df_au,1,columns),self.prob_au(df_au,2,columns)],index=["Disengaged","Partially engaged","Engaged"]).T
  
      fig = plt.figure(figsize=(12,8))
      ax=sns.heatmap(df_map, cmap ='magma_r', linewidths = 0.70, annot=True, vmin=0,vmax=1, annot_kws={"size": 16})
      
      plt.xticks(fontsize=18)
      plt.yticks(fontsize=18)

      # Increase font size for color bar ticks and label
      cbar = ax.collections[0].colorbar
      cbar.ax.tick_params(labelsize=18)
      return fig, df_map