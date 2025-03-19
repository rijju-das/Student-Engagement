import pandas as pd
import numpy as np
class AU_mapping():
    def prob_au(df, columns, labels, threshold=0.002):
        """
        Calculate Statistical Discriminative Coefficient (SDC) for each AU.
        
        Args:
            df (pd.DataFrame): DataFrame containing AU columns and label column 'Label_y'.
            columns (list): List of AU column names.
            labels (list): List of unique labels ('engaged', 'partially engaged', 'disengaged').
            threshold (float): Threshold for considering an AU as "activated".
            
        Returns:
            dict: Dictionary with AUs as keys and SDC scores as values.
        """
        sdc_scores = {}
        total_samples = df.shape[0]  # Total number of samples

        # Calculate P(c) for each AU (overall activation rate across all labels)
        P_c = {c: df[df[c] >= threshold].shape[0] / total_samples for c in columns}

        # Calculate P(l_i) for each label
        P_li = {label: df[df["Label_y"] == label].shape[0] / total_samples for label in labels}

        for c in columns:
            sdc = 0
            for label in labels:
                # Calculate P(c | l_i) - probability of activation given the label
                d = df[df["Label_y"] == label].shape[0]
                if d == 0:
                    continue  # Skip if no samples for this label
                
                activated_count = df.loc[(df[c] >= threshold) & (df['Label_y'] == label)].shape[0]
                P_c_li = activated_count / d  # P(c | l_i)
                
                if P_c[c] > 0:  # Avoid division by zero
                    sdc += P_li[label] * (P_c_li / P_c[c])
            
            sdc_scores[c] = round(sdc, 4)
        
        return sdc_scores


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
    
        fig = plt.figure(figsize=(5,5))
        sns.heatmap(df_map, cmap ='Purples', linewidths = 0.70, vmin=0,vmax=1)
        return fig, df_map