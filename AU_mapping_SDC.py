import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class AU_mapping():
    def prob_au(self, df, columns, labels, threshold=0.1):
        """
        Calculate Statistical Discriminative Coefficient (SDC) for each AU.
        
        Args:
            df (pd.DataFrame): DataFrame containing AU columns and label column 'Label_y'.
            columns (list): List of AU column names.
            labels (list): List of unique labels ['disengaged', 'partially engaged', 'engaged'].
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
                d = df[df["Label_y"] == label].shape[0]
                if d == 0:
                    continue  # Skip if no samples for this label
                
                activated_count = df.loc[(df[c] >= threshold) & (df['Label_y'] == label)].shape[0]
                P_c_li = activated_count / d  # P(c | l_i)
                
                if P_c[c] > 0:  # Avoid division by zero
                    sdc += P_li[label] * np.log(P_c_li / P_c[c])
            
            sdc_scores[c] = round(sdc, 4)
        
        return sdc_scores

    def au_heatmap(self, df):
        """
        Maps AUs to engagement labels by calculating SDC scores and plotting them via a heatmap.

        Parameters
        ----------
        df: pandas DataFrame
            The features dataframe

        Returns
        -------
        df_map: pandas DataFrame
            DataFrame containing SDC scores for each AU
        fig: matplotlib figure object
            Heatmap plot of SDC scores
        """
        # Extract AU columns
        df_au = df.loc[:, "AU01_r":"AU45_r"]
        columns = df_au.columns
        df_y = df["Label_y"]
        df_au = pd.concat([df_au, df_y], axis=1)
        
        # Define your engagement labels
        labels = [0, 1, 2]
        l = df_au.shape[0]
        print(l)
        # Calculate SDC scores
        sdc_scores = self.prob_au(df_au, columns, labels, threshold=0.1)
        
        # Create DataFrame for heatmap
        df_map = pd.DataFrame.from_dict(sdc_scores, orient='index', columns=["Statistical Discriminant Coefficient (SDC)"])
        
        # Plotting the heatmap
        fig = plt.figure(figsize=(6, 6))
        ax=sns.heatmap(df_map, cmap='magma', linewidths=0.7, annot=False, vmin=-0.12,vmax=0, annot_kws={"size": 16})
        # plt.title("AU Discriminative Power (SDC Scores)")
        # plt.xlabel("SDC Score")
        # plt.ylabel("Action Units")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Increase font size for color bar ticks and label
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        plt.show()

        return fig, df_map
