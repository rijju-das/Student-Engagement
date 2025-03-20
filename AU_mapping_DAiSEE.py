import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class AU_mapping():
    def activation_rate_heatmap(self, df, threshold=0.1):
        """
        Computes the overall activation rate for each AU (from AU01_r to AU45_r) and
        plots the result as a heatmap.
        
        The overall activation rate for an AU is defined as the fraction of frames
        in which its intensity exceeds the specified threshold.
        
        Parameters
        ----------
        df : pandas DataFrame
            DataFrame containing frame-level AU values.
        threshold : float, optional
            Threshold above which an AU is considered activated. Default is 0.1.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The resulting heatmap figure.
        activation_df : pandas DataFrame
            A DataFrame containing the overall activation rates for each AU.
        """
        # Select columns from AU01_r to AU45_r
        df_au = df.loc[:, "AU01_r":"AU45_r"]
        
        # Compute the overall activation rate for each AU
        activation_rates = {}
        for col in df_au.columns:
            activation_rates[col] = np.mean(df_au[col] >= threshold)
        
        # Convert the activation rates dictionary into a DataFrame
        # Each row represents an AU; we create a single-column DataFrame.
        activation_df = pd.DataFrame.from_dict(activation_rates, orient='index', columns=['Activation Rate'])
        
        # Optionally, sort by the activation rate
        # activation_df = activation_df.sort_values(by='Activation Rate', ascending=False)
        
        # Plot the activation rates as a heatmap.
        # Since there is only one column, the heatmap will be a single-column plot.
        fig = plt.figure(figsize=(4, 6))
        sns.heatmap(activation_df, cmap='magma_r', annot=True, linewidths=0.7, cbar_kws={'label': 'Activation Rate'},
                    vmin=0, vmax=1)
        plt.title("Overall AU Activation Rates")
        plt.xlabel("Activation Rate")
        plt.ylabel("Action Units")
        plt.tight_layout()
        plt.show()
        
        return fig, activation_df

    def au_heatmap(self, df):
        """
        Unified function to display overall AU activation rates as a heatmap.
        
        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame containing frame-level AU values.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The resulting heatmap figure.
        df_map : pandas DataFrame
            DataFrame containing overall activation rates for each AU.
        """
        return self.activation_rate_heatmap(df)
