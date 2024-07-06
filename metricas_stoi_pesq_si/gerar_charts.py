import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def load_csv_and_plot(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    # Extract the preprocessing type from the noisy_filepath
    def extract_preprocessamento(filepath):
        base = os.path.basename(filepath)
        if base.startswith('denoised_enhanced_'):
            return 'denoised_enhanced'
        elif base.startswith('denoised_'):
            return 'denoised'
        elif base.startswith('enhanced_'):
            return 'enhanced'
        else:
            return 'unknown'

    # Check if the 'clean_filepath' column exists
    if 'clean_filepath' not in df.columns or 'noisy_filepath' not in df.columns:
        print("Erro: Coluna 'clean_filepath' ou 'noisy_filepath' não encontrada no DataFrame.")
        return

    # Apply extraction function to create 'preprocessamento' column
    df['preprocessamento'] = df['noisy_filepath'].apply(extract_preprocessamento)

    # Convert metrics to numeric, coercing errors
    df['stoi'] = pd.to_numeric(df['stoi'], errors='coerce')
    df['pesq'] = pd.to_numeric(df['pesq'], errors='coerce')
    df['si-sdr'] = pd.to_numeric(df['si-sdr'], errors='coerce')

    # Define the metrics to plot
    metrics = ['stoi', 'pesq', 'si-sdr']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(18, 18), constrained_layout=True)

    # Get the unique preprocessing types and sort them
    preprocessamentos = df['preprocessamento'].unique()
    preprocessamentos.sort()

    # Create the violin plots
    for i, metric in enumerate(metrics):
        data_to_plot = pd.DataFrame(columns=['preprocessamento', metric])
        for p in preprocessamentos:
            data_to_plot = pd.concat([data_to_plot, df[df['preprocessamento'] == p][['preprocessamento', metric]]], axis=0)
        sns.violinplot(ax=axes[i], x='preprocessamento', y=metric, data=data_to_plot, palette='husl', inner='quartile')
        axes[i].set_title(f'Violin Plot {metric.upper()}')
        axes[i].set_xticks(range(len(preprocessamentos)))
        axes[i].set_xticklabels(preprocessamentos)

    # Save the plot and show
    plt.savefig('metrics_charts.png')
    plt.show()


if __name__ == "__main__":
    # Adjusted path for Windows using raw string (r'...')
    csv_file_path = r'C:\Users\BIA\Desktop\DAPS-dataset\f1_script4.csv'
    load_csv_and_plot(csv_file_path)
