# src/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def generate_sensitivity_plots(csv_path, output_dir):
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. Parameter Sensitivity Line Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='window_size', y='f1_score', hue='alphabet_size', marker='o')
    plt.title('Automata Parameter Sensitivity (Window Size vs F1 Score)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'plots', 'parameter_sensitivity.png'))
    plt.close()

    # 2. Transition Density Heatmap
    # Önce density hesapla: Transition Count / (State Count * State Count)
    df['transition_density'] = df['transition_count'] / (df['state_count'] ** 2)
    pivot_density = df.pivot(index="window_size", columns="alphabet_size", values="transition_density")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_density, annot=True, cmap='YlOrRd', fmt=".4f")
    plt.title('Transition Density Heatmap (W vs A)')
    plt.savefig(os.path.join(output_dir, 'plots', 'transition_density_heatmap.png'))
    plt.close()
    
    print(f"📊 [Görselleştirme] Grafikler {output_dir}/plots/ altına başarıyla kaydedildi!")