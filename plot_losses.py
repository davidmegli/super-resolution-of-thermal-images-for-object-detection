import pandas as pd
import matplotlib.pyplot as plt

def plot_losses(csv_file):
    """
    Legge un file CSV contenente dati di addestramento e validazione e crea 3 grafici:
    - `box_loss` (train + val)
    - `cls_loss` (train + val)

    Args:
        csv_file (str): Percorso al file CSV.
    """
    # Leggi il CSV con Pandas
    data = pd.read_csv(csv_file)

    # Controlla se le colonne richieste sono presenti
    required_columns = [
        "epoch", "train/box_loss", "val/box_loss",
        "train/cls_loss", "val/cls_loss"
    ]
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Il file CSV non contiene tutte le colonne richieste.")

    # Parametri per la grandezza del font
    title_fontsize = 18
    label_fontsize = 16
    legend_fontsize = 18
    ticks_fontsize = 14
    linewidth = 2

    # Plot `box_loss` (train e val)
    plt.figure(figsize=(6, 5))
    plt.plot(data["epoch"], data["train/box_loss"], label="TRAIN BOX LOSS", color='blue', linewidth=linewidth)
    plt.plot(data["epoch"], data["val/box_loss"], label="VALIDATION BOX LOSS", color='green', linewidth=linewidth)
    #plt.title("Box Loss (Train vs Validation)", fontsize=title_fontsize)
    plt.xlabel("EPOCH", fontsize=label_fontsize)
    plt.ylabel("BOX LOSS", fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid()
    plt.show()

    # Plot `cls_loss` (train e val)
    plt.figure(figsize=(6, 5))
    plt.plot(data["epoch"], data["train/cls_loss"], label="TRAIN CLASS LOSS", color='blue', linewidth=linewidth)
    plt.plot(data["epoch"], data["val/cls_loss"], label="VALIDATION CLASS LOSS", color='green', linewidth=linewidth)
    #plt.title("Class Loss (Train vs Validation)", fontsize=title_fontsize)
    plt.xlabel("EPOCH", fontsize=label_fontsize)
    plt.ylabel("CLASS LOSS", fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid()
    plt.show()

# Esempio di utilizzo
plot_losses('C:\\Users\\verba\\Desktop\\#UNI Ing Inf Corsi 3° anno\\TESI\\Progetto\\Model) FINETUNED YOLOv11m on FLIR training set 8bit jpeg 100 epochs\\results.csv')