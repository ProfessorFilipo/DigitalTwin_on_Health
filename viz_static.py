import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurações de pasta de saída
LOG_CSV = 'sim_logs.csv'
DEVICE_METRICS_CSV = 'device_metrics.csv'
ZONE_METRICS_CSV = 'zone_metrics.csv'
OUT_DIR = 'plots'
os.makedirs(OUT_DIR, exist_ok=True)

sns.set(style='whitegrid')

# Carregar dados
df = pd.read_csv(LOG_CSV)
df_dev = pd.read_csv(DEVICE_METRICS_CSV) if os.path.exists(DEVICE_METRICS_CSV) else None
df_zone = pd.read_csv(ZONE_METRICS_CSV) if os.path.exists(ZONE_METRICS_CSV) else None

# Garantir coluna 'time' numérica
if 'time' in df.columns:
    df['time'] = pd.to_numeric(df['time'], errors='coerce')

# Função para obter cores
def cmap_colors(n, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / max(1, n - 1)) for i in range(n)]

# 1) throughput ao longo do tempo
if 'type' in df.columns and 'bytes' in df.columns and 'time' in df.columns:
    traffic = df[df['type'] == 'traffic'].copy()
    BIN = 60.0
    traffic['tbin'] = (traffic['time'] // BIN) * BIN
    agg = traffic.groupby('tbin')['bytes'].sum().reset_index()
    agg['kbps'] = agg['bytes'] * 8.0 / BIN / 1000.0
    plt.figure(figsize=(10, 4))
    plt.plot(agg['tbin'], agg['kbps'], marker='o', linestyle='-')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Throughput (kbps)')
    plt.title('Aggregate Network Throughput Over Time')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'throughput_time.png'), dpi=150)
    plt.close()

# 2) Top 20 dispositivos por bytes
if not traffic.empty:
    bytes_dev = traffic.groupby('device')['bytes'].sum().reset_index().sort_values('bytes', ascending=False)
    top = bytes_dev.head(20).reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    color = cmap_colors(1, 'viridis')[0]
    sns.barplot(data=top, x='bytes', y='device', color=color)
    plt.xlabel('Total Bytes Sent')
    plt.ylabel('Device')
    plt.title('Top 20 Devices by Total Data')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'bytes_per_device_top20.png'), dpi=150)
    plt.close()

# 3) Distribuição de latência (boxplot) para os 10 principais dispositivos
if not traffic.empty:
    top_devices = bytes_dev['device'].head(10).tolist()
    df_lat = traffic[traffic['device'].isin(top_devices)].copy()
    plt.figure(figsize=(10, 6))
    box_color = cmap_colors(1, 'Pastel1')[0]
    sns.boxplot(data=df_lat, x='device', y='latency', color=box_color)
    plt.yscale('symlog')
    plt.xlabel('Device')
    plt.ylabel('Latency (seconds)')
    plt.title('Latency Distribution for Top 10 Devices')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'latency_box_top10.png'), dpi=150)
    plt.close()

# 4) Consumo de energia por dispositivo (barra)
if df_dev is not None and not df_dev.empty:
    df_dev_sorted = df_dev.sort_values('total_energy_wh', ascending=False).head(30).reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    energy_color = cmap_colors(1, 'magma')[0]
    sns.barplot(data=df_dev_sorted, x='total_energy_wh', y='device', color=energy_color)
    plt.xlabel('Total Energy (Wh)')
    plt.ylabel('Device')
    plt.title('Top 30 Devices by Energy Consumption')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'energy_per_device_top30.png'), dpi=150)
    plt.close()

# 5) Média de throughput por zona (se disponível)
if df_zone is not None and not df_zone.empty:
    z = df_zone.sort_values('avg_kbps', ascending=False).reset_index(drop=True)
    plt.figure(figsize=(8, 4))
    colors = cmap_colors(len(z), 'YlOrBr')
    sns.barplot(data=z, x='zone', y='avg_kbps', color=colors[0])
    plt.xlabel('Zone')
    plt.ylabel('Average Throughput (kbps)')
    plt.title('Average Throughput per Zone')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'throughput_per_zone.png'), dpi=150)
    plt.close()

print('All plots have been generated and saved in the folder:', OUT_DIR)

# Após salvar todos os gráficos individualmente, gerar o PDF consolidado
from matplotlib.backends.backend_pdf import PdfPages
import glob

# Lista de arquivos PNG a serem incluídos
png_files = sorted(glob.glob(os.path.join(OUT_DIR, '*.png')))

# Caminho do PDF de saída
pdf_path = os.path.join(OUT_DIR, 'report.pdf')

with PdfPages(pdf_path) as pdf:
    for png in png_files:
        fig = plt.figure()
        img = plt.imread(png)
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

print(f'PDF report created at: {pdf_path}')
