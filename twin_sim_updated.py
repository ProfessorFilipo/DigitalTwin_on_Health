# twin_sim_updated.py
import simpy, random, csv, time
import numpy as np
import pandas as pd

# --- Config ---
SIM_TIME = 3600  # seconds (1 hour)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Load params ---
params = pd.read_csv('params.csv')  # use CSV from earlier

# --- Simple link model ---
class Link:
    def __init__(self, env, capacity_mbps, latency_ms):
        self.env = env
        self.capacity_bps = capacity_mbps * 1e6
        self.latency = latency_ms / 1000.0
        self.last_busy = None
    def transmit(self, pkt_size_bytes):
        tx_time = (pkt_size_bytes * 8) / self.capacity_bps
        if self.last_busy is None or self.env.now >= self.last_busy:
            self.last_busy = self.env.now + tx_time
            yield self.env.timeout(tx_time + self.latency)
        else:
            wait = self.last_busy - self.env.now
            yield self.env.timeout(wait)
            self.last_busy += tx_time
            yield self.env.timeout(tx_time + self.latency)

# --- Device process ---
def device_process(env, row, out_link, log):
    dev = row['device_id']
    zone = row.get('zone', 'unknown')
    pattern = row['comm_pattern']
    avg_kbps = row['avg_data_rate_kbps']
    pkt_size = int(row['packet_size_bytes'])
    priority = row['priority']
    power_idle = float(row['power_idle_w'])
    power_active = float(row['power_active_w'])
    last_active_time = 0.0
    total_active_energy_j = 0.0
    last_event_time = env.now

    while env.now < SIM_TIME:
        # determine next send based on pattern
        if 'continuous' in pattern:
            inter = (pkt_size * 8) / (avg_kbps * 1000.0)
            yield env.timeout(max(0.01, np.random.exponential(inter)))
            if 'burst' in pattern and random.random() < 0.01:
                size = pkt_size * 20
            else:
                size = pkt_size
        elif pattern == 'periodic':
            inter = max(0.5, 60.0)
            yield env.timeout(max(0.1, np.random.normal(inter, inter*0.1)))
            size = pkt_size
        elif pattern == 'aggregate':
            yield env.timeout(np.random.exponential(1.0))
            size = pkt_size * 10
        elif pattern == 'interactive':
            yield env.timeout(np.random.exponential(5.0))
            size = pkt_size * 5
        else:
            yield env.timeout(1.0)
            size = pkt_size

        # possible peak event
        if random.random() < 0.005:
            size = int(size * (row['peak_data_rate_kbps'] / max(1, row['avg_data_rate_kbps'])))

        # transmit over link
        start = env.now
        yield from out_link.transmit(size)
        end = env.now

        # log traffic event
        log.append({'time': end, 'device': dev, 'zone': zone, 'type': 'traffic', 'bytes': size, 'latency': end-start, 'priority': priority})

        # energy accounting: active during transmission
        duration = end - start
        energy_j = power_active * duration
        total_active_energy_j += energy_j
        last_active_time = end
        log.append({'time': end, 'device': dev, 'zone': zone, 'type': 'power_active', 'power_w': power_active, 'duration_s': duration, 'energy_j': energy_j})

    # upon process end, record idle baseline energy for entire simulation
    uptime = SIM_TIME
    idle_energy_j = power_idle * uptime
    log.append({'time': env.now, 'device': dev, 'zone': zone, 'type': 'power_idle', 'power_w': power_idle, 'duration_s': uptime, 'energy_j': idle_energy_j})
    # also record aggregate active energy per device for convenience
    log.append({'time': env.now, 'device': dev, 'zone': zone, 'type': 'power_active_total', 'energy_j': total_active_energy_j})

# --- Runner ---
def run():
    env = simpy.Environment()
    # create links
    link_de = Link(env, capacity_mbps=100, latency_ms=1)
    link_ec = Link(env, capacity_mbps=1000, latency_ms=0.5)
    link_cs = Link(env, capacity_mbps=1000, latency_ms=0.5)

    logs = []

    # instantiate processes according to device_type
    for _, row in params.iterrows():
        dtype = row['device_type']
        if dtype in ['Monitor','Ventilator','InfusionPump']:
            env.process(device_process(env, row, link_de, logs))
        elif dtype == 'Edge_Node':
            def edge_proc(env, row=row, out_link=link_ec, log=logs):
                dev = row['device_id']
                zone = row.get('zone', 'EquipmentRoom')
                while env.now < SIM_TIME:
                    yield env.timeout(1.0)
                    size = max(1000, int(row['avg_data_rate_kbps']*100))
                    start = env.now
                    yield from out_link.transmit(size)
                    end = env.now
                    log.append({'time': end, 'device': dev, 'zone': zone, 'type': 'traffic', 'bytes': size, 'latency': end-start, 'priority': row['priority']})
                    # energy events (aggregate)
                    dur = end-start
                    energy_j = row['power_active_w'] * dur
                    log.append({'time': end, 'device': dev, 'zone': zone, 'type': 'power_active', 'power_w': row['power_active_w'], 'duration_s': dur, 'energy_j': energy_j})
                # idle energy record
                idle_energy_j = row['power_idle_w'] * SIM_TIME
                log.append({'time': env.now, 'device': dev, 'zone': zone, 'type': 'power_idle', 'power_w': row['power_idle_w'], 'duration_s': SIM_TIME, 'energy_j': idle_energy_j})
            env.process(edge_proc(env))
        elif dtype == 'Server_Local':
            def server_proc(env, row=row, out_link=link_cs, log=logs):
                dev = row['device_id']
                zone = row.get('zone', 'EquipmentRoom')
                total_active_energy = 0.0
                while env.now < SIM_TIME:
                    if random.random() < 0.01:
                        size = 5*1024*1024  # 5 MB backup
                        start = env.now
                        yield from out_link.transmit(size)
                        end = env.now
                        log.append({'time': end, 'device': dev, 'zone': zone, 'type': 'traffic', 'bytes': size, 'latency': end-start, 'priority': row['priority']})
                        dur = end-start
                        energy_j = row['power_active_w'] * dur
                        total_active_energy += energy_j
                        log.append({'time': end, 'device': dev, 'zone': zone, 'type': 'power_active', 'power_w': row['power_active_w'], 'duration_s': dur, 'energy_j': energy_j})
                    yield env.timeout(10.0)
                idle_energy_j = row['power_idle_w'] * SIM_TIME
                log.append({'time': env.now, 'device': dev, 'zone': zone, 'type': 'power_idle', 'power_w': row['power_idle_w'], 'duration_s': SIM_TIME, 'energy_j': idle_energy_j})
                log.append({'time': env.now, 'device': dev, 'zone': zone, 'type': 'power_active_total', 'energy_j': total_active_energy})
            env.process(server_proc(env))
        else:
            # admin workstation or others - simple traffic
            def admin_proc(env, row=row, out_link=link_cs, log=logs):
                dev = row['device_id']
                zone = row.get('zone', 'EquipmentRoom')
                while env.now < SIM_TIME:
                    yield env.timeout(np.random.exponential(5.0))
                    size = row['avg_data_rate_kbps'] * 100
                    start = env.now
                    yield from out_link.transmit(size)
                    end = env.now
                    log.append({'time': end, 'device': dev, 'zone': zone, 'type': 'traffic', 'bytes': size, 'latency': end-start, 'priority': row['priority']})
                idle_energy_j = row['power_idle_w'] * SIM_TIME
                log.append({'time': env.now, 'device': dev, 'zone': zone, 'type': 'power_idle', 'power_w': row['power_idle_w'], 'duration_s': SIM_TIME, 'energy_j': idle_energy_j})
            env.process(admin_proc(env))

    env.run(until=SIM_TIME)

    # post-process logs into DataFrame
    df = pd.DataFrame(logs)
    if df.empty:
        print('No logs produced')
        return

    # ensure consistent dtypes
    # separate traffic and power records
    df_traffic = df[df['type'] == 'traffic'].copy()
    df_power = df[df['type'].str.contains('power')].copy()  # copy to avoid SettingWithCopyWarning

    # aggregate overall metrics
    total_bytes = df_traffic['bytes'].sum()
    avg_rate_kbps = total_bytes * 8.0 / SIM_TIME / 1000.0
    print('Total MB sent: %.2f MB' % (total_bytes/1024/1024))
    print('Avg aggregate rate: %.2f kbps' % avg_rate_kbps)

    # latency stats overall
    if not df_traffic.empty:
        print('Latency p50/p95 (s):', df_traffic['latency'].quantile(0.5), df_traffic['latency'].quantile(0.95))

    # energy: compute per device active and idle
    # active energy: sum energy_j where type == power_active or power_active_total
    active_df = df_power[df_power['type'].isin(['power_active','power_active_total'])]
    # some rows have energy_j, some have power_w/duration_s; normalize
    if 'duration_s' in active_df.columns:
        active_df = active_df.copy()
        active_has_energy = active_df['energy_j'].notna()
        # compute where missing
        missing_energy_idx = active_df[~active_has_energy & active_df['type']=='power_active'].index
        for idx in missing_energy_idx:
            row = active_df.loc[idx]
            active_df.at[idx,'energy_j'] = row['power_w'] * row['duration_s']
    active_energy_per_device = active_df.groupby('device')['energy_j'].sum().to_dict()

    idle_df = df_power[df_power['type'] == 'power_idle'].copy()
    idle_energy_per_device = idle_df.groupby('device')['energy_j'].sum().to_dict()

    # total energy per device (Wh)
    devices = sorted(set(list(active_energy_per_device.keys()) + list(idle_energy_per_device.keys())))
    energy_summary = []
    for d in devices:
        active_j = active_energy_per_device.get(d, 0.0)
        idle_j = idle_energy_per_device.get(d, 0.0)
        total_j = active_j + idle_j
        energy_wh = total_j / 3600.0
        energy_summary.append({'device': d, 'active_j': active_j, 'idle_j': idle_j, 'total_wh': energy_wh})

    df_energy_summary = pd.DataFrame(energy_summary)

    # metrics per device: bytes, avg_rate_kbps, latency p50/p95
    bytes_per_device = df_traffic.groupby('device')['bytes'].sum().to_dict()
    count_per_device = df_traffic.groupby('device').size().to_dict()
    latency_p50 = df_traffic.groupby('device')['latency'].quantile(0.5).to_dict()
    latency_p95 = df_traffic.groupby('device')['latency'].quantile(0.95).to_dict()

    device_metrics = []
    for d in sorted(set(list(bytes_per_device.keys()) + [r['device'] for r in energy_summary])):
        b = bytes_per_device.get(d, 0)
        avg_kbps_dev = b * 8.0 / SIM_TIME / 1000.0
        device_metrics.append({
            'device': d,
            'total_bytes': b,
            'avg_kbps': avg_kbps_dev,
            'p50_latency_s': latency_p50.get(d, np.nan),
            'p95_latency_s': latency_p95.get(d, np.nan),
            'total_energy_wh': df_energy_summary[df_energy_summary['device']==d]['total_wh'].values[0] if d in df_energy_summary['device'].values else 0.0
        })

    df_device_metrics = pd.DataFrame(device_metrics)
    df_device_metrics.to_csv('device_metrics.csv', index=False)
    print('Device metrics saved to device_metrics.csv')

    # metrics per zone
    if 'zone' in df_traffic.columns:
        bytes_zone = df_traffic.groupby('zone')['bytes'].sum().reset_index()
        bytes_zone['avg_kbps'] = bytes_zone['bytes'] * 8.0 / SIM_TIME / 1000.0
        # energy per zone
        df_energy_all = pd.concat([idle_df, active_df], ignore_index=True, sort=False)
        energy_zone = df_energy_all.groupby('zone')['energy_j'].sum().reset_index()
        energy_zone['energy_wh'] = energy_zone['energy_j'] / 3600.0
        zone_summary = bytes_zone.merge(energy_zone, on='zone', how='outer').fillna(0)
        zone_summary.to_csv('zone_metrics.csv', index=False)
        print('Zone metrics saved to zone_metrics.csv')

    # overall energy
    total_energy_wh = df_energy_summary['total_wh'].sum()
    print('Estimated total energy (Wh) during sim: %.2f Wh' % total_energy_wh)

    # save full logs
    df.to_csv('sim_logs.csv', index=False)
    print('Logs saved to sim_logs.csv')

if __name__ == '__main__':
    run()
