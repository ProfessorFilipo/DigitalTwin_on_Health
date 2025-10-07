# twin_sim.py
import simpy, random, csv, time
import numpy as np
import pandas as pd

# --- Config ---
SIM_TIME = 3600  # seconds (1 hour)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Load params ---
params = pd.read_csv('params.csv')  # use CSV from section 2

# --- Simple link model ---
class Link:
    def __init__(self, env, capacity_mbps, latency_ms):
        self.env = env
        self.capacity_bps = capacity_mbps * 1e6
        self.latency = latency_ms / 1000.0
        self.store = simpy.Store(env)
        self.utilization_time = 0.0
        self.last_busy = None
    def transmit(self, pkt_size_bytes):
        tx_time = (pkt_size_bytes * 8) / self.capacity_bps
        start = self.env.now
        if self.last_busy is None or self.env.now >= self.last_busy:
            self.last_busy = self.env.now + tx_time
        else:
            # queueing simple: wait until last_busy
            wait = self.last_busy - self.env.now
            yield self.env.timeout(wait)
            self.last_busy += tx_time
        # simulate transmission + latency
        yield self.env.timeout(tx_time + self.latency)

# --- Device process ---
def device_process(env, row, out_link, log):
    dev = row['device_id']
    pattern = row['comm_pattern']
    avg_kbps = row['avg_data_rate_kbps']
    pkt_size = int(row['packet_size_bytes'])
    priority = row['priority']
    power_idle = float(row['power_idle_w'])
    power_active = float(row['power_active_w'])
    last_sent = env.now
    while env.now < SIM_TIME:
        # determine next send based on pattern
        if 'continuous' in pattern:
            # send small packets at rate avg_kbps
            inter = (pkt_size * 8) / (avg_kbps * 1000.0)
            yield env.timeout(max(0.01, np.random.exponential(inter)))
            # sometimes burst
            if 'burst' in pattern and random.random() < 0.01:
                # send burst of larger packet
                size = pkt_size * 20
            else:
                size = pkt_size
        elif pattern == 'periodic':
            inter = max(0.5, 60.0)  # once per minute nominal
            yield env.timeout(np.random.normal(inter, inter*0.1))
            size = pkt_size
        elif pattern == 'aggregate':
            # edge/server traffic aggregated less frequently
            yield env.timeout(np.random.exponential(1.0))
            size = pkt_size * 10
        elif pattern == 'interactive':
            yield env.timeout(np.random.exponential(5.0))
            size = pkt_size * 5
        else:
            yield env.timeout(1.0)
            size = pkt_size

        # simulate possible peak event
        if random.random() < 0.005:
            size = size * (row['peak_data_rate_kbps'] / max(1,row['avg_data_rate_kbps']))

        # transmit over link
        start = env.now
        tx = out_link.transmit(size)
        yield from tx
        end = env.now
        # log event
        log.append({'time': end, 'device': dev, 'bytes': size, 'latency': end-start, 'priority': priority})
        # energy accounting simple (active during transmission)
        # record power use event
        log.append({'time': end, 'device': dev+'_power', 'power_w': power_active, 'duration_s': end-start})

# --- Runner ---
def run():
    env = simpy.Environment()
    # create links
    # devices -> edge (100 Mbps, 1ms)
    link_de = Link(env, capacity_mbps=100, latency_ms=1)
    # edge -> core (1000 Mbps, 0.5ms)
    link_ec = Link(env, capacity_mbps=1000, latency_ms=0.5)
    # core -> server (1000 Mbps, 0.5ms)
    link_cs = Link(env, capacity_mbps=1000, latency_ms=0.5)

    logs = []

    # instantiate device processes
    for _, row in params.iterrows():
        if row['device_type'] in ['Monitor','Ventilator','InfusionPump']:
            env.process(device_process(env, row, link_de, logs))
        elif row['device_type'] == 'Edge_Node':
            # simplify: edge aggregates incoming from many; model as periodic uploads to server
            def edge_proc(env, row=row, out_link=link_ec, log=logs):
                while env.now < SIM_TIME:
                    yield env.timeout(1.0)  # aggregate every 1s
                    size = max(1000, int(row['avg_data_rate_kbps']*100))  # synthetic
                    start = env.now
                    yield from out_link.transmit(size)
                    end = env.now
                    log.append({'time': end, 'device': row['device_id'], 'bytes': size, 'latency': end-start, 'priority': row['priority']})
            env.process(edge_proc(env))
        elif row['device_type'] == 'Server_Local':
            # server may initiate backups occasionally (bulk)
            def server_proc(env, row=row, out_link=link_cs, log=logs):
                while env.now < SIM_TIME:
                    # small chance of backup/bulk
                    if random.random() < 0.01:
                        size = 5*1024*1024  # 5 MB
                        start = env.now
                        yield from out_link.transmit(size)
                        end = env.now
                        log.append({'time': end, 'device': row['device_id'], 'bytes': size, 'latency': end-start, 'priority': row['priority']})
                    yield env.timeout(10.0)
            env.process(server_proc(env))

    env.run(until=SIM_TIME)

    # post-process logs
    df = pd.DataFrame(logs)
    if df.empty:
        print('No logs produced')
        return
    # compute utilization approximations and metrics
    df_bytes = df[df['device'].str.contains('_power')==False]
    total_bytes = df_bytes['bytes'].sum()
    avg_rate_kbps = total_bytes * 8.0 / SIM_TIME / 1000.0
    print('Total MB sent: %.2f MB' % (total_bytes/1024/1024))
    print('Avg aggregate rate: %.2f kbps' % avg_rate_kbps)

    # energy approx
    ###df_power = df[df['device'].str.contains('_power')]
    ###df_power['energy_j'] = df_power['power_w'] * df_power['duration_s']
    df_power = df[df['device'].str.contains('_power')].copy()
    df_power['energy_j'] = df_power['power_w'] * df_power['duration_s']

    energy_wh = df_power['energy_j'].sum() / 3600.0
    print('Estimated energy (Wh) during sim (active only): %.2f Wh' % energy_wh)

    # simple latency stats
    if 'latency' in df_bytes.columns:
        print('Latency p50/p95 (s):', df_bytes['latency'].quantile(0.5), df_bytes['latency'].quantile(0.95))

    # save logs
    df.to_csv('sim_logs.csv', index=False)
    print('Logs saved to sim_logs.csv')

if __name__ == '__main__':
    run()
