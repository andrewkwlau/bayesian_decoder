import numpy as np
import pandas as pd
from dataclasses import dataclass, fields
import glob
import re
from pathlib import Path


@dataclass
class NpxlData:
    mouse_ID: str
    date: str
    tau: float
    rewardzone: list

    def __post_init__(self):
        self.find_areas()
        self.load_meta()


    def find_areas(self):
        # Find all areas for the given mouse_ID and date
        folder = Path(f"../datafiles/{self.mouse_ID}/{self.date}")
        if self.date == '20240813':
            pattern = re.compile(rf"{self.date}_az_{self.mouse_ID}_imec0_\d+_(.*?)_{str(int(self.tau * 1000))}ms")
        else:
            pattern = re.compile(rf"{self.date}_az_{self.mouse_ID}_\d+_(.*?)_{str(int(self.tau * 1000))}ms")
        areas = set()
        for file in folder.glob("*.npz"):
            match = pattern.search(file.name)
            if match:
                areas.add(match.group(1))
        self.areas = sorted(areas)
        
    
    def load_area(self, area):
        self.loaded_area = area
        ms = str(int(self.tau * 1000))
        # Set path
        if self.date == '20240812':
            spikerate_lgt_path = Path(f"../datafiles/{self.mouse_ID}/{self.date}/{self.date}_az_{self.mouse_ID}_725_{area}_{ms}ms_spike_rate.npz")
            spikerate_drk_path = Path(f"../datafiles/{self.mouse_ID}/{self.date}/{self.date}_az_{self.mouse_ID}_1322_{area}_{ms}ms_spike_rate.npz")
            spikecount_lgt_path = Path(f"../datafiles/{self.mouse_ID}/{self.date}/{self.date}_az_{self.mouse_ID}_725_{area}_{ms}ms_spike_count.npz")
            spikecount_drk_path = Path(f"../datafiles/{self.mouse_ID}/{self.date}/{self.date}_az_{self.mouse_ID}_1322_{area}_{ms}ms_spike_count.npz") 
            # Load data
            self.fr_lgt = np.load(spikerate_lgt_path)['fr']
            self.fr_drk = np.load(spikerate_drk_path)['fr']
            self.spikes_lgt = np.load(spikecount_lgt_path)['count']
            self.spikes_drk = np.load(spikecount_drk_path)['count']
            self.pos_lgt = np.load(spikecount_lgt_path)['pos'][:,1,:]
            self.pos_drk = np.load(spikecount_drk_path)['pos'][:,1,:]

        elif self.date == '20240813':
            spikerate_lgt_path = Path(f"../datafiles/{self.mouse_ID}/{self.date}/{self.date}_az_{self.mouse_ID}_imec0_725_{area}_{ms}ms_spike_rate.npz")
            spikerate_drk_path = Path(f"../datafiles/{self.mouse_ID}/{self.date}/{self.date}_az_{self.mouse_ID}_imec0_1322_{area}_{ms}ms_spike_rate.npz")
            spikecount_lgt_path = Path(f"../datafiles/{self.mouse_ID}/{self.date}/{self.date}_az_{self.mouse_ID}_imec0_725_{area}_{ms}ms_spike_count.npz")
            spikecount_drk_path = Path(f"../datafiles/{self.mouse_ID}/{self.date}/{self.date}_az_{self.mouse_ID}_imec0_1322_{area}_{ms}ms_spike_count.npz") 
            # Load data
            self.fr_lgt = np.load(spikerate_lgt_path)['fr']
            self.fr_drk = np.load(spikerate_drk_path)['fr']
            self.spikes_lgt = np.load(spikecount_lgt_path)['count']
            self.spikes_drk = np.load(spikecount_drk_path)['count']
            self.pos_lgt = np.load(spikecount_lgt_path)['pos'][:,1,:]
            self.pos_drk = np.load(spikecount_drk_path)['pos'][:,1,:]

        else:    
            data_lgt_path = Path(f"../datafiles/{self.mouse_ID}/{self.date}/{self.date}_az_{self.mouse_ID}_725_{area}_{ms}ms_10cm_imec0.ap.npz")
            data_drk_path = Path(f"../datafiles/{self.mouse_ID}/{self.date}/{self.date}_az_{self.mouse_ID}_1322_{area}_{ms}ms_10cm_imec0.ap.npz") 
            # Load data
            self.fr_lgt = np.load(data_lgt_path)['fr']
            self.fr_drk = np.load(data_drk_path)['fr']
            self.spikes_lgt = np.load(data_lgt_path)['count']
            self.spikes_drk = np.load(data_drk_path)['count']
            self.pos_lgt = np.load(data_lgt_path)['pos'][:,1,:]
            self.pos_drk = np.load(data_drk_path)['pos'][:,1,:]
                    
        # Preprocess data
        self.preprocess_data()
        # Get number of units
        self.num_units = self.spikes_lgt.shape[2]
        # Show data
        self.show_data(area)

    
    def load_meta(self):
        meta_filepath = glob.glob(f"../datafiles/{self.mouse_ID}/{self.date}/*.csv")[0]
        self.get_trial_start(meta_filepath)

    
    def preprocess_data(self):
        self.spikes_lgt = np.swapaxes(self.spikes_lgt, 1,2)
        self.spikes_drk = np.swapaxes(self.spikes_drk, 1,2)
        self.fr_lgt = np.swapaxes(self.fr_lgt, 1,2)
        self.fr_drk = np.swapaxes(self.fr_drk, 1,2)
        

    def get_trial_start(self, meta_filepath):
        meta = pd.read_csv(meta_filepath, sep='\t', on_bad_lines='skip')
        step = float(meta[meta['mouse_id'] == 'rand_start_int'][self.mouse_ID].values[0])
        start = float(meta[meta['mouse_id'] == 'rand_start'][self.mouse_ID].values[0])
        end = float(meta[meta['mouse_id'] == 'rand_start_lim'][self.mouse_ID].values[0]) + step        
        self.start_locations = np.arange(start, end, step) // 10 + 1
        self.num_chunks = len(self.start_locations)


    def show_data(self, area):
        print()
        print("Data of:", self.mouse_ID)
        print("Date:", self.date)
        print("Tau:", self.tau)
        print("Reward Zone:", self.rewardzone)
        print("Areas found:", self.areas)
        print("Area loaded:", area)
        print("Number of units:", self.num_units)
        print()
        print("Trial, Time Bins, Neurons")
        print(self.spikes_lgt.shape)
        print(self.fr_lgt.shape)
        print(self.pos_lgt.shape)
        print(self.spikes_drk.shape)
        print(self.fr_drk.shape)
        print(self.pos_drk.shape)
        print()
        print('first pbin lgt:', np.unique(self.pos_lgt[:,0]))
        print([pos for pos in self.pos_lgt[:,0]])
        print()
        print('first pbin drk:', np.unique(self.pos_drk[:,0]))
        print([pos for pos in self.pos_drk[:,0]])
        print()
        print("trial start locations:", self.start_locations)
        print("number of chunks:", self.num_chunks)

    
    def clean_up(self):
        keep_only = ['mouse_ID', 'date', 'tau', 'rewardzone', 'areas', 'start_locations', 'num_chunks']
        for field in fields(self):
            if field.name not in keep_only:
                delattr(self, field.name)
