import numpy as np
import pandas as pd
from dataclasses import dataclass, fields
import glob
import re
from pathlib import Path



data_dir = Path("../datafiles/")



@dataclass
class Mouse:
    mouse_ID: str
    dates: list = None
    sessions: dict = None
    mouse_dir: Path = None

    def __post_init__(self):
        self.mouse_dir = data_dir / self.mouse_ID
        self.find_sesh_dates()
        pass

    def find_sesh_dates(self):
        # Regex pattern to match directories with date format YYYYMMDD
        date_pattern = re.compile(r"^\d{8}$")
        self.dates = []
        # Iterate through directories in mouse_dir
        if self.mouse_dir.exists() and self.mouse_dir.is_dir():
            for item in self.mouse_dir.iterdir():
                if item.is_dir() and date_pattern.match(item.name):
                    self.dates.append(item.name)
        self.dates.sort()

        


@dataclass
class Sesh():
    mouse_ID: str
    date: str
    tau: float
    rewardzone: list
    sesh_dir: Path = None
    areas: list = None
    start_locations: np.ndarray = None
    num_startloc: int = None
    
    def __post_init__(self):
        self.sesh_dir = data_dir / self.mouse_ID / self.date
        self.find_areas()
        self.load_meta()

    @classmethod
    def from_mouse(cls, mouse: Mouse, date: str, tau: float, rewardzone: list):
        return cls(
            mouse_ID=mouse.mouse_ID,
            date=date,
            tau=tau,
            rewardzone=rewardzone
        )

    def find_areas(self):
        tau = str(int(self.tau * 1000))
        # Set folder name patterns
        pattern = re.compile(
            rf"{self.date}_az_{self.mouse_ID}.*?_([A-Za-z0-9]+)_(?:light|dark)?_?[^_]*{tau}ms"
        )
        # Find areas by matching folder name patterns
        areas = set()
        for file in self.sesh_dir.glob("*.npz"):
            match = pattern.search(file.name)
            if match:
                areas.add(match.group(1))
        self.areas = sorted(areas)

    def load_meta(self):
        # Load metadata file
        meta_filepath = glob.glob(f"../datafiles/{self.mouse_ID}/{self.date}/*.csv")[0]
        meta = pd.read_csv(meta_filepath, sep='\t', on_bad_lines='skip')
        # Extract parameters
        step = float(meta[meta['mouse_id'] == 'rand_start_int'][self.mouse_ID].values[0])
        start = float(meta[meta['mouse_id'] == 'rand_start'][self.mouse_ID].values[0])
        end = float(meta[meta['mouse_id'] == 'rand_start_lim'][self.mouse_ID].values[0]) + step        
        # Compute start location and number of chunks
        self.start_locations = np.arange(start, end, step) // 10 + 1
        self.num_startloc = len(self.start_locations)




@dataclass
class Shuffled():
    mouse_ID: str
    date: str
    tau: float
    rewardzone: list
    sesh_dir: Path
    start_locations: np.ndarray
    num_startlocs: int
    area: str
    num_reps: int = None
    num_units: int = None

    def __post_init__(self):
        self.load_data()
        self.num_reps = self.spikes_lgt_shuffled.shape[0]
        self.num_units = self.spikes_lgt_shuffled.shape[-1]

    @classmethod
    def from_sesh(cls, sesh: Sesh):
        return cls(
            mouse_ID=sesh.mouse_ID,
            date=sesh.date,
            tau=sesh.tau,
            rewardzone=sesh.rewardzone,
            sesh_dir=sesh.sesh_dir,
            start_locations=sesh.start_locations,
            num_startlocs=sesh.num_startloc,
            area="shuffled"
        )
    
    def load_data(self):
        shuffled_dir = self.sesh_dir / "shuffled"
        tau = str(int(self.tau*1000)) + "ms"
        matching_files = [f for f in shuffled_dir.glob("*.npz") if tau in f.name]

        print("Loading shuffled data...")
        for file in matching_files:
            if "light" in file.name:
                lgt_shuffled = np.load(file)
            elif "dark" in file.name:
                drk_shuffled = np.load(file)

        fr_lgt_shuffled = lgt_shuffled['fr']
        fr_drk_shuffled = drk_shuffled['fr']
        spikes_lgt_shuffled = lgt_shuffled['count']
        spikes_drk_shuffled = drk_shuffled['count']
        pos_lgt = lgt_shuffled['pos'][:, 1, :, :]
        pos_drk = drk_shuffled['pos'][:, 1, :, :]

        print("Swapping axes...")
        self.fr_lgt_shuffled = np.transpose(fr_lgt_shuffled, (3, 2, 0, 1))
        self.fr_drk_shuffled = np.transpose(fr_drk_shuffled, (3, 2, 0, 1))
        self.spikes_lgt_shuffled = np.transpose(spikes_lgt_shuffled, (3, 2, 0, 1))
        self.spikes_drk_shuffled = np.transpose(spikes_drk_shuffled, (3, 2, 0, 1))
        self.pos_lgt = np.transpose(pos_lgt, (2, 1, 0))
        self.pos_drk = np.transpose(pos_drk, (2, 1, 0))

        print(self.fr_lgt_shuffled.shape)
        print(self.spikes_lgt_shuffled.shape)
        print(self.pos_lgt.shape)
        print(self.fr_drk_shuffled.shape)
        print(self.spikes_drk_shuffled.shape)
        print(self.pos_drk.shape)




@dataclass
class Data():
    mouse_ID: str
    date: str
    tau: float
    rewardzone: list
    sesh_dir: Path
    start_locations: np.ndarray
    num_startlocs: int
    area: str
    num_units: int = None
    rep_idx: int = None
    fr_lgt: np.ndarray = None
    fr_drk: np.ndarray = None
    spikes_lgt: np.ndarray = None
    spikes_drk: np.ndarray = None
    pos_lgt: np.ndarray = None
    pos_drk: np.ndarray = None


    def __post_init__(self):
        if self.area != "shuffled":
            self.load_data()
            self.num_units = self.spikes_lgt.shape[-1]
        self.show_data()

    @classmethod
    def from_sesh(cls, sesh: Sesh, area: str):
        return cls(
            mouse_ID=sesh.mouse_ID,
            date=sesh.date,
            tau=sesh.tau,
            rewardzone=sesh.rewardzone,
            sesh_dir=sesh.sesh_dir,
            start_locations=sesh.start_locations,
            num_startlocs=sesh.num_startloc,
            area=area
        )
    
    @classmethod
    def from_shuffled(cls, sesh: Shuffled, rep_idx: int):
        return cls(
            mouse_ID=sesh.mouse_ID,
            date=sesh.date,
            tau=sesh.tau,
            rewardzone=sesh.rewardzone,
            sesh_dir=sesh.sesh_dir,
            start_locations=sesh.start_locations,
            num_startlocs=sesh.num_startlocs,
            area=sesh.area,
            num_units=sesh.num_units,
            rep_idx=rep_idx,
            fr_lgt = sesh.fr_lgt_shuffled[rep_idx],
            fr_drk = sesh.fr_drk_shuffled[rep_idx],
            spikes_lgt = sesh.spikes_lgt_shuffled[rep_idx],
            spikes_drk = sesh.spikes_drk_shuffled[rep_idx],
            pos_lgt = sesh.pos_lgt[rep_idx],
            pos_drk = sesh.pos_drk[rep_idx],
        )


    def load_data(self):
        tau = str(int(self.tau * 1000))+"ms"
        # Find all files matching the area and time bin size
        matching_files = [
            f for f in self.sesh_dir.glob("*.npz")
            if self.area in f.name and tau in f.name
        ]
        # Classify files based on key words file names
        for file in matching_files:
            # Condition in light
            if any(keyword in file.name for keyword in ['light', '725']):
                if 'spike_rate' in file.name:
                    spikerate_lgt_path = file
                elif 'spike_count' in file.name:
                    spikecount_lgt_path = file
                else:
                    spikerate_lgt_path = spikecount_lgt_path = file
            # Condition in dark
            elif any(keyword in file.name for keyword in ['dark', '1322']):
                if 'spike_rate' in file.name:
                    spikerate_drk_path = file
                elif 'spike_count' in file.name:
                    spikecount_drk_path = file
                else:
                    spikerate_drk_path = spikecount_drk_path = file
        # Load data
        self.fr_lgt = np.load(spikerate_lgt_path)['fr']
        self.fr_drk = np.load(spikerate_drk_path)['fr']
        self.spikes_lgt = np.load(spikecount_lgt_path)['count']
        self.spikes_drk = np.load(spikecount_drk_path)['count']
        self.pos_lgt = np.load(spikecount_lgt_path)['pos'][:, 1, :]
        self.pos_drk = np.load(spikecount_drk_path)['pos'][:, 1, :]
        # Swap axes to have shape (Trials, Tbins, Units)
        self.spikes_lgt = np.swapaxes(self.spikes_lgt, 1,2)
        self.spikes_drk = np.swapaxes(self.spikes_drk, 1,2)
        self.fr_lgt = np.swapaxes(self.fr_lgt, 1,2)
        self.fr_drk = np.swapaxes(self.fr_drk, 1,2)

        # 20251208: set fr to spikes / tau to ensure consistency
        self.fr_lgt = self.spikes_lgt / self.tau
        self.fr_drk = self.spikes_drk / self.tau



    def show_data(self):
        print("-" * 50)
        print("Mouse:", self.mouse_ID)
        print("Session:", self.date)
        print(f"Tbin Size: {self.tau}s or {int(self.tau * 1000)}ms")
        print("Area loaded:", self.area)
        print("Number of units:", self.num_units)
        print()
        print("{:<15} {:<30}".format("Data Type", "Trials, Tbins, Units"))
        print("{:<15} {:<30}".format("spikes_lgt", str(self.spikes_lgt.shape)))
        print("{:<15} {:<30}".format("fr_lgt", str(self.fr_lgt.shape)))
        print("{:<15} {:<30}".format("pos_lgt", str(self.pos_lgt.shape)))
        print("{:<15} {:<30}".format("spikes_drk", str(self.spikes_drk.shape)))
        print("{:<15} {:<30}".format("fr_drk", str(self.fr_drk.shape)))
        print("{:<15} {:<30}".format("pos_drk", str(self.pos_drk.shape)))
        print()
        print('First pbin in lgt:', np.unique(self.pos_lgt[:, 0]))
        print('First pbin in drk:', np.unique(self.pos_drk[:, 0]))
        print("Corrected trial start locations:", self.start_locations)
        print("Number of starting locations:", self.num_startlocs)

        


