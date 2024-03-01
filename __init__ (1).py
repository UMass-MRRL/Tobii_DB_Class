import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import f_oneway, zscore
import scipy.stats

class recording_data:
    def __init__(self, participant=None, data = None, durationData = None):
        self.participant = participant
        self.data = data
        self.durationData = durationData

class tracking_data:
    def __init__(self, file_location) -> None:
        self.header = ['Alaris', 'ICU Medical', 'Baxter', 'Ivenix']
        self.metrics_dir = f"{file_location}/eyetracking Metrics.tsv"
        self.data_dir = f"{file_location}/eyetracking Data Export.tsv"
        self.file_location_dir = file_location
        self.dataset = None
        self._build_dir()
        if not os.path.exists(f'{self.file_location_dir}/Indv') :self._data_setup()

    def _build_dir(self) -> None:
        FOLDERNAME = ['Fig', 'Out', 'Repo']
        for NAME in FOLDERNAME:
            if not os.path.exists('Data/'+NAME) : os.mkdir('Data/'+NAME)


    def _data_setup(self) -> None:
        with open(self.data_dir, 'r') as f:
            header = f.readline()
            while True:
                try:
                    data = f.readline()
                    row  = data.split("\t")
                    if len(row) < 2: raise Exception("Calculation Done")
                except Exception as e:
                    print(e)
                    break
                if not os.path.exists(f'{self.file_location_dir}/Indv') :os.mkdir(f'{self.file_location_dir}/Indv') 
                FILENAME = row[5] # Participant00X
                if not os.path.exists(f"{self.file_location_dir}/Indv/{FILENAME}.tsv"): # if File exists, skip
                    with open(f"{self.file_location_dir}/Indv/{FILENAME}.tsv", 'w') as ff:
                        ff.write(header)
                with open(f"{self.file_location_dir}/Indv/{FILENAME}.tsv", 'a') as ff:
                    ff.write(data)

    def readData2(self, archive:bool=False) -> list:
        if archive and os.path.exists('data2.npy') : print("Loading data from archive.");self.dataset =  np.load('data2.npy', allow_pickle=True);return self.dataset
        dataset = []
        for file in tqdm(os.listdir('Data/Indv'), desc="Processing Data", leave=False):
            individual_data = pd.read_csv(os.path.join('Data','Indv',file), delimiter="\t", low_memory=False, header=0)
            individual_data = individual_data.loc[1:, ['Recording timestamp','Participant name', 'Computer timestamp', 'Sensor', 'Event', 'Pupil diameter filtered', 'Eye movement type', 'Gaze event duration', 'General']]
            dataset.append(recording_data(file, individual_data))
        dataset = np.array(dataset, dtype=object)
        np.save('data2.npy', dataset, True)
        self.dataset = dataset
        return dataset
    
    def getDurations(self, individual_data:list) -> pd.DataFrame:
        devices = ['Baxter', 'Ivenix', 'Alaris', 'ICU Medical']
        tasks = [
            'Interval',
            'Drug Selection ', 
            'Channel Selection ',
            'Flowrate / VTBI Selection '
            ]
        dataset = []
        for d in devices:
            subdataset = []
            for t in tasks:
                task_name = d+" "+t
                start_data = np.array(individual_data.loc[(individual_data['Event'].str.replace(" ", "") == (task_name+"Start").replace(" ", ""))|(individual_data['Event'].str.replace(" ", "").str.lower() == (task_name+"Start").replace(" ", "").lower()), 'Computer timestamp'])
                end_data = np.array(individual_data.loc[(individual_data['Event'].str.replace(" ", "") == (task_name+'End').replace(" ", ""))|(individual_data['Event'] == task_name+"End "), 'Computer timestamp'])
                if len(start_data) != len(end_data):
                    print(f"{np.array(individual_data['Participant name'])[0]} error at {task_name}")
                    print(start_data, end_data)
                    continue
                if len(start_data) == 0:
                    totalDuration = None
                else:
                    totalDuration = 0
                    for idx in range(len(start_data)):
                        start = start_data[idx]
                        end = end_data[idx]
                        totalDuration += end - start 
                # print(f"\t {task_name} \t: {totalDuration/1000000} sec")
                subdataset.append(totalDuration)
            dataset.append(subdataset)
        return pd.DataFrame(np.array(dataset), index=devices, columns=tasks)   


            

    def readData(self, archive:bool=False) -> list:
        if archive and os.path.exists('data.npy') : print("Loading data from archive.");self.dataset =  np.load('data.npy', allow_pickle=True);return self.dataset
        met = pd.read_csv(self.metrics_dir, delimiter="\t", low_memory=False, header=0)
        data = np.array(met.loc[1:, ['TOI', 'Duration_of_interval', 'Start_of_interval', 'Participant']])
        filenames = []
        dataset = []
        string = ""
        for toi, dur, start, filename in tqdm(data, desc="Processing Data", leave=False):
            if toi not in self.header: continue
            if filename not in filenames:
                filenames.append(filename)
                dataset.append([0,0,0,0,[],[],[],[], []])
            dataset[filenames.index(filename)][self.header.index(toi)] += int(dur)
            start = start
            end = int(start) + int(dur)
            rec = pd.read_csv(f"{self.file_location_dir}/Indv/{filename}.tsv", delimiter="\t", low_memory=False)
            eyeDiameter = np.array(rec.loc[(rec['Recording timestamp'] >= start*1000)&(rec['Recording timestamp'] <= end*1000)&(rec['Sensor'] == 'Eye Tracker'), ['Pupil diameter filtered']])
            dataset[filenames.index(filename)][self.header.index(toi)+4].append(eyeDiameter)
            dataset[filenames.index(filename)][-1] = filename
            string += f"\n\t {toi}\t {len(eyeDiameter)}"
        print(f"{filename} starts runs {start}~{end}. {string}")
        dataset = np.array(dataset, dtype=object)
        np.save('data.npy', dataset, True)
        self.dataset = dataset
        return dataset
    
    def plot_duration_box(self, data):
        duration = data[:,:4]
        labels = np.array([i[0] for i in data[:,-1]])
        # print(f"{len(duration[0])}, {len(labels[0])}")
        fig = plt.figure(figsize =(10, 7))
        ax = fig.add_subplot(111)
        ax.set_title('Task Duration')
        ax.set_ylabel('Duration [ms]')
        ax.set_xticklabels(self.header)
        bp = ax.boxplot(duration)
        fig.savefig('Task Duration.png')
        return bp


    def plot_individual_data(self):
        # assert(self.dataset)
        for participant_data in tqdm(self.dataset, desc="Participants", total=31, position=1, leave=False):
            diameters = participant_data[4:-1]
            fname, toi = participant_data[-1] 
            a = [num for elem in diameters for num in elem[0]]
            max_bound = float(max(a)*1.1) # 10% max margin
            min_bound = float(min(a)*0.9) # 10% min Margine
            fig = plt.figure(figsize =(10, 7))
            fig2 = plt.figure(figsize =(10, 7))
            fig3 = plt.figure(figsize =(10, 7))
            if not os.path.exists(f'Data/Fig/{fname}') : os.mkdir(f'Data/Fig/{fname}') 
            for idx, diameter in enumerate(tqdm(diameters, desc="Each Device", position=2, leave=False)):
                WINDOW = 100 
                diameter = list(np.array(pd.DataFrame([k for j in diameter for k in j]).fillna(method='ffill')).T[0])
                subfig_id = self.header.index(self.header[idx])+1
                t_score_normal = zscore(diameter)  
                moving_z = self.moving_average(t_score_normal, WINDOW) 
                raw_moving_z = self.moving_average(diameter, WINDOW) 
                normal = self.normalize(raw_moving_z)
                normal_moving = self.moving_average(normal, WINDOW) 
                # print(f"\n{len(normal)} -> {len(normal_moving)}\n")

                ax = fig.add_subplot(2,2,subfig_id)
                ax.plot(np.linspace(0,1,len(t_score_normal)), t_score_normal)    
                ax.plot(np.linspace(0,1,len(moving_z)), moving_z)   
                ax.set_title(f"Eye Diameter T-norm {self.header[idx]}")
                ax.legend(["t Score", f"Moving average k={WINDOW}"])

                ax2 = fig2.add_subplot(2,2,subfig_id)
                ax2.plot(np.linspace(0,1,len(diameter)), diameter)
                ax2.plot(np.linspace(0,1,len(raw_moving_z)), raw_moving_z)
                ax2.set_ybound(min_bound, max_bound)
                ax2.legend(["Raw", f"Moving average k={WINDOW}"])
                ax2.set_title(f"Eye Diameter {self.header[idx]}")

                ax3 = fig3.add_subplot(2,2,subfig_id)
                ax3.plot(np.linspace(0,1,len(normal)), normal)
                ax3.legend(["Raw", f"Moving average k={WINDOW}"])
                ax3.set_title(f"Normalized Eye Diameter {self.header[idx]}")
            fig.savefig(f'Data/Fig/{fname}/Diameter_Tnorm.png')
            fig2.savefig(f'Data/Fig/{fname}/Diameter.png')
            fig3.savefig(f'Data/Fig/{fname}/Diameter_norm.png')
            plt.close(fig)
            plt.close(fig2)
            plt.close(fig3)


    def average_plot(self, dataset:list, title="Average Data") -> list:
        if len(dataset[0][0]) != len(dataset[0][1]): dataset = self.resample_pupil(dataset)
        out = [[],[],[],[]]
        fig = plt.figure(figsize =(10, 7))
        ax = fig.add_subplot(111)
        # d1, d2, d3, d4 = dataset
        for idy, dd in enumerate(dataset): # for each device
            for idx in range(len(dd[0])): # for each timestamp
                out[idy].append(np.average([dd[i][idx] for i in range(len(dd))])) # averages 31 participants
        for idy in range(len(out)): # for each device
            ax.plot(np.linspace(0,1,len(out[idy])), out[idy], label=self.header[idy])    
        ax.set_title(f"{title}")
        ax.legend()
        ax.set_xlabel("Completion rate")
        ax.set_xlabel("Diameter [mm]")
        fig.savefig(f'{title}.png')
        return out


    
    def readPupil(self) -> list:
        pupil_data = self.dataset[:,4:]
        per_device = [[],[],[],[]]
        for d in tqdm(pupil_data, desc="PupilaryData", leave=False):
            for i in range(4):
                val = list(np.array(pd.DataFrame([k for j in d[i] for k in j]).fillna(method='ffill')).T[0])
                # print(len(val))
                per_device[i].append(val)
        self.pupil = per_device
        return per_device        
    

    def resample_pupil(self, data:list, size=3000) -> list:
        per_device = [[],[],[],[]]
        for idx, device_data in enumerate(tqdm(data, desc="Resampling", leave=False)):
            for i in range(len(device_data)):
                per_device[idx].append(signal.resample(device_data[i], size))
        return per_device  


    @staticmethod
    def normalize(data:list) -> list:
        min_val = min(data)
        max_val = max(data)
        out = []
        # print(data)
        for i in data:
            out.append((i-min_val)/(max_val - min_val))
        return out
    

    def normalizer(self, data:list) -> list:
        per_device = [[],[],[],[]]
        for idx, device_data in enumerate(tqdm(data, desc="Normalizing", leave=False)):
            for i in range(len(device_data)):
                per_device[idx].append(self.normalize(device_data[i]))
        return per_device  



    def getANOVA(self, data) -> None:
        print("###### ANOVA ######")
        for idx, d in enumerate(data):
            d = [self.moving_average(a, 100) for a in d]
            print(f"{self.header[idx]}\t: {f_oneway(*d)}")
        print("\n")

    def getCramer(self, data) -> None:
        print("###### Crabmer ######")
        for idx, d in enumerate(data):
            d = [self.moving_average(a, 100) for a in d]
            x2 = scipy.stats.chi2(d, correction=False)[0]
            N = np.sum(d)
            print(f"$\Xi^2$ {self.header[idx]}\t: {x2} \t Cramer's V \t : {np.sqrt((x2/N)/(min(np.array(d).shape-1)))}")

        print("\n")



    def getTukey(self, data) -> None:
        print("###### Tukey/tookey/ hsd ######")
        for idx, d in enumerate(data):
            d = [self.moving_average(a, 100) for a in d]
            zd = [self.zScore(a) for a in d]
            print(f"{self.header[idx]}\t: {scipy.stats.tukey_hsd(*zd)}")
        print("\n")

    def getKruskal(self, data) -> None:
        print("###### Kruskal  ######")
        for idx, d in enumerate(data):
            d = [self.moving_average(a, 100) for a in d]
            zd = [zscore(a) for a in d]
            print(f"{self.header[idx]}\t: {scipy.stats.kruskal(*zd)}")
        print("\n")


    def getAlexander_Govern(self, data) -> None:
        print("###### alexandergovern  ######")
        for idx, d in enumerate(data):
            d = [self.moving_average(a, 100) for a in d]
            zd = [zscore(a) for a in d]
            print(f"{self.header[idx]}\t: {scipy.stats.alexandergovern(*zd)}")
        print("\n")


    def getFligner(self, data) -> None:
        print("###### Fligner  ######")
        for idx, d in enumerate(data):
            d = [self.moving_average(a, 100) for a in d]
            zd = [zscore(a) for a in d]
            print(f"{self.header[idx]}\t: {scipy.stats.fligner(*zd)}")
        print("\n")


    def getLevene(self, data) -> None:
        print("###### levene  ######")
        for idx, d in enumerate(data):
            d = [self.moving_average(a, 100) for a in d]
            zd = [zscore(a) for a in d]
            print(f"{self.header[idx]}\t: {scipy.stats.levene(*zd)}")
        print("\n")


    def getBartlett(self, data) -> None:
        print("###### Bartlett's  ######")
        for idx, d in enumerate(data):
            d = [self.moving_average(a, 100) for a in d]
            zd = [zscore(a) for a in d]
            print(f"{self.header[idx]}\t: {scipy.stats.bartlett(*zd)}")
        print("\n")


    def getMood(self, data) -> None:
        print("###### Mood's median ######")
        for idx, d in enumerate(data):
            d = [self.moving_average(a, 100) for a in d]
            zd = [zscore(a) for a in d]
            print(f"{self.header[idx]}\t: {scipy.stats.median_test(*zd)}")
        print("\n")


    def getFriedman(self, data) -> None:
        print("######  Friedman  ######")
        for idx, d in enumerate(data):
            d = [self.moving_average(a, 100) for a in d]
            zd = [zscore(a) for a in d]
            print(f"{self.header[idx]}\t: {scipy.stats.friedmanchisquare(*zd)}")
        print("\n")


    @staticmethod
    def hasher(data, extra:str)-> str:
        return str(hash(str(np.nansum(data))+str(hash(str(extra)))))
        
        
    @staticmethod
    def moving_average(x, n):
        ret = np.cumsum(x, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    @staticmethod
    def zScore(x):
        return zscore(x)