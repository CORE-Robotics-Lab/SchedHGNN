import csv
import requests
from datetime import datetime
import netCDF4
import numpy as np

MODE='HARD'
# MODE='EASY'

def load_data():
    particles = {}
    largest_metric = {}
    smallest_metric = {}
    with open('common/smoothed_particles.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            particle_index = int(row["particle_index"])
            time_seconds = int(row["time_posix_seconds"])
            if not (particle_index in particles.keys()):
                particles[particle_index] = {}
            assert not (time_seconds in particles[particle_index].keys())
            particles[particle_index][time_seconds] = {}
            for key, value in row.items():
                if key == "particle_index" or key == "time_posix_seconds":
                    continue
                particles[particle_index][time_seconds][key] = float(value)
                if key in largest_metric.keys():
                    if float(value) > largest_metric[key]:
                        largest_metric[key] = float(value)
                    if float(value) < smallest_metric[key]:
                        smallest_metric[key] = float(value)
                else:
                    largest_metric[key] = float(value)
                    smallest_metric[key] = float(value)

    print("largest:", largest_metric)
    print("smallest:", smallest_metric)
    return particles


def load_data_v2():
    particles = {}
    largest_metric = {"time": 0}
    smallest_metric = {"time": 1e12}
    with open('common/paths_with_wave.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            particle_index = int(row["PathId"])
            time_seconds = int(row["Time"])
            largest_metric["time"] = max(largest_metric["time"], time_seconds)
            smallest_metric["time"] = min(smallest_metric["time"], time_seconds)
            if not (particle_index in particles.keys()):
                particles[particle_index] = {}
            assert not (time_seconds in particles[particle_index].keys())
            particles[particle_index][time_seconds] = {}
            for key, value in row.items():
                if key == "PathId" or key == "Time":
                    continue
                particles[particle_index][time_seconds][key] = float(value)
                if key in largest_metric.keys():
                    largest_metric[key] = max(float(value), largest_metric[key])
                    smallest_metric[key] = min(float(value), smallest_metric[key])
                else:
                    largest_metric[key] = float(value)
                    smallest_metric[key] = float(value)

    print("largest:", largest_metric)
    print("smallest:", smallest_metric)

    get_wave_data(north=largest_metric["Latitude"],
                  west=smallest_metric["Longitude"],
                  east=largest_metric["Longitude"],
                  south=smallest_metric["Latitude"],
                  time_start=smallest_metric["time"],
                  time_end=largest_metric["time"])

    return particles


def load_data_v6(filename='common/data_Apr_12/scenario_1_samples.csv'):
    particles = {}
    largest_metric = {"time": 0}
    smallest_metric = {"time": 1e12}
    last_particle_looked_at = 0
    last_latitude = 0
    last_longitude = 0
    last_time = 0
    time_changes = []
    max_latitude_change = 0
    max_longitude_change = 0
    is_first_row_for_this_particle = True
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            particle_index = int(row["PathId"])
            if particle_index != last_particle_looked_at:
                if particle_index in particles.keys():
                    continue
                else:
                    is_first_row_for_this_particle = True
                    last_particle_looked_at = particle_index
            time_seconds = int(row["Time"])
            largest_metric["time"] = max(largest_metric["time"], time_seconds)
            smallest_metric["time"] = min(smallest_metric["time"], time_seconds)
            if not (particle_index in particles.keys()):
                particles[particle_index] = {}
            assert not (time_seconds in particles[particle_index].keys()), "particle_index=%d, time=%d" % (
            particle_index, time_seconds)
            particles[particle_index][time_seconds] = {}
            for key, value in row.items():
                if key == "PathId" or key == "Time":
                    continue
                particles[particle_index][time_seconds][key] = float(value)
                if key in largest_metric.keys():
                    largest_metric[key] = max(float(value), largest_metric[key])
                    smallest_metric[key] = min(float(value), smallest_metric[key])
                else:
                    largest_metric[key] = float(value)
                    smallest_metric[key] = float(value)
            if not is_first_row_for_this_particle:
                max_longitude_change = max(max_longitude_change,
                                           abs(particles[particle_index][time_seconds]["Longitude"] - last_longitude))
                max_latitude_change = max(max_latitude_change,
                                          abs(particles[particle_index][time_seconds]["Latitude"] - last_latitude))
                time_changes.append(time_seconds - last_time)
            last_time = time_seconds
            last_longitude = particles[particle_index][time_seconds]["Longitude"]
            last_latitude = particles[particle_index][time_seconds]["Latitude"]
            is_first_row_for_this_particle = False

    print("largest:", largest_metric)
    print("smallest:", smallest_metric)
    print("max_longitude change", max_longitude_change)
    print("max_latitude_change", max_latitude_change)
    print("mean/std/min/max time change", np.mean(time_changes), np.std(time_changes), np.min(time_changes),
          np.max(time_changes))


    return particles


def get_wave_map_with_time(time, filename='common/WaveWatch_III_Global_Wave_Model_best.ncd.nc'):
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    precip_nc_file = filename
    nc = netCDF4.Dataset(precip_nc_file, mode='r')
    nc.variables.keys()
    all_lat = nc.variables['lat'][:].data
    all_long = nc.variables['lon'][:].data
    time_var = nc.variables['time']
    dtime = netCDF4.num2date(time_var[:], time_var.units)
    all_unix_time = np.array(list(map(lambda x: (x._to_real_datetime() - datetime(1970, 1, 1)).total_seconds(), dtime)))
    all_precip = nc.variables['Thgt'][:].data
    time_index = find_nearest(all_unix_time, time)
    return all_precip[time_index, 0, :, :]


def get_wave_map_with_time_fast(time, filename='common/WaveWatch_III_Global_Wave_Model_best.ncd.nc'):
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    precip_nc_file = filename
    nc = netCDF4.Dataset(precip_nc_file, mode='r')
    nc.variables.keys()
    all_lat = nc.variables['lat'][:].data
    all_long = nc.variables['lon'][:].data
    time_var = nc.variables['time']
    dtime = netCDF4.num2date(time_var[:], time_var.units)
    all_unix_time = np.array(list(map(lambda x: (x._to_real_datetime() - datetime(1970, 1, 1)).total_seconds(), dtime)))
    all_precip = nc.variables['Thgt'][:].data
    time_index = find_nearest(all_unix_time, time)
    return all_precip[time_index, 0, :, :]


class WaveData:
    def __init__(self, precip_nc_file='common/WaveWatch_III_Global_Wave_Model_new.ncd.nc'):

        self.nc = netCDF4.Dataset(precip_nc_file, mode='r')
        self.nc.variables.keys()
        self.all_lat = self.nc.variables['lat'][:].data
        self.all_long = self.nc.variables['lon'][:].data
        time_var = self.nc.variables['time']
        dtime = netCDF4.num2date(time_var[:], time_var.units)
        self.all_unix_time = np.array(
            list(map(lambda x: (x._to_real_datetime() - datetime(1970, 1, 1)).total_seconds(), dtime)))
        self.all_precip = self.nc.variables['Thgt'][:].data
        self.all_precip[np.isnan(self.all_precip)] = -1

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def get_wave(self, time, x, y):
        if x < 0 or y < 0 or x >= self.all_lat.shape[0] or y >= self.all_long.shape[0]:
            return -1
        else:
            return self.all_precip[time, 0, x, y]

    def get_full_wave_data(self, time_seconds):
        time_index = self.find_nearest(self.all_unix_time, time_seconds)
        return self.all_precip[time_index, 0, :, :]

    def get_adj_wave_data(self, time_seconds, long, lat):
        time_index = self.find_nearest(self.all_unix_time, time_seconds)
        lat_index = self.find_nearest(self.all_lat, lat)
        long_index = self.find_nearest(self.all_long, long + 360)
        wave_info = [self.get_wave(time_index, lat_index - 1, long_index - 1),
                     self.get_wave(time_index, lat_index - 1, long_index),
                     self.get_wave(time_index, lat_index - 1, long_index + 1),
                     self.get_wave(time_index, lat_index, long_index - 1),
                     self.get_wave(time_index, lat_index, long_index),
                     self.get_wave(time_index, lat_index, long_index + 1),
                     self.get_wave(time_index, lat_index + 1, long_index - 1),
                     self.get_wave(time_index, lat_index + 1, long_index),
                     self.get_wave(time_index, lat_index + 1, long_index + 1)]
        return wave_info

    def get_region_wave_data(self, long_start, lat_start, time_delta=1, time_duration=100, long_duration=10, lat_duration=10):
        # time is in hour unit
        time_delta_second = time_delta * 3600
        time_duration = max(int(121 / time_delta), time_duration)  # data is imported for 5 days, t_len = 121
        lat_per_grid = 0.5  # 1 grid: 30 knots ~ 55 km, 1 degree change in latitude: 110km
        long_per_grid = np.cos(long_start * np.pi / 360) * 0.5  # 1 degree change in longitude: 110*cos(longitude) km
        wave_data = np.zeros((long_duration, lat_duration, time_duration))
        time_start = self.all_unix_time[0]
        for t_i in range(time_duration):
            for lg_i in range(long_duration):
                for lt_i in range(lat_duration):
                    t = time_start + t_i * time_delta_second
                    long = lat_start + lg_i * long_per_grid
                    lat = long_start + lt_i * lat_per_grid
                    time_index = self.find_nearest(self.all_unix_time, t)
                    lat_index = self.find_nearest(self.all_lat, lat)
                    long_index = self.find_nearest(self.all_long, long + 360)
                    wave_data[lg_i, lt_i, t_i] = self.get_wave(time_index, lat_index, long_index)

        return wave_data

    def get_region_wave_data2(self, long_start, lat_start, long_end, lat_end, time_delta=1, time_duration=100):
        # time is in hour unit
        time_delta_second = time_delta * 3600
        time_duration = max(int(121 / time_delta), time_duration)  # data is imported for 5 days, t_len = 121
        lat_per_grid = 0.5  # 1 grid: 30 knots ~ 55 km, 1 degree change in latitude: 110km
        long_per_grid = np.cos(
            long_start * np.pi / 360) * 0.5  # 1 degree change in longitude: 110*cos(longitude) km
        long_duration = long_end - long_start
        lat_duration = lat_end - lat_start
        wave_data = np.zeros((long_duration, lat_duration, time_duration))
        time_start = self.all_unix_time[0]
        for t_i in range(time_duration):
            for lg_i in range(long_duration):
                for lt_i in range(lat_duration):
                    t = time_start + t_i * time_delta_second
                    long = lat_start + lg_i * long_per_grid
                    lat = long_start + lt_i * lat_per_grid
                    time_index = self.find_nearest(self.all_unix_time, t)
                    lat_index = self.find_nearest(self.all_lat, lat)
                    long_index = self.find_nearest(self.all_long, long + 360)
                    wave_data[lt_i, lg_i, t_i] = self.get_wave(time_index, lat_index, long_index)

        return wave_data

    def get_region_wave_all_data(self, long_start=-76, lat_start=30, long_end=-66, lat_end=42):
        lat_start_index = self.find_nearest(self.all_lat, lat_start)
        long_start_index = self.find_nearest(self.all_long, long_start + 360.0)
        lat_end_index = self.find_nearest(self.all_lat, lat_end)
        long_end_index = self.find_nearest(self.all_long, long_end + 360.0)
        lat_duration = lat_end_index - lat_start_index
        long_duration = long_end_index - long_start_index

        wave_data = np.zeros((lat_duration, long_duration, len(self.all_unix_time)))
        for t_id in range(len(self.all_unix_time)):
            for lt_id in range(lat_duration):
                for lg_id in range(long_duration):
                    wave_data[lt_id, lg_id, t_id] = self.get_wave(t_id, lt_id+lat_start_index, lg_id+long_start_index)
        return wave_data

    def get_region_intensity_all_data(self, long_start=-76, lat_start=30, long_end=-66, lat_end=42):

        wave_data = self.get_region_wave_all_data(long_start, lat_start, long_end, lat_end)
        intensity_data = np.ones_like(wave_data)
        if MODE == 'HARD':
            # hard
            print("Wave data is parsed into 4 levels (hard mode)")
            intensity_data[np.where(wave_data > 0)] = 2 / 3
            intensity_data[np.where(wave_data > 2)] = 1 / 3
            intensity_data[np.where(wave_data > 4)] = 0

        else:
            print("Wave data is parsed into 3 levels (easy mode)")
            # easy
            intensity_data[np.where(wave_data > 2)] = 2 / 3
            intensity_data[np.where(wave_data > 4)] = 1 / 3

        return intensity_data




    def get_random_region_wave_data(self, time_delta=1, time_duration=100, long_duration=10, lat_duration=10):
        # time is in hour unit
        time_delta_second = time_delta * 3600
        time_duration = max(int(121/time_delta), time_duration) # data is imported for 5 days, t_len = 121
        time_duration_second = time_duration * 3600
        t_len = len(self.all_unix_time)
        lat_len = len(self.all_lat)
        long_len = len(self.all_long)
        i = np.random.choice(t_len - time_duration + 1)
        j = np.random.choice(lat_len - lat_duration + 1)
        k = np.random.choice(long_len - long_duration + 1)
        time_start = self.all_unix_time[i]
        lat_start = self.all_lat[j]
        long_start = self.all_long[k]
        lat_per_grid = 0.5  # 1 grid: 30 knots ~ 55 km, 1 degree change in latitude: 110km
        long_per_grid = 0.5/np.cos(long_start * np.pi / 360) # 1 degree change in longitude: 110*cos(latitude) km

        wave_data = np.zeros((long_duration, lat_duration, time_duration))

        for t_i in range(time_duration):
            for lg_i in range(long_duration):
                for lt_i in range(lat_duration):
                    t = time_start + t_i * time_delta_second
                    long = lat_start + lg_i * long_per_grid
                    lat = long_start + lt_i * lat_per_grid
                    time_index = self.find_nearest(self.all_unix_time, t)
                    lat_index = self.find_nearest(self.all_lat, lat)
                    long_index = self.find_nearest(self.all_long, long + 360)
                    wave_data[lg_i, lt_i, t_i] = self.get_wave(time_index, lat_index, long_index)

        return wave_data



def add_wave_data(precip_nc_file='common/WaveWatch_III_Global_Wave_Model_best.ncd.nc', load_location='common/paths.csv',
                  save_location='common/paths_with_wave.csv'):
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def get_wave(time, x, y):
        if x < 0 or y < 0 or x >= all_lat.shape[0] or y >= all_long.shape[0]:
            return -1
        else:
            return all_precip[time, 0, x, y]

    # precip_nc_file = 'common/WaveWatch_III_Global_Wave_Model_best.ncd.nc'
    nc = netCDF4.Dataset(precip_nc_file, mode='r')
    nc.variables.keys()
    all_lat = nc.variables['lat'][:].data
    all_long = nc.variables['lon'][:].data
    time_var = nc.variables['time']
    dtime = netCDF4.num2date(time_var[:], time_var.units)
    all_unix_time = np.array(list(map(lambda x: (x._to_real_datetime() - datetime(1970, 1, 1)).total_seconds(), dtime)))
    all_precip = nc.variables['Thgt'][:].data
    all_precip[np.isnan(all_precip)] = -1

    with open(load_location, mode='r') as csv_file, open(save_location, mode="w") as new_data_file:
        csv_reader = csv.DictReader(csv_file)
        csv_writer = csv.writer(new_data_file, delimiter=",")
        csv_writer.writerow(["PathId", "Time", "Latitude", "Longitude", "Course", "Speed",
                             "wave_top_left", "wave_top_center", "wave_top_right",
                             "wave_middle_left", "wave_middle_center", "wave_middle_right",
                             "wave_bot_left", "wave_bot_center", "wave_bot_right"])
        line_count = 0
        for row in csv_reader:
            line_count += 1
            time_seconds = int(row["Time"])
            lat = float(row["Latitude"])
            long = float(row["Longitude"]) + 360
            time_index = find_nearest(all_unix_time, time_seconds)
            lat_index = find_nearest(all_lat, lat)
            long_index = find_nearest(all_long, long)
            to_write = [row["PathId"], row["Time"], row["Latitude"], row["Longitude"], row["Course"], row["Speed"],
                        get_wave(time_index, lat_index - 1, long_index - 1),
                        get_wave(time_index, lat_index - 1, long_index),
                        get_wave(time_index, lat_index - 1, long_index + 1),
                        get_wave(time_index, lat_index - 1, long_index - 1),
                        get_wave(time_index, lat_index - 1, long_index),
                        get_wave(time_index, lat_index - 1, long_index + 1),
                        get_wave(time_index, lat_index + 1, long_index - 1),
                        get_wave(time_index, lat_index + 1, long_index),
                        get_wave(time_index, lat_index + 1, long_index + 1)]
            csv_writer.writerow(to_write)
            new_data_file.flush()


def get_wave_data(north=77.5000, west=0.0000, east=359.5000, south=-77.5000, time_start='2020-01-01T00:00:00Z', time_end='2020-01-09T23:27:42Z'):
    response = requests.get(
        "https://pae-paha.pacioos.hawaii.edu/thredds/ncss/ww3_global/WaveWatch_III_Global_Wave_Model_best.ncd",
        params={
            "var": "Thgt",
            "north": str(north),
            "west": str(west),
            "east": str(east),
            "south": str(south),
            "disableProjSubset": "on",
            "horizStride": "1",
            "time_start": time_start,
            "time_end": time_end,
            "timeStride": "1",
            "vertCoord": "",
            "addLatLon": "false"
        })

    print(response.url)
    with open("common/WaveWatch_III_Global_Wave_Model_new.ncd.nc", "wb") as f:
        f.write(response.content)


def load_data_basic():
    dest_file = 'smoothed_particles.csv'
    with open(dest_file, 'r') as dest_f:
        data_iter = csv.reader(dest_f,
                               quotechar='"')
        data = [data for data in data_iter]
    return np.asarray(data)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    Waves = WaveData(precip_nc_file="common/WaveWatch_III_Global_Wave_Model_new.ncd.nc")
    all_precip = Waves.all_precip
    precip_shape = all_precip.shape
    #precip_is_nan = (all_precip == -1)#np.count_nonzero(all_precip == -1)
    # for i in range(10):
    #     data = Waves.get_random_region_wave_data(time_duration=1)[:, :, 0].reshape(-1)
    #     fig, ax = plt.subplots(figsize=(10, 7))
    #     ax.hist(data)
    #     fig.savefig('hist_%i.png'%i)

    # for i in range(len(precip_is_nan)):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     precip_is_nan_time_x, precip_is_nan_time_y = precip_is_nan[i, 0, :, :]
    #     hist, xedges, yedges = np.histogram2d(precip_is_nan_time_x, precip_is_nan_time_y)#, bins=4, range=[[0, 4], [0, 4]])
    #     # Construct arrays for the anchor positions of the 16 bars.
    #     xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    #     xpos = xpos.ravel()
    #     ypos = ypos.ravel()
    #     zpos = 0
    #
    #     # Construct arrays with the dimensions for the 16 bars.
    #     dx = dy = 0.5 * np.ones_like(zpos)
    #     dz = hist.ravel()
    #
    #     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    #     plt.show()
    #     fig.save_fig('3d_hist_%d.png'%i)
    # np.set_printoptions(threshold=np.inf)
    #all_precip = Waves.all_precip.reshape(-1)#, precip_shape[-2]*precip_shape[-1])
    wave = Waves.get_region_wave_all_data(-76, 30, -66, 42)
    # wave_is_nan = (wave==-1)
    # x1_is_nan, y1_is_nan = np.where(wave_is_nan[:, :, 0])
    # for i in range(wave.shape[-1]):
    #     x_is_nan, y_is_nan = np.where(wave_is_nan[:, :, i])
    #     print (np.all(x1_is_nan == x_is_nan), np.all(y1_is_nan == y_is_nan))
    from matplotlib.ticker import PercentFormatter
    #fig, ax = plt.subplots(figsize=(10, 7))
    wave_flatten = wave.reshape(-1)
    plt.hist(wave_flatten, weights=np.ones(len(wave_flatten)) / len(wave_flatten))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig('histogram_selected_region.png')


    intensity = np.ones_like(wave)
    intensity[np.where(wave > 0)] = 2/3
    intensity[np.where(wave > 2)] = 1/3
    intensity[np.where(wave > 4)] = 0




    #np.set_printoptions(threshold=np.inf)
    #print(intensity)
    # fig, ax = plt.subplots(figsize=(10, 7))
    # ax.hist(intensity.reshape(-1))
    # fig.savefig('histogram_intensity.png')

        #print(i, np.where(wave_is_nan[:, :, i]))
        #print(i, np.any(wave_is_nan[:, :, i].reshape(-1)), np.all(wave_is_nan[:, :, i].reshape(-1)))

    #print(wave.shape)

    # Generate npz file


    # wave = wave[wave!=-1]
    # fig, ax = plt.subplots(figsize=(10, 7))
    # ax.hist(wave.reshape(-1))
    # fig.savefig('histogram.png')
    # print(wave)
    #fig, ax = plt.subplots(figsize=(10, 7))
    #ax.hist(all_precip[:1000000])
    # for i in range(wave.shape[-1]):
    #     plt.imshow(wave[:, :, i])
    #     plt.savefig('plots/wave_%d.png'%i)
    #fig.savefig('histogram.png')
    #info = Waves.get_adj_wave_data(100, 100, 100)
    #print(info)
    #data = Waves.get_region_wave_data(100, 100, 100)
    #print(data)
    #all_precip = Waves.all_precip
    #print(all_precip.shape)
    # print(np.count_nonzero(all_precip == -1) / len(all_precip))  # 38.49% are originally nan

    # for i in range(10):
    #     #print(
    #     Waves.get_random_region_wave_data()
    #print(Waves.get_full_wave_data(100))
    #print(get_wave_map_with_time(100, filename="common/WaveWatch_III_Global_Wave_Model_new.ncd.nc"))
    #print(get_wave_map_with_time_fast(100, filename="common/WaveWatch_III_Global_Wave_Model_new.ncd.nc"))

