import h5py
import glob
import os
import numpy as np
import pandas as pd
import Sources.process_list as process_list

def get_file_names():
    return [os.path.basename(f) for f in glob.glob("./Data/*tprj")]

class HDF5Handler:
    def __init__(self, file_name):
        self.n = None
        self.m = None
        self.path = file_name
        # self.file = h5py.File(f'./{file_name}')['TerapulseDocument']

    @staticmethod
    def measurements(f):
        return f['Measurements']

    @staticmethod
    def get_ref_data(measurements):
        return measurements['Ref Data']

    @staticmethod
    def get_image_data(measurements):
        return measurements['Image Data']

    @staticmethod
    def get_spectra_data(measurements):
        return measurements['Spectra Data']

    @staticmethod
    def get_keys_dic(measurements, data_type='Image Data'):
        dic = {}
        keys = measurements[data_type].keys()
        for k in keys:
            sName = measurements[data_type][k]['sample'].attrs['SampleName']
            assert len(sName) == 1, 'Len problem'
            dic[sName[0][0]] = k
        return dic

    def get_data_df( self, s_name, data_type='Image Data'):
        with h5py.File(f'{self.path}', 'r') as file:
            measurements = file['TerapulseDocument']['Measurements']
            dic = self.get_keys_dic(measurements, data_type=data_type)
            if data_type == 'Image Data':
                df_data = pd.DataFrame(self.get_image_data(measurements)[dic[s_name]]['sample']['data'][:])
                df_data = df_data.iloc[:, 3:].reset_index(drop=True).T.reset_index(drop=True)
            elif data_type == 'Spectra Data':
                df_data = pd.DataFrame(self.get_spectra_data(measurements)[dic[s_name]]['sample']['ydata'][:])
                df_data = df_data.reset_index(drop=True)
                df_data_x = pd.DataFrame(self.get_spectra_data(measurements)[dic[s_name]]['sample']['xdata'][:])
                df_data_x = df_data_x.reset_index(drop=True)
                return df_data, df_data_x
            elif data_type == 'Ref Data':
                df_data = pd.DataFrame(self.get_ref_data(measurements)[dic[s_name]]['sample']['ydata'][:])
                df_data = df_data.reset_index(drop=True)
            else:
                raise ValueError("Invalid data type. Choose 'Image Data' or 'Spectra Data'")
            return df_data

    def get_shape_df(self, s_name, data_type='Image Data'):
        with h5py.File(f'{self.path}', 'r') as file:
            measurements = file['TerapulseDocument']['Measurements']
            dic = self.get_keys_dic(measurements, data_type='Image Data')
            if data_type == 'Image Data':
                df_data = pd.DataFrame(self.get_image_data(measurements)[dic[s_name]]['sample']['line'][:])
            else:
                raise ValueError("Invalid data type. Choose 'Image Data'")
            return len(df_data), np.max(df_data), df_data

    # Only works if a reference data have been acquired, create some error messages
    def get_time_axis(self):
        with h5py.File(f'{self.path}', 'r') as file:
            time_axis = file['TerapulseDocument']['Measurements']["Ref Data"]['SPECTRA-2147483646']['sample']['xdata'][:]
            
        return time_axis[:,0]
    

def get_keys_dic(measurements, data_type='Image Data'):
    dic = {}
    keys = measurements[data_type].keys()
    for k in keys:
        sName = measurements[data_type][k]['sample'].attrs['SampleName']
        assert len(sName) == 1, 'Len problem'
        dic[sName[0][0]] = k
    return dic

def get_all_available_measurements(file_path):
    '''
    Getting the dictionnary of all measurements inside the .tprj file
    
    :param self: 
    :param file_path: path to the .trpj file
    '''

    keys = []
    with h5py.File(f'{file_path}', 'r') as file:
        measurements = file['TerapulseDocument']['Measurements']
        dic = get_keys_dic(measurements, data_type="Image Data")

        for key in dic.keys():
            keys.append(key)

    return keys

def get_project_data(file_path, name):
    '''
    Getting the THz signal data as a pandas dataframe and the time axis values
    
    :param file_path: path to file
    :param name: name of the measurement inside the file (use get_all_available_measurements() to get a list)
    '''

    Project = HDF5Handler(file_path)
    measurement = Project.get_data_df(name)
    time_axis = Project.get_time_axis()

    return measurement, time_axis

def create_3d_array_from_data(measurement):
    '''
    Docstring for create_3d_array_from_data
    
    :param measurement: dataframe with all THz values from a measurement in the project
    '''

    # Using functions for Erwan code to extract 2D slices than aranging them into a 3D array
    X,Y,x,y,nx,ny,pattern_x_arranged,pattern_y_arranged = process_list.spacial_correction(measurement[0],measurement[1])
    data_Y_arranged = process_list.data_correction(measurement, X, Y)

    # Getting informations about x and y scales
    xmin = np.min(x[0])
    xMax = np.max(x[0])
    stepx = x[0][1] - x[0][0]

    ymin = np.min(y[0])
    yMax = np.max(y[-1])
    stepy = y[1][0] - y[0][0]

    data2d_arr = data_Y_arranged.to_numpy()

    # Creating 3D array by adding all the Bscan from the Y direction
    data3d = []
    BScanY = []
    ref_value = Y[0]
    cpt = 0

    for line in data2d_arr : 
        
        if Y[cpt]!=ref_value :
            data3d.append(BScanY)
            BScanY = []
            ref_value = Y[cpt]
        BScanY.append(line)
        cpt +=1
    data3d.append(BScanY)

    data3d = np.array(data3d).transpose(2,0,1)

    return data3d, xmin, xMax,  stepx, ymin, yMax, stepy

def extract_3darray_from_measurement(file_path, name):
    '''
    Functin to extract the 3D array and axes informations directly from file using measurement name
    
    :param file_path: Description
    :param name: Description
    '''

    measurement, time_axis = get_project_data(file_path, name)
    data3d, xmin, xMax,  stepx, ymin, yMax, stepy = create_3d_array_from_data(measurement)

    return data3d, xmin, xMax,  stepx, ymin, yMax, stepy, time_axis

