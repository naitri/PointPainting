import numpy as np


class CalibrationData:
    def __init__(self, calib_file):
        calibs = read_calib_data(calib_file)
        P = calibs['P']
        
        self.P = np.reshape(P, (3, 4))
      
        Tr_cam_to_lidar = calibs["Tr_cam_to_lidar"]
        self.Tr_cam_to_lidar = np.reshape(Tr_cam_to_lidar, (3, 4))
        

        K = calibs["K"]
        self.K = np.reshape(K, (3, 3))

        D = calibs["D"]
        self.D = np.reshape(D, (1, 5))

        R0 = calibs["R0"]
        self.R0 = np.reshape(R0, (3, 3))


def read_calib_data(filepath):
    """ Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            
           
            # # The only non-float values in these files are dates, which
            # # we don't care about anyway
            try:
                key,value = line.split(':')
                data[key] = np.array([float(x) for x in value.split()])
            except:
                pass
          
    return data

