from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_types_from_msg, register_types
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def bag2np(path):
    # create reader instance and open for reading
    with AnyReader([Path(path)]) as reader:
        connections = [x for x in reader.connections]
        # i           = 0
        time = []
        # Get timestamps for VehicleLocalPosition and EstimatorInnovations. SensorGps has different timestamps.
        for connection, timestamp, rawdata in reader.messages(connections=connections): # message type: 'px4_msgs/msg/VehicleLocalPosition'
            msgtype = connection.msgtype.split('/')[-1]
            print(msgtype)
            if msgtype=='Bool':
                print(msgtype)
            # exec('global msg; msg = reader.deserialize(rawdata, {}.__msgtype__)'.format(msgtype))
            
            # if msgtype == 'VehicleLocalPosition':
            #     time.append(msg.timestamp)

        

        # time = np.array(time, dtype=int)
        # n_sample = time.shape[0]
        # data = np.zeros((n_sample, 8)) # data: [VehicleLocalPosition, EstimatorInnovations, SensorGps]
        # label = np.empty((n_sample))
        # temp_gps = np.zeros((3,))
        # for connection, timestamp, rawdata in reader.messages(connections=connections):
        #     msgtype = connection.msgtype.split('/')[-1]
        #     exec('global msg; msg = reader.deserialize(rawdata, {}.__msgtype__)'.format(msgtype))
            # if msgtype == 'TrajectorySetpoint':
            #     temp_gps = [msg.latitude_deg, msg.longitude_deg, msg.altitude_msl_m]
            #     # print(msg.latitude_deg, msg.longitude_deg, msg.altitude_msl_m)
            # elif msgtype == 'VehicleLocalPosition':
            #     idx = np.where(time==msg.timestamp)[0]
            #     data[idx,0:3] = [msg.x, msg.y, msg.z]
            #     if data[idx, 5] == 0:
            #         data[idx, 5:8] = temp_gps
            # elif msgtype == 'EstimatorInnovations':
            #     idx = np.where(time==msg.timestamp)[0]
            #     data[idx,3:5] = msg.gps_hpos
            #     if data[idx, 5] == 0:
            #         data[idx, 5:8] = temp_gps
            # #     temp
            # elif msgtype == 'SensorGps':
            # elif msgtype == 'Bool':
            # else:
            #     pass
            # print(f'{msgtype}:{msg.timestamp-1716306678425596}')
            # print(timestamp-1716306678425596)
            # i += 1
    return data



if __name__ == '__main__':
    # topic_list = ['VehicleOdometry', 'TrajectorySetpoint', 'VehicleAttitudeSetpoint',
    #                'VehicleCommand', 'SensorGps', 'VehicleTrajectoryWaypoint', 'FailsafeFlags', 
    #                'VehicleAttitude', 'VehicleGlobalPosition',  'VehicleLocalPosition',  'VehicleStatus']
    topic_list = ['TrajectorySetpoint',  'EstimatorInnovations', 'SensorGps', 'VehicleLocalPosition']

    # Register all topics
    for topic in topic_list:
        custom_msg_path = Path('/home/yifan/Git/mixed_sense/work/ros2_ws/px4/msg/{}.msg'.format(topic))
        msg_def         = custom_msg_path.read_text(encoding='utf-8')
        register_types(get_types_from_msg(msg_def, 'px4_msgs/msg/{}'.format(topic)))
        exec(f'from rosbags.typesys.types import px4_msgs__msg__{topic} as {topic}')

    path = '/home/yifan/Git/PIAD/data/bags/rosbag2_moving1'
    data = bag2np(path)
    scaler = StandardScaler().fit(data)
    signals_standard = scaler.transform(data)

    # Scale to range [0,1]
    minmax_scaler = MinMaxScaler().fit(signals_standard)
    signals_scaled = minmax_scaler.transform(signals_standard)
    plt.plot(signals_scaled)
    plt.legend(['x','y','z','inno1','inno2','gps1','gps2','gps3'])
    plt.show()
     