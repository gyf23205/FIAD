from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from rosbags.typesys.stores.ros2_humble import std_msgs__msg__Bool as Bool

def bag2np(path):
    # create reader instance and open for reading
    with AnyReader([Path(path)]) as reader:
        connections = [x for x in reader.connections]
        stamps = [np.empty((0,)) for _ in range(len(topic_list)+1)]
        data = [[] for _ in range(len(topic_list)+1)]
        first = True

        for connection, timestamp, rawdata in reader.messages(connections=connections): # message type: 'px4_msgs/msg/VehicleLocalPosition'
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            msgtype = connection.msgtype.split('/')[-1]

            if first:
                base = timestamp
                first = False
            # print('type: ', connection.msgtype)
            # print('time: ', timestamp - base)

            if msgtype == 'TrajectorySetpoint':
                data[0].append(np.array(msg.position))
                stamps[0] = np.append(stamps[0],timestamp)
            elif msgtype == 'EstimatorInnovations':
                data[1].append(np.array([*msg.gps_hpos, msg.gps_vpos]))
                stamps[1] = np.append(stamps[1],timestamp)
            elif msgtype == 'VehicleLocalPosition':
                data[2].append(np.array([msg.x, msg.y, msg.z]))
                stamps[2] = np.append(stamps[2],timestamp)
            elif msgtype == 'SensorGps':
                data[3].append(np.array([msg.latitude_deg, msg.longitude_deg, msg.altitude_msl_m]))
                stamps[3] = np.append(stamps[3],timestamp)
            elif msgtype == 'Bool':
                data[4].append(np.array([msg.data]))
                stamps[4] = np.append(stamps[4],timestamp)
            else:
                pass

        # Take the length of the longest array as the number of samples
        lens = np.array([len(x) for x in data])
        n_topics = len(lens) # Including the flag
        longest_topic = np.argmax(lens)
        n_sample = lens[longest_topic]
        n_channels = [len(x[0]) for x in data]
        n_channel_total = sum(n_channels)

        # Align short arraies with the longest array according to time stamps
        idx = []
        for i in range(n_topics): # Number of topics including the flag
            idx.append(find_index(stamps[i], stamps[longest_topic]))

        data_exp = np.zeros((n_sample, n_channel_total))
        n_recorded = 0
        for i in range(n_topics):
            data_exp[:, n_recorded:n_recorded+n_channels[i]] = expand(data[i], idx[i], n_sample)
            n_recorded += n_channels[i]
                
    # Return data and lable seperately
    return data_exp[:, :-1], data_exp[:, -1]

def find_index(x, y):
    '''
    For each elements in x, find the index of the element in y which is cloest to the element in x.
    '''
    idx = np.zeros(x.shape, dtype=int)
    for i in range(x.shape[0]):
        diff = np.abs(y - x[i])
        idx[i] = np.argmin(diff)
    return idx

def expand(x, idx, n_sample):
    x_exp = np.empty((n_sample,*(x[0].shape)))
    first = True
    for i in range(len(x)):
        if first:
            x_exp[0:idx[i]] = x[i]
            first = False
        if i < len(x) - 1:
            x_exp[idx[i]: idx[i+1]] = x[i]
        else:
            x_exp[idx[i]:] = x[-1]
    return x_exp



if __name__ == '__main__':
    topic_list = ['TrajectorySetpoint',  'EstimatorInnovations', 'SensorGps', 'VehicleLocalPosition']
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    # Register all topics
    add_types = {}
    for topic in topic_list:
        # print(topic)
        msg_def  = Path('/home/yifan/Git/mixed_sense/work/ros2_ws/px4/msg/{}.msg'.format(topic)).read_text(encoding='utf-8')
        # register_types()
        print('px4_msgs/msg/{}'.format(topic))
        add_types.update(get_types_from_msg(msg_def, 'px4_msgs/msg/{}'.format(topic)))
        # exec(f'from rosbags.typesys.types import px4_msgs__msg__{topic} as {topic}')
    typestore.register(add_types)

    path = '/home/yifan/Git/PIAD/data/bags/rosbag2_moving1'
    data, label = bag2np(path)
    scaler = StandardScaler().fit(data)
    signals_standard = scaler.transform(data)

    # Scale to range [0,1]
    minmax_scaler = MinMaxScaler().fit(signals_standard)
    signals_scaled = minmax_scaler.transform(signals_standard)
    plt.plot(signals_scaled[:, 9:12])
    # plt.plot(label)
    # plt.legend(['traj1', 'flag'])
    plt.legend(['traj1','traj2','traj3','innoX','innoY', 'innovZ', 'posX', 'posY', 'posZ', 'gps1','gps2','gps3', 'flag'])
    plt.show()
     