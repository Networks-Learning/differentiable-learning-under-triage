import pickle5 as pickle
import numpy as np
import os
import sys
import torch

# def load_pickle(PATH, FIELDNAMES, FIELD):
#     dataList = []
#     with open(PATH, "rb") as input_file:
#         while True:
#             try:
#                 # Taking the dictionary instance as the input Instance
#                 inputInstance = pickle.load(input_file)
#                 # plugging it into the list
#                 dataInstance =  [inputInstance[FIELDNAMES[0]],inputInstance[FIELDNAMES[1]]]
#                 # Finally creating an example objects list
#                 dataList.append(Example().fromlist(dataInstance,fields=FIELD))
#             except EOFError:
#                 break
#
#     # At last creating a data Set Object
#     exampleListObject = Dataset(dataList, fields=data_fields)
#     return exampleListObject


def load_data(file_name):
    assert(os.path.exists(file_name+'.pkl'))
    with open(file_name + '.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def save_data(data, file_path):
    with open(file_path + '.pkl','wb') as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def delete_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)

def get_X_Y(clusters):
    X = torch.cat([cluster['x'] for cluster in clusters])
    Y = torch.cat([cluster['y'] for cluster in clusters])
    return X,Y

def test_get_X_Y(clusters):
    X_test = torch.cat([cluster['test']['x'] for cluster in clusters])
    Y_test = torch.cat([cluster['test']['y'] for cluster in clusters])
    return X_test, Y_test

def get_loss(mloss,hloss, assignments):
    # clusters = data['clusters']
    loss = []
    assert(mloss.shape==hloss.shape and hloss.shape==assignments.shape)
    for id,d in enumerate(assignments):
        if d==1:
            loss.append(hloss[id])
        else:
            # print(mloss)
            loss.append(mloss[id].item())
    loss = np.array(loss,dtype=float)
    return loss.mean()

def find_machine_samples(machine_loss, hloss,constraint):
    gprediction = machine_loss - hloss
    sorted = torch.clone(torch.argsort(gprediction))
    # print(sorted)
    # print(gprediction[sorted])
    num_machine = int((1.0-constraint) * machine_loss.shape[0])
    index = num_machine
    # if num_outsource == 0.0:
    #     index = machine_loss.shape[0]
    # else:
    #     index = -num_outsource+2
    while (index < gprediction.shape[0] and gprediction[sorted[index]] <= 0):  #add it for us_
        index += 1

    machine_list = sorted[:index]
    t = gprediction[sorted[index-1]]
    # print(gprediction,index)
    # threshold = gprediction[sorted[index]]
    # machine_list = []
    # for i in range(gprediction.shape[0]):
    #     if gprediction[i]<=threshold:
    #         machine_list.append(i)

    return machine_list.cpu().data.numpy(),t

def test_find_machine_samples(machine_loss, hloss,constraint,machine_type):
    gprediction = machine_loss - hloss
    sorted = torch.clone(torch.argsort(gprediction))
    print(gprediction[sorted])
    num_machine = int((1.0 - constraint) * machine_loss.shape[0])
    # num_machine = 0
    index = num_machine
    # if num_outsource == 0.0:
    #     index = machine_loss.shape[0]
    # else:
    #     index = -num_outsource+2
    if machine_type=='teamwork_':
        while (index < gprediction.shape[0] and gprediction[sorted[index]] <= 0):
            index += 1

    # if machine_type in ['us_']:
    #     while index < gprediction.shape[0] and torch.exp(gprediction[sorted[index]]) <= 0.35:
    #         index += 1
    # print(index)
    # print('0.3')
    machine_list = sorted[:index]
    print(len(machine_list))
    # t = gprediction[sorted[index]]
    # print(gprediction,index)
    # threshold = gprediction[sorted[index]]
    # machine_list = []
    # for i in range(gprediction.shape[0]):
    #     if gprediction[i]<=threshold:
    #         machine_list.append(i)

    return machine_list.cpu().data.numpy()
# def test_get_loss(data,assignments,machine_type=''):
#     clusters = data['clusters']
#     loss = []
#     assert(clusters.shape[0]==assignments.shape[0] and clusters[1]['test']['x'].shape[0]==assignments.shape[1])
#     for i,cluster in enumerate(clusters):
#         assignment = assignments[i]
#
#         for id,d in enumerate(assignment):
#             # print(loss)
#             if d==1:
#                 loss.append(cluster['test']['hloss'][id])
#             else:
#                 loss.append(cluster['test'][machine_type + 'mloss'][id])
#     loss = np.array(loss,dtype=float)
#     return loss.mean()*100.0

