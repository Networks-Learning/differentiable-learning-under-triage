import pickle
import sys

def save_data(data, file_path):
    with open(file_path + '.pkl','wb') as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)

def gauss(x, mu, std, coef,ycoef, offset,xoffset):
    denom = np.sqrt(2 * np.pi) * std
    print(coef,mu,std)
    return ycoef*(torch.exp(-((coef*(x-xoffset)-mu)*(coef*(x-xoffset)-mu)) / float(2 * std * std)) / denom) + offset


def sigmoid(x):
    return 1.0/(1+torch.exp(-x))

def linear(x,w,b):
    return torch.multiply(w,x) + b

def generate_X(start,end,num_samples,dim):
    X = torch.unsqueeze(torch.linspace(start, end, num_samples), dim=dim)  # x data (tensor), shape=(100, 1)
    return X


def generate_cluster(x, id,mu, std, coef,ycoef,offset,xoffset):
    if id in [2,0]:
        y = sigmoid(x)
    else:
        y = sigmoid(torch.multiply(torch.tensor([5.0]),x))
    return y
    
    
def get_hloss(file_path):
    data = load_data(file_path)
    clusters = data['clusters']
    num_clusters = len(clusters)
    loss = torch.nn.MSELoss(reduction='none')

    for id in range(num_clusters):
        y = clusters[id]['y']
        y_test = clusters[id]['test']['y']

        hpred = y + clusters[id]['hnoise']
        hloss = loss(hpred, y)

        test_hpred = y_test + clusters[id]['test']['hnoise']
        test_hloss = loss(test_hpred, y_test)

        data['clusters'][id]['hloss'] = hloss
        data['clusters'][id]['hpred'] = hpred

        data['clusters'][id]['test']['hloss'] = test_hloss
        data['clusters'][id]['test']['hpred'] = test_hpred

    save_data(data,file_path)


if __name__ == '__main__':
    file_name = 'synthetic'
    clusters = []
    num_clusters = 4
    num_samples_per_cluster = 30
    frac = 0.6
    num_train = int(frac*num_samples_per_cluster)

    mu = [0, 0, 0, 0]
    std = [0.3, 0.3, 0.3, 0.3]
    coef = [1, 1, 1, 1]
    ycoef = [1,-1,1,-4]
    offset = [-1, -0.2, 0, 1.6]
    xoffset = [0, 0.2, -0.3, -0.1]
    dim = 1



    data = {'clusters': np.array([dict.fromkeys(['x','y','hnoise','test']) for i in range(num_clusters)]),
            'mnet':None}
    x = generate_X(-3, 3, num_clusters*num_samples_per_cluster, dim)
    noise = [0.01,0.02,0.04,0.08]

    for i,id in enumerate([2,0,3,1]):
        x_cluster = x[i*num_samples_per_cluster:(i+1)*num_samples_per_cluster]
        y = generate_cluster(x_cluster, i,mu=mu[i], std=std[i],coef=coef[i],ycoef=ycoef[i],offset=offset[i],xoffset = xoffset[i])

        order = np.array(range(num_samples_per_cluster))
        np.random.shuffle(order)
        x_cluster = x_cluster[order]
        y = y[order]

        x_train = x_cluster[:num_train]
        x_test = x_cluster[num_train:]
        y_train = y[:num_train]
        y_test = y[num_train:]


        data['clusters'][id]['x'] = x_train
        data['clusters'][id]['y'] = y_train
        data['clusters'][id]['test'] = dict.fromkeys(['x','y','hnoise'])
        data['clusters'][id]['test']['x'] = x_test
        data['clusters'][id]['test']['y'] = y_test
        data['clusters'][id]['hnoise'] = torch.normal(0.0,float((int(1)+1)*noise[id]),(x_train.shape[0],1))
        data['clusters'][id]['test']['hnoise'] = torch.normal(0.0,float((int(1)+1)*noise[id]),(x_test.shape[0],1))



    save_data(data,file_name)
    get_hloss(file_name)

