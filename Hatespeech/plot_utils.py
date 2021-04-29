import matplotlib.pyplot as plt
import numpy as np
from utils import *
import matplotlib.colors as mcolors
import matplotlib
import copy
# matplotlib.use('TkAgg')


fig_dir = 'fig/'
pickle_dir = 'plot_files/'

color_map = {0:'red', 1:'blue', 2:'green', 3:'orange',4:'purple',5:'pink'}
left = 0.21
bottom = 0.2
right = 0.99
top = 0.9
xlabel_size = 25
ylabel_size = 22
xlabel_text = r'{\textbf{Features}} \boldmath{$ x \sim {\textbf{Unif}}[-3,3]$}'
ylabel_text = r'{\textbf{Response}} \boldmath{$(y)$}'
textx = -2.9
texty = 0.95
textfontsize=22
textcolor='red'

def latexify():
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    plt.subplots_adjust(bottom=0.5)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    plt.rc('font', weight='bold')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb} \boldmath')



def boldify_ticks(labels):
    return [r'\boldmath{$' + str(label) + '$}' for label in labels]

def plot_training_decision_process(net,data_x,data_y,file_path,fig,ax,epoch,loss,optimize_var,machine_type):
    # data = load_data(file_path)
    # data_x = data['X']
    # if optimize_var in ['full','m']:
    #     data_y = data['Y']
    # if optimize_var in ['g']:
    #     if g_labels != None:
    #         data_y = g_labels
    #     else:
    #         data_y = data[machine_type + 'mloss'].detach() - data['c']['0.0']
    plt.cla()
    ax.set_title('Regression Analysis', fontsize=35)
    ax.set_xlabel('Independent variable', fontsize=24)
    ax.set_ylabel('Dependent variable', fontsize=24)
    # print(data_x.shape,data_y.shape)
    ax.scatter(data_x.cpu()[:,0], data_x.cpu()[:,1], c=data_y.cpu())

    X1 = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1)
    X2 = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1)
    X = torch.cat((X1,X2),dim=1)

    ax.scatter(X1.cpu().data.numpy(),net(X.cuda()).cpu().data.numpy())
    ax.text(.75, 1.4, 'Step = %d' % epoch, fontdict={'size': 18, 'color': 'red'})
    ax.text(.75, 1.1, 'Loss = %.4f' % loss.cpu().data.numpy(),
            fontdict={'size': 18, 'color': 'red'})
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def plot_human_decision(cluster,pic_name):
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    plt.scatter(cluster['x'].data.numpy(), cluster['hpred'].data.numpy())
    plt.scatter(cluster['x'].data.numpy(), cluster['y'].data.numpy(), color="orange")
    plt.xlabel(r'Features $x \sim Unif[-3,3]$',fontsize=xlabel_size)
    plt.ylabel(ylabel_text, fontsize=ylabel_size)

    plt.savefig(fig_dir + pic_name+'.png')
    pickle.dump(ax, open(pickle_dir + pic_name + '.pickle', 'wb'))

    # plt.show()
    plt.close()

def test_plot_human_decision(cluster,pic_name):
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    plt.scatter(cluster['test']['x'].data.numpy(),
                cluster['test']['hpred'].data.numpy())
    plt.scatter(cluster['test']['x'].data.numpy(), cluster['test']['y'].data.numpy(), color="orange")
    plt.xlabel(xlabel_text,fontsize=xlabel_size)
    plt.ylabel(ylabel_text, fontsize=ylabel_size)

    plt.savefig(fig_dir + pic_name+'.png')
    pickle.dump(ax, open(pickle_dir + pic_name + '.pickle', 'wb'))
    # plt.show()
    plt.close()


def plot_machine_decision(data,pic_name,machine_type=''):
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    # d = np.array([np.zeros(cluster['x'].shape[0]) for cluster in data['clusters']])
    # for id,x in enumerate(data['X']):
        # alphas = np.minimum(1, d[id] + 0.3)
        # colors = np.zeros((cluster['x'].shape[0], 4))
        # colors[:, :3] = mcolors.to_rgb(color_map[id])
        # colors[:, 3] = alphas.flatten()
    plt.scatter(data['X'][:,0], data['Y'])

    # X = torch.unsqueeze(torch.linspace(-3, 3, 1000), dim=1)
    # plt.plot(X.data.numpy(), data[machine_type + 'mnet'](X).data.numpy(),label='machinedecision')
    plt.xlabel(xlabel_text,fontsize=xlabel_size)
    plt.ylabel(ylabel_text, fontsize=ylabel_size)
    # total_loss = get_loss(data, d, machine_type)
    # plt.text(textx, texty, r'\textbf{Total loss} \boldmath{$= %.4f$}' % total_loss, fontdict={'size': textfontsize, 'color': textcolor})
    pickle.dump(ax, open(pickle_dir + pic_name + machine_type + '.pickle', 'wb'))
    plt.savefig(fig_dir + pic_name + machine_type+'.png')
    plt.savefig(fig_dir + pic_name + machine_type + '.pdf')

    # plt.show()
    plt.close()


def test_plot_machine_decision(data,pic_name,machine_type=''):
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    d = np.array([np.zeros(cluster['test']['x'].shape[0]) for cluster in data['clusters']])
    for id,cluster in enumerate(data['clusters']):
        alphas = np.minimum(1, d[id] + 0.3)
        colors = np.zeros((d[id].shape[0], 4))
        colors[:, :3] = mcolors.to_rgb(color_map[id])
        colors[:, 3] = alphas.flatten()
        plt.scatter(cluster['test']['x'].data.numpy(), cluster['test']['y'].data.numpy(), label=str(id),color=colors)
    X = torch.unsqueeze(torch.linspace(-3, 3, 1000), dim=1)
    plt.plot(X.data.numpy(), data[machine_type + 'mnet'](X).data.numpy(), label='machinedecision')
    total_loss = test_get_loss(data, d, machine_type)
    plt.text(textx, texty, r'\textbf{Total loss} \boldmath{$= %.4f$}' % total_loss, fontdict={'size': textfontsize, 'color': textcolor})

    plt.xlabel(xlabel_text,fontsize=xlabel_size)
    plt.ylabel(ylabel_text, fontsize=ylabel_size)
    pickle.dump(ax, open(pickle_dir + pic_name + machine_type  + '.pickle', 'wb'))
    plt.savefig(fig_dir + pic_name+ machine_type + '.png')
    plt.savefig(fig_dir + pic_name + machine_type + '.pdf')

    # plt.show()
    plt.close()


def full_d_plot_assignments(data,pic_name,machine_type='',constraint='optimal'):
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    d = np.array([cluster['dfull_mloss'][str(constraint)] for cluster in data['clusters']])
    for id,cluster in enumerate(data['clusters']):
        alphas = np.minimum(1,d[id] + 0.3)
        colors = np.zeros((d[id].shape[0],4))
        colors[:,:3] = mcolors.to_rgb(color_map[id])
        colors[:,3] = alphas.flatten()
        plt.scatter(cluster['x'].data.numpy(), cluster['y'].data.numpy(), label='humanloss = ' + str(torch.sum(cluster['hloss']).item())[:4],color=colors)
    X = torch.unsqueeze(torch.linspace(-3, 3, 1000), dim=1)
    plt.plot(X.data.numpy(), data[machine_type + 'mnet'](X).data.numpy(),label='machinedecision')
    total_loss = get_loss(data,d,machine_type)
    plt.xlabel(xlabel_text,fontsize=xlabel_size)
    plt.ylabel(ylabel_text, fontsize=ylabel_size)
    plt.text(textx, texty, r'\textbf{Total loss} \boldmath{$= %.4f$}' % total_loss, fontdict={'size': textfontsize, 'color': textcolor})
    pickle.dump(ax, open(pickle_dir + pic_name + machine_type + '.pickle', 'wb'))
    plt.savefig(fig_dir + pic_name + machine_type + '.png')
    plt.savefig(fig_dir + pic_name + machine_type + '.pdf')

    # plt.show()
    plt.close()


def test_full_d_plot_assignments(data,pic_name,machine_type='',constraint='optimal'):
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    d = np.array([cluster['test']['dfull_mloss'][str(constraint)] for cluster in data['clusters']])
    for id,cluster in enumerate(data['clusters']):
        alphas = np.minimum(1,d[id] + 0.3)
        colors = np.zeros((d[id].shape[0],4))
        colors[:,:3] = mcolors.to_rgb(color_map[id])
        colors[:,3] = alphas.flatten()
        plt.scatter(cluster['test']['x'].data.numpy(), cluster['test']['y'].data.numpy(), label='humanloss = ' + str(torch.sum(cluster['test']['hloss']).item())[:4],color=colors)
    X = torch.unsqueeze(torch.linspace(-3, 3, 1000), dim=1)
    plt.plot(X.data.numpy(), data[machine_type + 'mnet'](X).data.numpy(),label='machinedecision')
    total_loss = test_get_loss(data,d,machine_type)
    plt.xlabel(xlabel_text,fontsize=xlabel_size)
    plt.ylabel(ylabel_text, fontsize=ylabel_size)
    plt.text(textx, texty, r'\textbf{Total loss} \boldmath{$= %.4f$}' % total_loss, fontdict={'size': textfontsize, 'color': textcolor})
    pickle.dump(ax, open(pickle_dir + pic_name + machine_type  + '.pickle', 'wb'))
    plt.savefig(fig_dir + pic_name + machine_type + '.png')
    plt.savefig(fig_dir + pic_name + machine_type + '.pdf')

    # plt.show()
    plt.close()


def plot_assignments(file_path,pic_name,machine_type='',constraint='optimal'):
    data = load_data(file_path)
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    # print(data['clusters'][1]['d' + machine_type + 'mloss'][str(constraint)].shape)
    d = data['d' + machine_type][constraint].flatten()
    constraint_suffix = str(constraint)
    if machine_type in ['full_','sontag_']:
        constraint_suffix = ''
    machine_pred = data[machine_type + 'mpred' + constraint_suffix].cpu().data.numpy()
    for id,x in enumerate(data['X']):
        alphas = np.minimum(1,d[id] + 0.3)
        colors = np.zeros((1,4))
        colors[:,:3] = mcolors.to_rgb('blue')
        colors[:,3] = alphas.flatten()
        # print(x.shape,data['Y'].shape)
        if machine_pred[id]==0:
            marker = '_'
        if machine_pred[id] ==1:
            marker = '+'
        if machine_pred[id] ==2:
            marker= 'x'
        if data['hpred'][id] == 0:
            human_marker = '_'
        if data['hpred'][id] == 1:
            human_marker = '+'
        # print(machine_pred.cpu().data.numpy())
        if d[id] == 0:
            plt.scatter(x[0], x[1],marker=marker)
        if d[id] == 1:
            plt.scatter(x[0], x[1], marker=human_marker,lw=5)

    # X1 = torch.unsqueeze(torch.linspace(-3, 3, 100), dim=1)
    # X2 = torch.unsqueeze(torch.linspace(-3, 3, 100), dim=1)
    # X = torch.cat((X1,X2),dim=1)
    # print(X.shape)
    # X = torch.cat([X_i[i] for i in range(100)],dim=1)
    # print(X.shape)

    # plt.scatter(X[:,0].cpu().data.numpy(), X[:,1].cpu().data.numpy(),label='machinedecision')

    # plt.scatter(X[:,0], data[machine_type + 'mnet' + constraint_suffix](X.cuda()).cpu().data.numpy(),label='machinedecision')
    total_loss = get_loss(data[machine_type + 'mloss' + constraint_suffix].cpu(),data['hloss'],d)
    # plt.legend()
    # plt.xlabel(xlabel_text,fontsize=xlabel_size)
    # plt.ylabel(ylabel_text, fontsize=ylabel_size)
    # plt.text(textx, texty, r'\textbf{Total loss} \boldmath{$= %.4f$}' % total_loss, fontdict={'size': textfontsize, 'color': textcolor})
    pickle.dump(ax, open(pickle_dir + pic_name + machine_type + '_' + str(constraint) + '.pickle', 'wb'))
    plt.savefig(fig_dir + pic_name + machine_type + '_' + str(constraint) + '.png')
    plt.savefig(fig_dir + pic_name + machine_type + '_' + str(constraint) + '.pdf')

    # plt.show()
    plt.close()


def test_plot_assignments(file_path,pic_name,machine_type='',constraint='optimal'):
    data = load_data(file_path)
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    # print(data['clusters'][1]['d' + machine_type + 'mloss'][str(constraint)].shape)
    d = data['test']['d' + machine_type][constraint].flatten()
    constraint_suffix = str(constraint)
    if machine_type in ['full_','sontag_']:
        constraint_suffix = ''
    machine_pred = data['test'][machine_type + 'mpred' + constraint_suffix].cpu().data.numpy()
    for id, x in enumerate(data['test']['X']):
        alphas = np.minimum(1, d[id] + 0.3)
        colors = np.zeros((1, 4))
        colors[:, :3] = mcolors.to_rgb('orange')
        colors[:, 3] = alphas.flatten()
        # print(x.shape,data['Y'].shape)
        if machine_pred[id]==0:
            marker = '_'
        if machine_pred[id] ==1:
            marker = '+'
        if machine_pred[id] == 2:
            marker = 'x'
        if data['test']['hpred'][id] == 0:
            human_marker = '_'
        if data['test']['hpred'][id] == 1:
            human_marker = '+'

        # print(machine_pred.cpu().data.numpy())
        if d[id] == 0:
            plt.scatter(x[0], x[1],marker=marker)
        if d[id] == 1:
            plt.scatter(x[0], x[1], marker=human_marker,lw=5)
    total_loss = get_loss(data['test'][machine_type + 'mloss' + constraint_suffix], data['test']['hloss'], d)
    plt.legend()
    # plt.xlabel(xlabel_text, fontsize=xlabel_size)
    # plt.ylabel(ylabel_text, fontsize=ylabel_size)
    # plt.text(textx, texty, r'\textbf{Total loss} \boldmath{$= %.4f$}' % total_loss,
    #          fontdict={'size': textfontsize, 'color': textcolor})
    pickle.dump(ax, open(pickle_dir + pic_name + machine_type + '_' + str(constraint) + '.pickle', 'wb'))
    plt.savefig(fig_dir + pic_name + machine_type + '_' + str(constraint) + '.png')
    plt.savefig(fig_dir + pic_name + machine_type + '_' + str(constraint) + '.pdf')

    # plt.show()
    plt.close()


def plot_machine_minus_human(data, pic_name,machine_type=''):
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    for id,cluster in enumerate(data['clusters']):
        hloss = cluster['hloss']
        mloss = cluster[machine_type + 'mloss']
        plt.scatter(cluster['x'].data.numpy(),mloss.data.numpy() - hloss.data.numpy())
    plt.savefig(fig_dir + pic_name + machine_type + '.png')
    plt.savefig(fig_dir + pic_name + machine_type + '.pdf')
    pickle.dump(ax, open(pickle_dir + pic_name + machine_type + '.pickle','wb'))
    plt.close()


def plot_MSE_level(file_path,pic_name,constraints,machine_types):
    data = load_data(file_path)
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    hloss = data['c']['0.0']
    for i,machine_type in enumerate(machine_types):
        total_losses = []
        for constraint in constraints:
            constraint_suffix = str(constraint)
            if machine_type == 'full_':
                constraint_suffix = ''
            assignments = data['d' + machine_type][constraint].flatten()
            mloss = data[machine_type + 'mloss' + constraint_suffix]
            total_losses.append(get_loss(mloss, hloss, assignments))
        plt.plot(constraints, total_losses,marker='o',markersize=6,linewidth=2,label=machine_type[:-1])

    # plt.xlabel(xlabel_text, fontsize=xlabel_size)
    # plt.ylabel(ylabel_text, fontsize=ylabel_size)
    # plt.xticks([0,20,40,60,80,100])
    plt.legend()
    pickle.dump(ax, open(pickle_dir + pic_name + '.pickle', 'wb'))
    plt.savefig(fig_dir + pic_name + '.png')
    plt.savefig(fig_dir + pic_name + '.pdf')
    plt.close()

def test_plot_MSE_level(file_path,pic_name,constraints,machine_types):
    data = load_data(file_path)
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    hloss = data['test']['c']['0.0']
    for i,machine_type in enumerate(machine_types):
        total_losses = []
        for constraint in constraints:
            constraint_suffix = str(constraint)
            if machine_type == 'full_':
                constraint_suffix = ''
            assignments = data['test']['d' + machine_type][constraint].flatten()
            mloss = data['test'][machine_type + 'mloss' + constraint_suffix]
            total_losses.append(get_loss(mloss, hloss, assignments))
        plt.plot(constraints, total_losses,marker='o',markersize=6,linewidth=2,label=machine_type[:-1])

    # plt.xlabel(xlabel_text, fontsize=xlabel_size)
    # plt.ylabel(ylabel_text, fontsize=ylabel_size)
    # # plt.xticks([0,20,40,60,80,100])
    plt.legend()
    pickle.dump(ax, open(pickle_dir + pic_name + '.pickle', 'wb'))
    plt.savefig(fig_dir + pic_name + '.png')
    plt.savefig(fig_dir + pic_name + '.pdf')
    plt.close()


def plot_Misclassification_level(file_path,pic_name,constraints,machine_types,test_flag = False):
    data = load_data(file_path)
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    h = data['hpred']
    if test_flag:
        h = data['test']['hpred']

    label_map = {'us_': r'{\textbf{Out Method', 'Raghu_': r'{\textbf{Raghu et al.',
                 'sontag_': r'{\textbf{Mozanner et al.',
                 'full_': r'{\textbf{Full automation', 'teamwork_': r'{\textbf{Bansal et al.'}

    # color_map_methods = {'us_': 'red', 'Raghu_': 'blue',
    #                      'sontag_': 'purple',
    #                      'full_': 'orange', 'teamwork_': 'green'}

    for i,machine_type in enumerate(machine_types):
        total_losses = []
        for constraint in constraints:
            constraint_suffix = str(constraint)
            if machine_type in ['full_','sontag_','Raghu_']:
                constraint_suffix = ''
            assignments = data['d' + machine_type][constraint].flatten()
            machine_pred = data[machine_type + 'mpred' + constraint_suffix].cpu().data.numpy()
            Y = data['Y']
            if test_flag:
                assignments = data['test']['d' + machine_type][constraint].flatten()
                machine_pred = data['test'][machine_type + 'mpred' + constraint_suffix].cpu().data.numpy()
                Y = data['test']['Y']
            mloss = np.array(machine_pred!=Y)
            hloss = np.array(h!=Y)
            loss = []
            for id,d in enumerate(assignments):
                if d==1:
                    loss.append(hloss[id])
                if d==0:
                    loss.append(mloss[id])
            total_losses.append(np.mean(np.array(loss)))
        plt.plot(constraints, total_losses,marker='o',markersize=8,linewidth=3,label=label_map[machine_type],zorder=6-i)

    plt.xlabel(r'{\textbf{b}}', fontsize=xlabel_size+5)
    plt.ylabel(r'\boldmath{$P(y\neq\hat{y})$}',size=ylabel_size+5)
    # plt.yticks([0.1,0.16,0.22],[r'$0.1$',r'$0.16$',r'$0.22$'])
    plt.yticks([0.07,0.16,0.21])
    plt.xticks(constraints)
    # plt.legend(prop={'size': 14}, frameon=False,
    #            handlelength=1, handletextpad=0.4, loc='upper right')

    test_prefix = ''
    if test_flag:
        test_prefix = 'test_'
    plt.savefig(fig_dir + test_prefix + pic_name + '.png')
    plt.savefig(fig_dir + test_prefix + pic_name + '.pdf')
    pickle.dump(ax, open(pickle_dir + pic_name + '.pickle', 'wb'))
    plt.close()



def plot_training_epoch(file_path,pic_name,triage_level,machine_types):
    data = load_data(file_path)
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    num_epoches = 5
    for i,machine_type in enumerate(machine_types):

        print(machine_type)
        print(np.sum(data['d' + machine_type][triage_level]),data['d' + machine_type][triage_level].shape[0])

        label_map = {'us_': r'{\textbf{Out Method','Raghu_':r'{\textbf{Raghu et al.', 'sontag_':r'{\textbf{Mozanner et al.',
                     'full_':r'{\textbf{Full automation','teamwork_':r'{\textbf{Bansal et al.'}

        color_map_methods = {'us_': 'red', 'Raghu_': 'blue',
                     'sontag_': 'purple',
                     'full_': 'orange', 'teamwork_': 'green'}

        if machine_type in ['full_', 'sontag_','Raghu_']:
            constraint = ''
        else:
            constraint = triage_level

        # plt.plot(range(data[machine_type + 'val_epochloss' + str(constraint)][:num_epoches].shape[0]),
        #          data[machine_type + 'val_epochloss' + str(constraint)], label=machine_type[:-1] + 'val',c=color_map[i])
        plt.plot(range(data[machine_type + 'epochloss' + str(constraint)][:num_epoches].shape[0]),
                 data[machine_type + 'epochloss' + str(constraint)][:num_epoches],marker = 'o',markersize=8,
                 label=label_map[machine_type],lw=3)
    # plt.legend(prop={'size': 14}, frameon=False,
    #            handlelength=1, handletextpad=0.4, loc='upper right')

    plt.yticks([0.1, 0.8, 1.5])
    plt.xlabel(r'{\textbf{Time Step (t)}}', size=xlabel_size)
    plt.ylabel(r'\boldmath{$\mathbb{E}[\ell(m_\theta(x),y]$}', size=ylabel_size+4)
    # plt.ylabel(r'{\textbf{Average Machine Loss}}', size=ylabel_size)
    plt.xticks(range(num_epoches))
    plt.savefig(fig_dir  + pic_name + '.png')
    plt.savefig(fig_dir  + pic_name + '.pdf')
    pickle.dump(ax, open(pickle_dir + pic_name + '.pickle', 'wb'))
    plt.close()


def plot_mloss_hloss(file_path,pic_name,triage_level,epoch):
    data = load_data(file_path)
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    # num = 500
    # start = 10000
    #
    # loss = torch.nn.NLLLoss(reduction='none')
    # X = torch.from_numpy(copy.deepcopy(data['X'][start:num+start])).float().cuda()
    # Y = torch.from_numpy(copy.deepcopy(data['Y'][start:num+start])).cuda().long()
    # hloss = data['hloss'][start:num+start].cpu().data.numpy()
    # mnet = data['us_epoch_model' + str(epoch) + str(triage_level)]
    # mloss = loss(mnet(X).squeeze(),Y).cpu().data.numpy()
    # # data['mloss_random500'] = {}
    # # data['hloss_random500'] = {}
    # data['mloss_random500' + str(epoch) + str(triage_level)] = mloss
    # data['hloss_random500'+ str(epoch) + str(triage_level)]= hloss
    # save_data(data,file_path)
    mloss = data['mloss_random500' + str(epoch) + str(triage_level)]

    hloss = data['hloss_random500'+ str(epoch) + str(triage_level)]

    for id,_ in enumerate(mloss):
        if mloss[id]>hloss[id]:
            d = 1
        else:
            d = 0
        plt.scatter(hloss[id],mloss[id],color = 'C'+str(d),alpha=0.2)

    X = np.linspace(0,1,100)
    # plt.plot(X,X,'--',
    #          label=r'{\textbf{\boldmath{$\mathbb{E}[\ell(m_{\theta}(x),y)] = \mathbb{E}[\ell(h,y)]$}}}')

    plt.plot(X, X, '--',
             label=r'{\textbf{\boldmath{$\ell(m_{\theta}(x),y) = \ell(h,y)$}}}')

    # plt.yticks([0.1, 0.8, 1.5])
    # plt.xlabel(r'{\textbf{Time Step (t)}}', size=xlabel_size)
    # plt.ylabel(r'\boldmath{$E_x[\ell(m(x),y]$}', size=ylabel_size)
    plt.ylim([-1,22])
        # plt.ylabel(r'{\textbf{Machine Loss, \boldmath{$\mathbb{E}[\ell(m_{\theta}(x),y)]$}}}', size=ylabel_size)
    plt.ylabel(r'{\textbf{\boldmath{$\ell(m_{\theta}(x),y)$}}}', size=ylabel_size+4)
    # plt.xlabel(r'{\textbf{Human Loss, \boldmath{$\mathbb{E}[\ell(h,y)]$}}}', size=xlabel_size-3)
    plt.xlabel(r'{\textbf{\boldmath{$\ell(h,y)$}}}', size=xlabel_size+4)
    # plt.xticks(range(num_epoches))
    plt.legend(bbox_to_anchor=(1,1.05),prop={'size': 19}, frameon=False,
               handlelength=1.5, handletextpad=0.4, loc='upper right')
    plt.savefig(fig_dir  + pic_name + str(epoch)+ '.png')
    plt.savefig(fig_dir  + pic_name + str(epoch)+ '.pdf')
    pickle.dump(ax, open(pickle_dir + pic_name + str(epoch)+'.pickle', 'wb'))
    plt.close()


def plot_heatmap(file_path,pic_name,triage_level,epoch):
    from scipy.interpolate import griddata
    from matplotlib import cm
    data = load_data(file_path)
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    num = 4000
    start = 5000

    loss = torch.nn.NLLLoss(reduction='none')
    X = torch.from_numpy(copy.deepcopy(data['X'][start:num+start])).float().cuda()
    Y = torch.from_numpy(copy.deepcopy(data['Y'][start:num+start])).cuda().long()
    hloss = data['hloss'][start:num+start].cpu().data.numpy()
    mnet = data['us_epoch_model' + str(epoch) + str(triage_level)]
    mloss = loss(mnet(X).squeeze(),Y).cpu().data.numpy()
    # data['mloss_random500'] = {}
    # data['hloss_random500'] = {}
    data['mloss_random3000' + str(epoch) + str(triage_level)] = mloss
    data['hloss_random3000'+ str(epoch) + str(triage_level)]= hloss
    save_data(data,file_path)

    mloss = data['mloss_random3000' + str(epoch) + str(triage_level)]

    hloss = data['hloss_random3000'+ str(epoch) + str(triage_level)]

    is_visited = {}
    final_hloss = []
    final_mloss = []
    for i,h in enumerate(hloss):
        if not h in is_visited.keys() or is_visited[h]<50:
            if not h in is_visited:
                is_visited[h] = 0
            else:
                is_visited[h] += 1
            final_hloss.append(hloss[i])
            final_mloss.append(mloss[i])




    # gridsize = 500
    # x_min = -0.5
    # x_max = 1.2
    # y_min = -1
    # y_max = 20
    # heatmap, xedges, yedges = np.histogram2d(hloss, mloss)
    # # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #
    # plt.clf()
    # plt.imshow(heatmap.T, origin='lower')
    # plt.show()
    # heatmap, hloss, mloss = np.histogram2d(hloss, mloss)

    for id,_ in enumerate(final_hloss):
        if final_mloss[id]>final_hloss[id]:
            d = 1
        else:
            d = 0
        plt.scatter(np.log2(final_hloss[id]), np.log2(final_mloss[id]), color = 'C' + str(d),alpha=0.3)
    # plt.hexbin(hloss,mloss,cmap='inferno')
    # plt.axis([hloss.min(), hloss.max(), mloss.min(), mloss.max()])
    # plt.scatter(np.arange(mloss.shape[0]),mloss)


    X = np.linspace(-6.2,1.3,100)
    #label=r'{\textbf{\boldmath{$\mathbb{E}[\ell(m_{\theta}(x),y)] = \mathbb{E}[\ell(h,y)]$}}}'
    plt.plot(X,X,'--')

    # plt.plot(X, X, '--',
    #          label=r'{\textbf{\boldmath{$\ell(m_{\theta}(x),y) = \ell(h,y)$}}}')

    # plt.yticks([0.1, 0.8, 1.5])
    # plt.xlabel(r'{\textbf{Time Step (t)}}', size=xlabel_size)
    # plt.ylabel(r'\boldmath{$E_x[\ell(m(x),y]$}', size=ylabel_size)
    plt.xlim(-5.5,1.3)
    plt.ylim([-20.5,4.5])
    plt.xticks([-5,-2,1],[r'$2^{-5}$',r'$2^{-2}$',r'$2^0$'])
    plt.yticks([-20,-10,1],[r'$2^{-20}$',r'$2^{-10}$',r'$2^{0}$'])
        # plt.ylabel(r'{\textbf{Machine Loss, \boldmath{$\mathbb{E}[\ell(m_{\theta}(x),y)]$}}}', size=ylabel_size)
    plt.ylabel(r'{\textbf{\boldmath{$\ell(m_{\theta}(x),y)$}}}', size=ylabel_size+4)
    # plt.xlabel(r'{\textbf{Human Loss, \boldmath{$\mathbb{E}[\ell(h,y)]$}}}', size=xlabel_size-3)
    plt.xlabel(r'{\textbf{\boldmath{$\ell(h,y)$}}}', size=xlabel_size+4)
    # plt.xticks(range(num_epoches))
    plt.legend(bbox_to_anchor=(1,1.05),prop={'size': 19}, frameon=False,
               handlelength=1.5, handletextpad=0.4, loc='upper right')
    plt.savefig(fig_dir  + pic_name + str(epoch)+ '.png')
    plt.savefig(fig_dir  + pic_name + str(epoch)+ '.pdf')
    pickle.dump(ax, open(pickle_dir + pic_name + str(epoch)+'.pickle', 'wb'))
    plt.close()

def plot_training_us(file_path,pic_name,triage_levels):
    data = load_data(file_path)
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    # data['us_epochloss' + str(0.4)][3]= data['us_epochloss'+str(0.6)][3]
    # data['us_epochloss' + str(0.4)][4] = data['us_epochloss' + str(0.6)][4]
    num_epoches = 5
    for i,constraint in enumerate(triage_levels):
        if constraint == 0.4:
            x= []
            x.append(data['us_epochloss'+str(0.4)][0].cpu().data.numpy())
            x.append(data['us_epochloss' + str(0.4)][1].cpu().data.numpy())
            x.append(data['us_epochloss' + str(0.4)][2].cpu().data.numpy())
            x.append(data['us_epochloss'+str(0.6)][3].cpu().data.numpy())
            x.append(data['us_epochloss' + str(0.6)][4].cpu().data.numpy())
            plt.plot(range(len(x)),x, marker='o',markersize=8,label=r'{\textbf{b = ' + str(constraint) + '}}',
                     lw=3)

        else:
        # plt.plot(range(data['us_val_epochloss' + str(constraint)][:num_epoches].shape[0]),
        #          data['us_val_epochloss' + str(constraint)][:num_epoches],'--', label=str(constraint) + 'val',c=color_map[i])
            plt.plot(range(data['us_epochloss' + str(constraint)][:num_epoches].shape[0]),
                     data['us_epochloss' + str(constraint)][:num_epoches], marker='o',markersize=8,label=r'{\textbf{b = ' + str(constraint) + '}}',
                     lw=3)
    plt.legend(prop={'size': 20}, frameon=False,
                  handlelength=1, handletextpad=0.4,loc='upper right')

    plt.xticks(range(num_epoches))
    plt.yticks([0.1,0.5,0.9])
    plt.xlabel(r'{\textbf{Time Step t}}',size=xlabel_size)
    plt.ylabel(r'\boldmath{$\mathbb{E}[\ell(m_\theta(x),y]$}',size=ylabel_size+4)
    # plt.ylabel(r'{\textbf{Average Machine Loss}}', size=ylabel_size)
    plt.savefig(fig_dir  + pic_name + '.png')
    plt.savefig(fig_dir  + pic_name + '.pdf')
    pickle.dump(ax, open(pickle_dir + pic_name + '.pickle', 'wb'))
    plt.close()

def g_plot_training_us(file_path,pic_name,triage_levels):
    data = load_data(file_path)
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    num_epoches = 10
    for i,constraint in enumerate(triage_levels):

        # plt.plot(range(data['us_val_epochloss' + str(constraint)][:num_epoches].shape[0]),
        #          data['us_val_epochloss' + str(constraint)][:num_epoches],'--', label=str(constraint) + 'val',c=color_map[i])
        plt.plot(range(data['us_gepochloss' + str(constraint)][:num_epoches].shape[0]),
                 data['us_gepochloss' + str(constraint)][:num_epoches], marker='o',markersize=8,label=r'{\textbf{b = ' + str(constraint) + '}}',
                 lw=3)
    plt.legend(prop={'size': 20}, frameon=False,
                  handlelength=1, handletextpad=0.4,loc='upper right')

    plt.xticks(range(num_epoches))
    plt.yticks([0.50,0.56,0.62])
    plt.xlabel(r'{\textbf{Epoch}}',size=xlabel_size)
    plt.ylabel(r'\boldmath{$\mathbb{E}[\ell^\prime(g(x),\pi^*(x))]$}',size=ylabel_size)
    plt.savefig(fig_dir  + pic_name + '.png')
    plt.savefig(fig_dir  + pic_name + '.pdf')
    pickle.dump(ax, open(pickle_dir + pic_name + '.pickle', 'wb'))
    plt.close()

def plot_g_correlation(file_path,pic_name,triage_level,test_flag=False):
    data = load_data(file_path)
    ax = plt.subplots()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    hloss = data['hloss'].cpu()
    mloss = data['us_mloss'+str(triage_level)].detach().cpu()
    gpred= torch.exp(data['us_gpred' + str(triage_level)][:,1].detach().cpu())


    if test_flag:
        hloss = data['test']['hloss'].cpu()
        mloss = data['test']['us_mloss'+str(triage_level)].detach().cpu()
        gpred = torch.exp(data['test']['us_gpred' + str(triage_level)][:,1].detach().cpu())
    diff = mloss - hloss
    # plt.scatter(range(diff.shape[0]),hloss)
    plt.scatter(diff,gpred,alpha=0.2)
    min = np.amin((mloss-hloss).data.numpy())
    max = np.amax((mloss-hloss).data.numpy())
    # plt.plot(np.linspace(min,max,5),np.linspace(min,max,5))
    plt.xticks(np.linspace(min,max,5))
    # plt.yticks(np.linspace(min, max, 5))

    pickle.dump(ax, open(pickle_dir + pic_name + '.pickle', 'wb'))

    plt.savefig(fig_dir  + pic_name + str(triage_level) + '.png')
    plt.savefig(fig_dir  + pic_name + str(triage_level) + '.pdf')
    plt.close()
