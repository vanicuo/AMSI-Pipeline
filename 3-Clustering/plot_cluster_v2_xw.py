import matplotlib.pyplot as plt
import mne
import torch
import numpy as np


class PlotCluster():

    def __init__(self,info,labels,peaks):

        self.info = mne.pick_info(info,
                                      sel=mne.pick_types(info, meg=True))

        self.info_mag = mne.pick_info(self.info,
                                      sel=mne.pick_types(self.info, meg='mag'))
        self.info_grad = mne.pick_info(self.info,
                                       sel=mne.pick_types(self.info, meg='grad'))
        self.labels = labels
        self.peaks = peaks

    def plot_cluster(self,rowPlot=15, colPlot = 10, is_show = False):
        '''
        plot all the cluster
        :param cls: labels matrix, like self.labels
        :param rowPlot: the number of clusters
        :param colPlot: the samples from the cluster
        :return:
        '''
        cls = self.labels
        peaks = self.peaks
        fig = plt.figure(num=23, figsize=(27.5, 15), clear=True, dpi=100)
        rowColPlot = [torch.stack([torch.ones(colPlot) * i + 1, torch.arange(0, colPlot)]).long().t()
                      for i in torch.arange(rowPlot)]
        # Plot Brod Line
        ax = plt.Axes(fig, [1 / colPlot * rowColPlot[0][1][1] - (1 / colPlot * rowColPlot[0][1][1] -
                                                                 (0.95 / colPlot / 2.1 + 0.95 / colPlot / 2.3)), -1,
                            (1 / colPlot * rowColPlot[0][1][1] - (0.95 / colPlot / 2.1 + 0.95 / colPlot / 2.3)), 3])
        fig.add_axes(ax)
        ax.plot([0, 0], [0, 1], color='r')
        ax.set_axis_off()
        for J in range(min(torch.tensor([np.where(cls == x)[0].shape[0]
                                         for x in range(np.unique(cls).shape[0])]).max(), colPlot - 1)):
            # Plot Line
            ax = plt.Axes(fig, [1 / colPlot * rowColPlot[0][J + 1][1] +
                                (0.95 / colPlot / 2.1 + 0.95 / colPlot / 2), -1,
                                (0.95 / colPlot - (0.95 / colPlot / 2.1 + 0.95 / colPlot / 2)), 3])
            fig.add_axes(ax)
            ax.plot([0, 0], [0, 1], linestyle='--', color='k')
            ax.set_axis_off()
        for (I, x) in enumerate(range(np.unique(cls).shape[0])):
            Topo_Show = peaks[np.where(cls == x)[0]]
            Topo_Show = Topo_Show.reshape(-1) if len(Topo_Show.shape) == 1 else Topo_Show
            TopoMean_Show = Topo_Show.mean(axis=0)
            # Plot mean Mag
            ax = plt.Axes(fig, [1 / colPlot * rowColPlot[I][0][1], 1 - 1 / rowPlot * rowColPlot[I][0][0],
                                0.95 / colPlot / 2.1, 0.95 / rowPlot])
            ax.set_axis_off()
            fig.add_axes(ax)
            mne.viz.plot_topomap(TopoMean_Show[mne.pick_types(self.info, meg='mag')], self.info_mag, axes=ax, show=False)
            ax.text(ax.get_xlim()[0], ax.get_ylim()[1] - 0.01,
                    'Cls: ' + str(I) + ';  ' + 'Num: ' + str(Topo_Show.shape[0]))
            # Plot mean Grad
            ax = plt.Axes(fig, [1 / colPlot * rowColPlot[I][0][1] + 0.95 / colPlot / 2,
                                1 - 1 / rowPlot * rowColPlot[I][0][0],
                                0.95 / colPlot / 2.3, 0.95 / rowPlot])
            ax.set_axis_off()
            fig.add_axes(ax)
            mne.viz.plot_topomap(TopoMean_Show[mne.pick_types(self.info, meg='grad')], self.info_grad, axes=ax, show=False)
            for (J, y) in enumerate(range(min(Topo_Show.shape[0], colPlot - 1))):

                # Plot Mag
                ax = plt.Axes(fig, [1 / colPlot * rowColPlot[I][J + 1][1], 1 - 1 / rowPlot * rowColPlot[I][J + 1][0],
                                    0.95 / colPlot / 2.1, 0.95 / rowPlot])
                ax.set_axis_off()
                fig.add_axes(ax)
                mne.viz.plot_topomap(Topo_Show[J][mne.pick_types(self.info, meg='mag')], self.info_mag, axes=ax, show=False)
                # Plot Grad
                ax = plt.Axes(fig, [1 / colPlot * rowColPlot[I][J + 1][1] + 0.95 / colPlot / 2,
                                    1 - 1 / rowPlot * rowColPlot[I][J + 1][0],
                                    0.95 / colPlot / 2.2, 0.95 / rowPlot])
                ax.set_axis_off()
                fig.add_axes(ax)
                mne.viz.plot_topomap(Topo_Show[J][mne.pick_types(self.info, meg='grad')], self.info_grad, axes=ax, show=False)
        if is_show:
            plt.show()
        return fig

    def plot_one_category(self,sel_cls):
        '''
        plot one class you appoint
        :param cls: labels
        :param sel_cls: input the class you want to plot
        :return:
        '''
        cls = self.labels
        peak = self.peaks
        Topo_Show = peak[np.where(cls == sel_cls)[0]]
        Topo_Show = Topo_Show.reshape(-1) if len(Topo_Show.shape) == 1 else Topo_Show
        TopoMean_Show = Topo_Show.mean(axis=0)
        Topo_Show = np.concatenate([TopoMean_Show.reshape(1, -1), Topo_Show], axis=0)

        fig = plt.figure(num=23, figsize=(27.5, 15), clear=True, dpi=100)
        colPlot = 5
        rowPlot = int(np.ceil((Topo_Show.shape[0] + 1) / colPlot))
        rowColPlot = torch.cat([torch.stack([torch.ones(colPlot) * i + 1, torch.arange(0, colPlot)]).long().t()
                                for i in torch.arange(rowPlot)])
        # Plot Line
        for J in range(colPlot - 1):
            ax = plt.Axes(fig, [1 / colPlot * np.unique(rowColPlot)[J] +
                                (0.95 / colPlot / 2.1 + 0.95 / colPlot / 2), -1,
                                (0.95 / colPlot - (0.95 / colPlot / 2.1 + 0.95 / colPlot / 2)), 3])
            fig.add_axes(ax)
            ax.plot([0, 0], [0, 1], linestyle='--', color='k')
            ax.set_axis_off()
        for (I, x) in enumerate(range(Topo_Show.shape[0])):
            # Plot mean Mag
            ax = plt.Axes(fig, [1 / colPlot * rowColPlot[I][1], 1 - 1 / rowPlot * rowColPlot[I][0],
                                0.95 / colPlot / 2.1, 0.95 / rowPlot])
            ax.set_axis_off()
            fig.add_axes(ax)
            mne.viz.plot_topomap(Topo_Show[I, mne.pick_types(self.info, meg='mag')], self.info_mag, axes=ax, show=False)
            if I == 0:
                ax.text(ax.get_xlim()[0], ax.get_ylim()[1] - 0.01,
                        'Cls: ' + str(I) + ';  ' + 'Num: ' + str(Topo_Show.shape[0]))
            # Plot mean Grad
            ax = plt.Axes(fig, [1 / colPlot * rowColPlot[I][1] + 0.95 / colPlot / 2,
                                1 - 1 / rowPlot * rowColPlot[I][0],
                                0.95 / colPlot / 2.3, 0.95 / rowPlot])
            ax.set_axis_off()
            fig.add_axes(ax)
            mne.viz.plot_topomap(Topo_Show[I, mne.pick_types(self.info, meg='grad')], self.info_grad, axes=ax, show=False)
        plt.show()
        return fig

    def plot_one(self,sel_cls,arr_epo):

        epochs = arr_epo.average()

def plot_map(cluster_epochs,show= False):
    '''
    绘某一个类叠加平均后的波形，拓扑图，以及绘制样本
    :param cluster_epochs: 某个类变成epochArray格式
    :param show: 是否绘制图片 dtype= bool default = Flase
    :return: fig 图像的fig类
    '''

    info = cluster_epochs.info
    # cluster_epochs.plot(butterfly=False, n_channels=50)
    evoked = cluster_epochs.average()

    grad = evoked.pick_types('grad')
    grad_data = grad.data
    evoked = cluster_epochs.average()
    mag = evoked.pick_types('mag')
    mag_data = mag.data

    time_point = np.arange(0,0.8,0.001)


    fig = plt.figure(num=23, figsize=(30, 15),clear=True, dpi=100)
    ax1 = fig.add_axes([0.035,0.78,0.355,0.18])
    ax2 = fig.add_axes([0.035,0.50,0.355,0.18])
    ax3 = fig.add_axes([0.01,0.08,0.2,0.3])
    ax4 = fig.add_axes([0.2,0.08,0.2,0.3])
    ax5 = fig.add_axes([0.35,0,0.1,1])

    ax5.plot([0, 0], [0, 1], color='r',linewidth='2.5')
    ax5.set_axis_off()


    ax3.set_axis_off()
    mne.viz.plot_topomap(mag_data[:,400],mag.info,axes=ax3,show= False)
    mne.viz.plot_topomap(grad_data[:,400],grad.info,axes=ax4,show= False)
    for i in range(102):
        ax1.plot(time_point,mag_data[i,:]*10e14,color='k',linestyle='-',linewidth='0.5')
    ax1.set_xlim(0,0.8)
    ax1.set_title('Magnetometers (102 channels)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('fT')

    for i in range(204):
        ax2.plot(time_point,grad_data[i,:]*10e12,color='k',linestyle='-',linewidth='0.5')
    ax2.set_xlim(0,0.8)
    ax2.set_title('Gradiometers (204 channels)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('fT/cm')

    ax3.set_title('  Mag-Topomap')
    ax4.set_title('  Grag-Topomap')

    for J in range(2):
        ax = plt.Axes(fig, [0.35+(J+1)*0.2,0,0.1,1])
        fig.add_axes(ax)
        ax.plot([0, 0], [0, 1], linestyle='--', color='k')
        ax.set_axis_off()

    cluster = cluster_epochs.get_data()
    cluster_n = cluster.shape[0]
    if cluster_n <=12:
        cluster_num = cluster_n
    else:
        cluster_num = 12


    for i in range(cluster_num):

        j = i // 4

        k = i - j * 4

        ax = plt.Axes(fig, [0.37+j*0.2,0.78-k*0.24,0.17,0.17])
        ax.set_axis_off()
        fig.add_axes(ax)
        mne.viz.plot_topomap(cluster[i,mne.pick_types(info, meg='mag'),400],mag.info, axes=ax, show=False)
        if i == 0:
            ax.text(ax.get_xlim()[0], ax.get_ylim()[1] - 0.01,
                   'Num: ' + str(cluster_n),fontsize = 30)

        ax = plt.Axes(fig, [0.46+j*0.2,0.78-k*0.24,0.17,0.17])
        ax.set_axis_off()
        fig.add_axes(ax)
        mne.viz.plot_topomap(cluster[i,mne.pick_types(info, meg='grad'),400],grad.info, axes=ax, show=False)


    # grad_dat = cluster_epochs.pick_types('grad')

    # info = cluster_epochs.info
    if show == True:
        plt.show()

    return fig

if __name__ == '__main__':

    path = "/data2/epoc_5.fif"
    cluster_epochs = mne.read_epochs(fname=path, preload=True, verbose=False)
    plot_map(cluster_epochs)