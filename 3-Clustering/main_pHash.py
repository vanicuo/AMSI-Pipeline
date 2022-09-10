import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import mne
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial import distance as D
import dip_fitting as dp
from plot_cluster_v2_xw import PlotCluster,plot_map
import pickle, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
device = torch.device("cuda")
# device = torch.device("cpu")
class PhashClustering():

    def __init__(self,data,info,peak_pos=400,peak_range=5,grad_threshold=0,img_resize=(48,48),
                 gof_threshold_low = 0.3,gof_threshold_high=0.5,verbose=False):

        '''
        :param data: shape=(N,306,800);波形数据,默认取中间时刻点(-5，+5)的306通道绘制拓扑图;type=np.array,;dtype=float32
        :param info: fif头文件读取的info信息;type=class;
        :param peak_pos: peak点的位置，default= 400
        :param peak_range: peak点周围多少距离求均值,default=5
        :param grad_threshold: 梯度拓扑图对距离矩阵的影响,default=0
        :param img_resize:  拓扑图缩小多大尺寸再进行pHash编码 type=tuple; default = (48,48)
        :param gf_threshold_before: 前期fit_goodness滤除样本指标;default=0.3
        :param gf_threshold_after: 后期it_goodness滤除样本指标;default=0.5
        '''

        self.data = data
        self.info = mne.pick_info(info,
                                      sel=mne.pick_types(info, meg=True))
        self.peak = self.data[:,:,(peak_pos-peak_range):(peak_pos+peak_range)].mean(axis=-1)
        self.info_mag = mne.pick_info(self.info,
                                      sel=mne.pick_types(self.info, meg='mag'))
        self.info_grad = mne.pick_info(self.info,
                                       sel=mne.pick_types(self.info, meg='grad'))
        self.verbose = verbose
        self.sample_num = self.data.shape[0]               #存放样本个数
        self.grad_threshold = grad_threshold                #梯度拓扑图对距离矩阵的影响
        self.img_resize = img_resize                        #将拓扑图缩小成多大
        self.gof_threshold_low = gof_threshold_low      #前期低阈值滤除样本
        self.gof_threshold_high = gof_threshold_high        #后期高阈值滤除样本
        self.__init_para()                                  #调用函数得到距离矩阵等参数


    def phash_code(self,img,hash_size=(8,8)):

        '''
        encode topography of peak by using phash algorithm
        :param img: img from mne.viz.plot_topomap()
        :param phash_size: the size of phash; type=tuple; len = 2; default = (8,8)
        :return: phash features, pHash编码后的64维度特征
        '''

        img_resize = self.img_resize
        img = img[::-1]
        hash_offset = hash_size[0]
        hash_code = np.zeros(hash_size)
        img_x = np.nan_to_num(img)
        min_value = np.min(img_x)
        img = np.nan_to_num(img, nan=min_value)
        img = cv2.resize(img, img_resize)
        dct_img = cv2.dct(img)
        dct_feature = dct_img[:hash_offset, :hash_offset]
        dct_mean = np.mean(dct_feature)
        hash_code[dct_feature >= dct_mean] = 1.0
        feature_vec_size = hash_size[0] * hash_size[1]
        hash_code = np.resize(hash_code,feature_vec_size)
        return hash_code


    def get_peak_all(self,channel_type):

        '''
        select peak channel(mag or grad)
        :param channel_type: type=str input='mag' or 'grad'
        :return: peak_point，从得到的peak点的306通道内提取channel_type指定的通道类型
        如果channel_type='mag', peak_time.shape=(102,); channel_type='grad',
        peak_time.shape=(204,)
        '''

        peak = self.peak
        info = self.info
        #
        if channel_type == 'mag':
            peak_point = peak[:, mne.pick_types(info, meg='mag')]
        elif channel_type =='grad':
            peak_point = peak[:, mne.pick_types(info, meg='grad')]

        #
        gfp = np.sqrt(np.sum(np.square(peak_point)) / peak_point.shape[0])
        if gfp != 0:
            peak_point = peak_point / gfp

        return peak_point


    def __phash_process(self):

        '''
        pHash encode for imgs
        :param img_size: resize img shape from 64*64 to img_resize, type=tuple
        :return: get participant's phash (mag_phash & grad_phash)
        '''
        peak_mag = self.get_peak_all('mag')
        peak_grad = self.get_peak_all('grad')

        # 便利一个被试的suoyou sample 调用phash_code 并将每个sample进行hash编码,
        # 编码按照mag通道和grad通道进行

        if self.verbose == True:
            print('-----pHash Code-----\n')

        for i in range(self.sample_num):
            _, _, _, img_mag = mne.viz.re_topomap.plot_topomap(peak_mag[i,:], self.info_mag)

            hash_code_mag = self.phash_code(img_mag)
            self.hash_codes_mag.append(hash_code_mag)

            _, _, _, img_grad = mne.viz.re_topomap.plot_topomap(peak_grad[i,:], self.info_grad)

            hash_code_grad = self.phash_code(img_grad)
            self.hash_codes_grad.append(hash_code_grad)

        plt.clf()

        return self.hash_codes_mag, self.hash_codes_grad


    def __dip_feature(self,samp_win=5,init_res=10,inter=4,
                      devices=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        '''
        进行dipole fit得到三个参数 pos ori goodness of fitting
        :param SampWin: default=5
        :param initRes: default=10
        :param inter:   default=4
        :param devices:
        :return:
        '''
        print(device)

        data = self.data
        info = self.info

        if self.verbose == True:
            print('-----Dipole Fitting-----\n')


        pos, ori, fit_goodness = dp.dip_do_fit_PCA(data, info, SampWin=samp_win,
                                                   initRes=init_res, inter=inter, device=devices)
        self.pos = pos.numpy()
        self.ori = ori.numpy()
        self.fit_goodness = fit_goodness.numpy()


    def hamming_distance(self, x, y):

        '''
        calculate hamming distance matrix
        :param x: input x
        :param y: input y
        :return: distance matrix
        '''

        xor = np.logical_xor(x,y)
        distance = xor.sum().astype(np.float32)

        return distance


    def __distance_matrix(self):

        '''
        对哈希编码的每个样本构造距离矩阵，矩阵分为两个，一个是mag矩阵另一个是grad矩阵，总的矩阵为二者之和
        :return:
        '''
        if self.verbose == True:
            print('##### Generate Distance Matrix #####\n')

        hash_codes_mag,hash_codes_grad = self.hash_codes_mag,self.hash_codes_grad
        grad_threshold = self.grad_threshold

        hash_mag_size = len(hash_codes_mag)

        distance_mag = np.resize(np.array([self.hamming_distance(x, y) for x in hash_codes_mag for y in hash_codes_mag]),
                                 (hash_mag_size, hash_mag_size))

        hash_grad_size = len(hash_codes_grad)

        distance_grad = np.resize(np.array([self.hamming_distance(x, y) for x in hash_codes_grad for y in hash_codes_grad]),
                                  (hash_grad_size, hash_grad_size))

        self.distance_mat = distance_mag + grad_threshold * distance_grad


    def __init_para(self):

        '''
        初始化参数，本函数调用在初始化函数中
        :return:
        '''

        self.flag_after_first_GOF_filter = []               #第一次gof得到的绝对索引
        self.flag_after_LOF = []                            #LOF得到的绝对索引
        self.flag_after_second_GOF_filter = []              #第二次gof得到的绝对索引
        self.flag_after_main_Category = []                  #保留主类的绝对索引
        self.pos = []                                       #存放偶极子拟合位置
        self.ori = []                                       #存放偶极子拟合方向
        self.hash_codes_mag = []                            #存放mag通道的phash编码
        self.hash_codes_grad = []                           #存放 grad通道的phash编码
        self.fit_goodness = []                              #存放偶极子fit_goodness指标
        self.distance_mat = []                              #存放距离矩阵
        self.labels = []                                    #存放标签
        self.remain_low_index = []                          #前期低阈值滤除样本保留的索引
        self.remain_high_index = []                         #后期高阈值滤除样本保留的索引（索引值在gof得到的索引基础上）
        self.remain_lof_index = []                          #lof算法滤除样本后保留的索引（索引值在low_lof得到的索引基础上）
        self.final_index = []                               #保留主类后保留下来的索引 （索引值在low_lof得到的索引基础上）

        gof_threshold_low = self.gof_threshold_low

        #初始化dip fit 的三个参数， pos,ori，fit_goodness
        self.__dip_feature()

        #初始化phash编码参数
        self.__phash_process()

        #初始化距离矩阵
        self.__distance_matrix()

        index = self.filter_bad_gof(gof_threshold_low)

        self.remain_low_index = index


    def filter_bad_gof(self,gof_threshold):

        '''
        remove samples which of fit goodness is lower than threshold
        :param coef:
        :return: the index of remained samples
        '''

        if self.verbose == True:
            print('-----GOF Filter-----\n')

        fit_goodness = self.fit_goodness
        index = np.where(fit_goodness>gof_threshold)[0]
        self.flag_after_first_GOF_filter = index

        return index


    def find_k_value(self,range_min=2,range_max = 18,if_plot= True):

        '''
        指定k的迭代范围，用轮廓系数确定k的类别，绘制出轮廓系数图
        :param range_min: k的最小取值范围; default=2
        :param range_max: k的最大取值范围; default=18
        :param if_plot:   是否画出轮廓系数曲线， default = True
        :return:
        '''

        index = self.remain_low_index
        silhouettteScore = list()
        distance_mat = np.array(self.distance_mat)
        distance_mat = distance_mat[index,:]
        distance_mat = distance_mat[:,index]

        #对指定k的范围套for循环进行聚类，并算出轮廓系数
        for i in range(range_min, range_max):

            model = AgglomerativeClustering(n_clusters=i, affinity='precomputed', linkage='average')
            model.fit(distance_mat)
            labels = model.labels_
            score = silhouette_score(distance_mat, labels,metric='precomputed')
            silhouettteScore.append(score)

        #绘制轮廓系数图
        if if_plot == True:
            plt.figure(figsize=(10, 6))
            plt.ylabel('silhouettteScore')
            plt.xlabel('k value ')
            plt.plot(range(range_min, range_max), silhouettteScore)
            plt.show()

    def auto_find_k_value(self):

        '''
        自动找k值的算法，通常会准
        :return: 返回自动找出的k的值
        '''

        distance_mat = np.array(self.distance_mat)
        index = self.remain_low_index
        distance_mat = distance_mat[index, :]
        distance_mat = distance_mat[:, index]

        range_min = 5
        sample_num = self.sample_num

        if sample_num > 150:

            range_max = int(sample_num/2)

        else:

            range_max = int(sample_num/1.5)

        silhouettte_score = []

        if range_max >= 150:

            range_min = 15

        for i in range(range_min, range_max):

            model = AgglomerativeClustering(n_clusters=i, affinity='precomputed', linkage='average')
            model.fit(distance_mat)
            labels = model.labels_
            score = silhouette_score(distance_mat, labels,metric='precomputed')
            silhouettte_score.append(score)
        silhouettte_score = np.array(silhouettte_score)
        sil_means = np.mean(silhouettte_score)
        min_index = min(np.where(silhouettte_score<sil_means)[0])

        k = np.where(silhouettte_score == np.max(silhouettte_score[min_index:]))[0][0]
        two_order_k = np.where(silhouettte_score == np.max(silhouettte_score[k+1:]))[0][0]
        # if k - two_order_k <= 0.1:
        #      k = two_order_k

        return k + range_min


    def phash_cluster(self,k_value):

        '''
        输入聚类个数，将初次阈值筛选后的距离矩阵传入进行聚类
        :param k_value: 聚类个数
        :return: 数据标签
        '''

        if self.verbose == True:
            print('##### Agglomerative Clustering ######\n')
        distance_mat = self.distance_mat
        index = self.remain_low_index
        distance_mat = distance_mat[index, :]
        distance_mat = distance_mat[:, index]
        model = AgglomerativeClustering(n_clusters=k_value, affinity='precomputed', linkage='average')
        model.fit(distance_mat)
        labels = model.labels_
        self.labels = labels

        return labels

    def phash_cluster_filter_outlier(self,k_value,remain_cls_sample_size=5,lof_k = 2):

        '''
        使用phash算法并使用lof算法去除离群点，并删除fit_goodness小于 gf_threshold_after的样本
        :param k_value: type=int 要聚类的个数
        :param remain_cls_sample_size: type=int 类的样本个数大于这个数时候使用lof
        :param lof_k: type=int lof算法使用时指定的k-distance中的k值
        :param if_return_data: type = bool (True or False) 是否return 滤除离群以后的数据
        :return: 滤除后的标签，滤除后的k值，滤除后的peak,滤除后的data数据，type=dict
        results.keys() = ['labels','number_of_category','peak_point','raw_data']
        如果要得到这个过程处理后的几个值， results = model.phash_cluster_filter_outlier(k_value=18)
        labels = results['labels']
        k = results['number_of_category']...

        '''
        lis = ['labels','number_of_category','peak_point','raw_data']
        #使用phash聚类得到label
        labels = self.phash_cluster(k_value)

        #滤除 gf_threshld_before的样本
        low_index = self.remain_low_index
        cls = np.unique(labels)
        pos = self.pos[low_index]
        ori = self.ori[low_index]
        peak = self.peak[low_index]
        gof_threshold_high = self.gof_threshold_high
        remain_index = []

        if self.verbose == True:
            print('-----LOF Filter-----\n')

        #for训练遍历整个类别，如果类别大于等于 remain_cls_sample_size时，使用lof检测离群点
        for i in cls:
            index = np.where(labels==i)[0]
            c_pos = pos[index]
            c_ori = ori[index]
            pos_size = len(c_pos)
            ori_size = len(c_ori)

            if pos_size >=remain_cls_sample_size:

                #使用dipole fit中 pos构造距离矩阵，并使用离群点算法
                c_distance_pos = np.resize(np.array([D.cityblock(i, j) for i in pos for j in c_pos]), (pos_size, pos_size))
                clf = LocalOutlierFactor(n_neighbors=lof_k, metric='precomputed')
                cs_po = clf.fit_predict(c_distance_pos)

                #使用dipole fit中 ori构造距离矩阵，并使用离群点算法
                c_distance_ori= np.resize(np.array([D.euclidean(i, j) for i in pos for j in c_ori]), (ori_size, ori_size))
                clf = LocalOutlierFactor(n_neighbors=lof_k, metric='precomputed')
                cs_ori = clf.fit_predict(c_distance_ori)

                #二者结果求并集
                cs = (cs_po + cs_ori)/2
                cs[cs <= 0] = 0
                ind = np.where(cs>0)[0]
                inx = index[ind]
                remain_index.extend(inx)

            else:
                remain_index.extend(index)

        remain_index = sorted(remain_index)
        #得到离群点检测算法的最终索引
        self.remain_lof_index= remain_index
        #作用在labels上
        labels = labels[remain_index]

        self.flag_after_LOF = self.flag_after_first_GOF_filter[remain_index]
        self.fit_goodness = self.fit_goodness[self.remain_low_index]
        self.fit_goodness = self.fit_goodness[self.remain_lof_index]
        ides = self.filter_bad_gof(gof_threshold_high)
        #得到gf_threshold_after阈值筛选后的索引
        self.remain_high_index = ides
        self.flag_after_second_GOF_filter = self.flag_after_LOF[ides]
        #作用在label上
        labels = labels[ides]

        cls = np.unique(labels).tolist()
        #返回最终还剩多少类
        for i in cls:

            labels[labels==i] = cls.index(i)

        remain_k = len(cls)


        peak = peak[self.remain_lof_index]
        peak = peak[self.remain_high_index]



        data = self.data[self.remain_low_index]
        data = data[self.remain_lof_index]
        data = data[self.remain_high_index]

        results = dict(zip(lis, [labels, remain_k, peak, data]))

        return results



    def main_cluster(self,main_cluster_num=4):
        '''
        去除类中样本个数小于main_cluster_num的类
        :param main_cluster_num: type = int;default = 4
        :return: 保留主类滤除后的标签，滤除后的k值，滤除后的peak,滤除后的data数据，type=dict
        results.keys() = ['labels','number_of_category','peak_point','raw_data']
        如果要得到这个过程处理后的几个值， results = model.phash_cluster_filter_outlier(k_value=18)
        labels = results['labels']
        k = results['number_of_category']...

        '''


        labels = self.labels[self.remain_lof_index]
        labels = labels[self.remain_high_index]
        lis = ['labels', 'number_of_category', 'peak_point', 'raw_data']

        cls = np.unique(labels)
        index_cont = []
        for cl in cls:
            cl_num = np.where(labels == cl)[0]
            if cl_num.shape[0]>=main_cluster_num:
                index_cont.extend(cl_num)
        index_cont = np.sort(index_cont)
        labels = labels[index_cont]
        cls = np.unique(labels).tolist()
        for i in cls:
            labels[labels == i] = cls.index(i)

        peak = self.peak[self.remain_low_index]
        peak = peak[self.remain_lof_index]
        peak = peak[self.remain_high_index]

        data = self.data[self.remain_low_index]
        data = data[self.remain_lof_index]
        data = data[self.remain_high_index]


        peak = peak[index_cont]
        data = data[index_cont]

        self.flag_after_main_Category = self.flag_after_second_GOF_filter[index_cont]
        self.list_final = index_cont
        k = len(cls)
        results = dict(zip(lis, [labels, k, peak, data]))

        return results

    def fit(self,main_cluster_num=4):

        '''
        完全自动化国臣
        :param main_cluster_num: 类内样本大于这个数，这个类会被保留；dtype=int;default=4
        :return:滤除后的标签，滤除后的k值，滤除后的peak,滤除后的data数据，type=dict
        results.keys() = ['labels','number_of_category','peak_point','raw_data']
        如果要得到这个过程处理后的几个值， results = model.phash_cluster_filter_outlier(k_value=18)
        labels = results['labels']
        k = results['number_of_category']...

        '''


        info = self.info
        lis = ['labels','number_of_category','peak_point','raw_data']

        k = self.auto_find_k_value()
        _, _, _, _ = self.phash_cluster_filter_outlier(k_value=k)
        labels, k, peak, data = self.main_cluster(main_cluster_num)

        plot_fig = PlotCluster(info, labels, peak).plot_cluster(k)



        results = dict(zip(lis,[labels,k,peak,data]))

        return results

def save_data(datas,info,labels,path):

    cls = np.unique(labels)
    infox =  mne.pick_info(info,sel = mne.pick_types(info, meg=True, ref_meg=False))
    for i in cls.tolist():
        index = np.where(labels == i)[0]
        data = datas[index]
        cluster_epochs = mne.EpochsArray(data=data, info=infox, verbose='error')
        fig = plot_map(cluster_epochs, show=False)
        cluster_epochs.save(os.path.join(path,'epoc_{}.fif'.format(str(i))), verbose='error',overwrite=True)
        plt.savefig(os.path.join(path,'pic_{}.png'.format(str(i))))


if __name__ == '__main__':

    home_dir = os.path.dirname(os.path.abspath(__file__))

    for sub_num in range(13):
        t0 = time.time()
        raw_path = home_dir+'/processed_data/sub{}_data.npy'.format(str(sub_num))
        info_path = home_dir+'/processed_data/sub{}_info.pkl'.format(str(sub_num))
        class_path = home_dir + '/processed_data/sub{}_initial_cluster_num.npy'.format(str(sub_num))
        save_path = home_dir+'/results_pic/result/sub{}/'.format(str(sub_num))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        f_read = open(info_path,'rb')
        info = pickle.load(f_read)
        class_num = np.load(class_path)
        datas = np.load(raw_path)

        print(datas.shape)
        datas = datas.astype('float32')
        model = PhashClustering(datas, info)
        info = model.info

        results = model.phash_cluster_filter_outlier(class_num)
        labels = results['labels']
        peak = results['peak_point']
        k = results['number_of_category']

        plot_fig = PlotCluster(info, labels, peak)
        temp = plot_fig.plot_cluster(k, is_show=False)
        temp.savefig(save_path + 'first.png')

        main_reuslts = model.main_cluster(4)
        labels = main_reuslts['labels']
        peak = main_reuslts['peak_point']
        k = main_reuslts['number_of_category']
        data = main_reuslts['raw_data']

        plot_fig = PlotCluster(info, labels, peak)
        temp = plot_fig.plot_cluster(k, is_show=False)
        temp.savefig(save_path+'second.png')

        save_data(data,info,labels,save_path)
        print(time.time()-t0)

