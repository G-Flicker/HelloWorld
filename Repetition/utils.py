import numpy as np
import tensorflow as tf
import os
import warnings
import collections


def create_VRP_dataset(
        n_problems,
        n_cust,
        data_dir,
        seed=None,
        data_type='train'):
    '''
    此函数将创建VRP实例并将其保存在磁盘上，如果文件可用，它将加载文件
    :param n_problems:要生成的问题数
    :param n_cust:在问题中顾客的个数
    :param data_dir:保存或载入文件的目录
    :param seed:用来生成数据的随机数种子
    :param data_type:生成数据的目的，可以是'train','val',或其他字符串
    :return:一个numpy数组，形状为[n_problem x (n_cust+1) x 3],在最后一维，有x,y,和顾客的需求，最后一个点是仓库并且他的需求是0
    '''

    # 建立随机数生成器
    n_nodes = n_cust + 1
    if seed == None:
        rnd = np.random  # 创建随机数对象
    else:
        rnd = np.random.RandomState(seed)  # 给随机数对象固定的随机数种子

    # 生成任务名称和数据文件
    task_name = 'vrp-size-{}-len-{}-{}.txt'.format(n_problems, n_nodes, data_type)  # 任务名
    fname = os.path.join(data_dir, task_name)  # join合并俩个字符串,结果为：data_dir/task_name

    # 创建/加载数据
    if os.path.exists(fname):
        print('正在加载数据集：{}'.format(task_name))
        data = np.loadtxt(fname, delimiter=' ')  # delimiter是分隔符,设定以空格为分隔符读取数据
        data = data.reshape(-1, n_nodes, 3)  # 改变形状,-1代表自动计算,将之前处理过的数据重新排列
    else:
        print('正在创建数据集：{}'.format(task_name))
        # 生成一个大小为n_problems的训练集
        x = rnd.uniform(0, 1, size=(n_problems, n_nodes, 2))  # 从左往右依次为采样(下界，采样上界，size=(样本大小)）,随机采样
        d = rnd.randint(1, 10, [n_problems, n_nodes, 1])  # 从左往右依次为采样(下界，采样上界，size=(样本大小)）,随机整数采样
        d[:, -1] = 0  # 设置最后一个点为仓库，且需求为0
        data = np.concatenate([x, d], 2)  # 在轴axis=2上连接俩个数据
        np.savetxt(fname, data.reshape(-1, n_nodes * 3))  # -1为自动计算，txt只能写二维数据，写不了三维数据，所以要转换为二维

    return data


class DataGenerator(object):
    def __init__(self,
                 args):
        '''
               此类为训练和测试生成VRP问题
               :param args:参数字典，它包括：
               args['random_seed']: 随机数种子
               args['test_size']: 测试的问题数
               args['n_nodes']: 节点个数
               args['n_cust']: 顾客数量
               args['batch_size']: 训练的批量大小
               '''
        self.args = args
        self.rnd = np.random.RandomState(seed=args['random_seed'])
        print('创建训练迭代器')

        # 创建测试数据
        self.n_problems = args['test_size']
        self.test_data = create_VRP_dataset(self.n_problems, args['n_cust'], './data',
                                            seed=args['random_seed'] + 1, data_type='test')

        self.reset()

    def reset(self):
        self.count = 0

    def get_train_next(self):
        '''
        获取下一批要训练的问题
        :return:数据,大小为：[batch_size x max_time x 3]
        '''

        input_pnt = self.rnd.uniform(0, 1,
                                     size=(self.args['batch_size'], self.args['n_nodes'], 2))

        demand = self.rnd.randint(1, 10, [self.args['batch_size'], self.args['n_nodes']])
        demand[:, -1] = 0  # 最后一个点为仓库，并将其需求设置为0

        input_data = np.concatenate([input_pnt, np.expand_dims(demand, 2)],
                                    2)  # np.expand_dims是在轴axis=2上扩充一个维,concatenate在轴axis=2上连接数据

        return input_data

    def get_test_next(self):
        '''
        得到下一批测试的问题
        :return: 数据
        '''
        if self.count < self.args['test_size']:
            input_pnt = self.test_data[self.count:self.count + 1]
            self.count += 1
        else:
            warnings.warn("重置测试迭代器")
            self.count = 0
            input_pnt = self.test_data[self.count:self.count + 1]
            self.count += 1

        return input_pnt

    def get_test_all(self):
        '''
        获取所有测试问题
        :return: 测试数据
        '''
        return self.test_data


class State(collections.namedtuple('State', ('load', 'demand', 'd_sat', 'mask'))):
    '''
    创建列表的框架
    '''
    pass


class Env(object):
    def __init__(self,
                 args):
        '''
        VRP环境
        :param args: 参数字典，包括：
        args['n_nodes']: 在VRP中的节点个数
        args['n_custs']: VRP中的顾客个数
        args['input_dim']:问题的维度是2
        '''
        self.capacity = args['capacity']
        self.n_nodes = args['n_nodes']
        self.n_cust = args['n_cust']
        self.input_dim = args['input_dim']
        self.input_data = tf.placeholder(tf.float32, \
                                         shape=[None, self.n_nodes, self.input_dim])

        self.input_pnt = self.input_data[:, :, :2]
        self.demand = self.input_data[:, :, -1]
        self.batch_size = tf.shape(self.input_pnt)[0]

    def reset(self, beam_width=1):
        '''
        重置环境，此环境可能与不同的解码一起使用。在使用波束搜索解码器的情况下，我们需要将掩码的行数增加一个波束宽度系数
        :param beam_width:波束宽度系数
        :return:
        '''

        # 维数
        self.beam_width = beam_width
        self.batch_beam = self.batch_size * beam_width

        self.input_pnt = self.input_data[:, :, :2]
        self.demand = self.input_data[:, :, -1]

        # 为波束搜索解码器调整 self.input_pnt 和 self.demand
        # self.input_pnt=nd.tile(self.input_pnt, [self.beam_width,1,1])

        # demand: [batch_size * beam_width, max_time]
        # demand[i] = demand[i+batchsize]
        self.demand = tf.tile(self.demand, [self.beam_width, 1])#tf.tile是复制

        # load: [batch_size * beam_width]
        self.load = tf.ones([self.batch_beam]) * self.capacity

        # 创建掩码
        self.mask = tf.zeros([self.batch_size * beam_width, self.n_nodes],
                             dtype=tf.float32)

        # 更新掩码， 如果顾客的需求和仓库都为0
        self.mask = tf.concat([tf.cast(tf.equal(self.demand, 0), tf.float32)[:, :-1],
                               tf.ones([self.batch_beam, 1])], 1)

        state = State(load=self.load,
                      demand=self.demand,
                      d_sat=tf.zeros([self.batch_beam, self.n_nodes]),
                      mask=self.mask)

        return state

    def step(self,
             idx,
             beam_parent=None):
        '''
        运行环境一步并更新需求，载入和掩码
        :param idx:
        :param beam_parent:
        :return:
        '''

        # 如果环境正在被波束搜索解码器使用
        if beam_parent is not None:
            # BatchBeamSeq: [batch_size*beam_width x 1]
            # [0,1,2,3,...,127,0,1,...],
            batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(self.batch_size), tf.int64),
                                                  [self.beam_width]), 1)
            # batchedBeamIdx:[batch_size*beam_width]
            batchedBeamIdx = batchBeamSeq + tf.cast(self.batch_size, tf.int64) * beam_parent#tf.cast是数据类型转换
            # demand:[batch_size*beam_width x sourceL]
            self.demand = tf.gather_nd(self.demand, batchedBeamIdx)
            # load:[batch_size*beam_width]
            self.load = tf.gather_nd(self.load, batchedBeamIdx)
            # MASK:[batch_size*beam_width x sourceL]
            self.mask = tf.gather_nd(self.mask, batchedBeamIdx)#tf.gather_nd是根据索引提取元素

        BatchSequence = tf.expand_dims(tf.cast(tf.range(self.batch_beam), tf.int64), 1)
        batched_idx = tf.concat([BatchSequence, idx], 1)#tf.concat是拼接函数张量

        # 满足了多少需求
        d_sat = tf.minimum(tf.gather_nd(self.demand, batched_idx), self.load)

        # 更新需求
        d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(self.demand), tf.int64))
        self.demand = tf.subtract(self.demand, d_scatter)

        # 更新加载
        self.load -= d_sat

        # 重新装满卡车 -- idx: [10,9,10] -> load_flag: [1 0 1]
        load_flag = tf.squeeze(tf.cast(tf.equal(idx, self.n_cust), tf.float32), 1)
        self.load = tf.multiply(self.load, 1 - load_flag) + load_flag * self.capacity#multiply是对应元素相乘

        # 当顾客需求为0时的掩码
        self.mask = tf.concat([tf.cast(tf.equal(self.demand, 0), tf.float32)[:, :-1],
                               tf.zeros([self.batch_beam, 1])], 1)

        # 当load=0的掩码
        # 当在仓库时仍然还有一个需求掩码

        self.mask += tf.concat([tf.tile(tf.expand_dims(tf.cast(tf.equal(self.load, 0),
                                                               tf.float32), 1), [1, self.n_cust]),
                                tf.expand_dims(
                                    tf.multiply(tf.cast(tf.greater(tf.reduce_sum(self.demand, 1), 0), tf.float32),
                                                tf.squeeze(tf.cast(tf.equal(idx, self.n_cust), tf.float32))), 1)], 1)

        state = State(load=self.load,
                      demand=self.demand,
                      d_sat=d_sat,
                      mask=self.mask)

        return state


def reward_func(sample_solution):
    '''
    VRP任务的奖励定义为路线长度的负值
    满足要求：一个列表张量，大小为decode_len，形状为[batch_size]
    :param sample_solution:一个列表张量大小为decode_len，形状为 [batch_size x input_dim]
    :return:尺寸为[batch_size]的张量
    例如:
        sample_solution = [[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]]
        sourceL = 3
        batch_size = 2
        input_dim = 2
        sample_solution_tilted[ [[5,5]
                                                    #  [6,6]]
                                                    # [[1,1]
                                                    #  [2,2]]
                                                    # [[3,3]
                                                    #  [4,4]] ]
    '''
    # 创建初始解决方案，形状为 [sourceL x batch_size x input_dim]
    # 创建解决方案样品，形状为 [sourceL x batch_size x input_dim]
    sample_solution = tf.stack(sample_solution, 0)

    sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1], 0),
                                        sample_solution[:-1]), 0)
    # 基于路径长度获取奖励值

    route_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow( \
        (sample_solution_tilted - sample_solution), 2), 2), .5), 0)
    return route_lens_decoded

