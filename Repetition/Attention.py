import tensorflow as tf


class AttentionVRPActor(object):
    # 一种通用的VRP注意模块
    def __init__(self, dim, use_tanh=False, C=10, _name='Attention', _scope=''):  # scope是范围
        self.use_tanh = use_tanh
        self._scope = _scope

        with tf.variable_scope(_scope + _name):  # 创建共享变量
            # self.v :是一个形状为[1 x dim]的变量
            self.v = tf.get_variable('v', [1, dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.v = tf.expand_dims(self.v, 2)

        self.emb_d = tf.layers.Conv1D(dim, 1, _scope=_scope + _name + '/emb_d')  # conv1d
        self.emb_ld = tf.layers.Conv1D(dim, 1, _scope=_scope + _name + '/emb_ld')  # conv1d_2

        self.project_d = tf.layers.Conv1D(dim, 1, _scope=_scope + _name + '/proj_d')  # conv1d_1
        self.project_ld = tf.layers.Conv1D(dim, 1, _scope=_scope + _name + '/proj_ld')  # conv1d_3
        self.project_query = tf.layers.Dense(dim, _scope=_scope + _name + '/proj_q')  #
        self.project_ref = tf.layers.Conv1D(dim, 1, _scope=_scope + _name + '/proj_ref')  # conv1d_4

        self.C = C  # tanh 探索参数
        self.tanh = tf.nn.tanh

    def __call__(self, query, ref, env):
        '''
        此函数获取一个查询张量与参考 rensor 并且返回逻辑运算.
        :param query:是解码器在当前时间步的隐藏状态 [batch_size x dim]
        :param ref:是编码器的隐藏状态集 [batch_size x max_time x dim]
        :param env:
        :return:
        e:卷积参考，形状为 [batch_size x max_time x dim]
        logits: [batch_size x max_time]
        '''
        # 从环境中获取当前需求和负载值
        demand = env.demand
        load = env.load
        max_time = tf.shape(demand)[1]

        # 嵌入需求并预测它
        # emb_d:[batch_size x max_time x dim ]
        emb_d = self.emb_d(tf.expand_dims(demand, 2))
        # d:[batch_size x max_time x dim ]
        d = self.project_d(emb_d)

        # 嵌入载入 - 需求
        # emb_ld:[batch_size*beam_width x max_time x hidden_dim]
        emb_ld = self.emb_ld(tf.expand_dims(tf.tile(tf.expand_dims(load, 1), [1, max_time]) -
                                            demand, 2))
        # ld:[batch_size*beam_width x hidden_dim x max_time ]
        ld = self.project_ld(emb_ld)

        # expanded_q,e: [batch_size x max_time x dim]
        e = self.project_ref(ref)
        q = self.project_query(query)  # [batch_size x dim]
        expanded_q = tf.tile(tf.expand_dims(q, 1), [1, max_time, 1])

        # v_view:[batch_size x dim x 1]
        v_view = tf.tile(self.v, [tf.shape(e)[0], 1, 1])

        # u : [batch_size x max_time x dim] * [batch_size x dim x 1] =
        #       [batch_size x max_time]
        u = tf.squeeze(tf.matmul(self.tanh(expanded_q + e + d + ld), v_view), 2)

        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u

        return e, logits


class AttentionVRPCritic(object):
    """一个通用的VRP注意模块"""

    def __init__(self, dim, use_tanh=False, C=10, _name='Attention', _scope=''):

        self.use_tanh = use_tanh
        self._scope = _scope

        with tf.variable_scope(_scope + _name):
            # self.v: is a variable with shape [1 x dim]
            self.v = tf.get_variable('v', [1, dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.v = tf.expand_dims(self.v, 2)

        self.emb_d = tf.layers.Conv1D(dim, 1, _scope=_scope + _name + '/emb_d')  # conv1d
        self.project_d = tf.layers.Conv1D(dim, 1, _scope=_scope + _name + '/proj_d')  # conv1d_1

        self.project_query = tf.layers.Dense(dim, _scope=_scope + _name + '/proj_q')  #
        self.project_ref = tf.layers.Conv1D(dim, 1, _scope=_scope + _name + '/proj_e')  # conv1d_2

        self.C = C  # tanh exploration parameter
        self.tanh = tf.nn.tanh

    def __call__(self, query, ref, env):
        """
        This function gets a query tensor and ref rensor and returns the logit op.
        Args:
            query: 在当前时间步长的解码器的隐藏状态. [batch_size x dim]
            ref: 编码器的隐藏状态集.
                [batch_size x max_time x dim]

            env: 保持需求和加载值并且帮助解码. 它还包含掩码
                env.mask: 一个用来将 logits 和 glimpses 掩码的矩阵.形状为：[batch_size x max_time].
                在这个矩阵中 0 表示没有掩码的节点. 在这个掩码中的任何正数表示节点不能被选择为下一个决策点
                env.demands: 随时间变化的需求列表.

        Returns:
            e: 卷积参考，形状为： [batch_size x max_time x dim]
            logits: [batch_size x max_time]
        """
        # 惩罚网络中我们需要第一个需求值
        demand = env.input_data[:, :, -1]
        max_time = tf.shape(demand)[1]

        # 嵌入需求并预测它
        # emb_d:[batch_size x max_time x dim ]
        emb_d = self.emb_d(tf.expand_dims(demand, 2))
        # d:[batch_size x max_time x dim ]
        d = self.project_d(emb_d)

        # expanded_q,e: [batch_size x max_time x dim]
        e = self.project_ref(ref)
        q = self.project_query(query)  # [batch_size x dim]
        expanded_q = tf.tile(tf.expand_dims(q, 1), [1, max_time, 1])

        # v_view:[batch_size x dim x 1]
        v_view = tf.tile(self.v, [tf.shape(e)[0], 1, 1])

        # u : [batch_size x max_time x dim] * [batch_size x dim x 1] =
        #       [batch_size x max_time]
        u = tf.squeeze(tf.matmul(self.tanh(expanded_q + e + d), v_view), 2)

        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u

        return e, logits