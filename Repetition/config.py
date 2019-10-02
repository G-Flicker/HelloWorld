import argparse
from datetime import datetime
import os
import sys
from TaskParams import task_lst

class printOut():
    def __init__(self,f=None,stdout_print=True):
        '''
        这个类被用来控制输出
        它将被同时写入文件f与屏幕
        :param f:要写入的文件名
        :param stdout_print:是否输出到控制台，默认是
        '''
        self.out_file=f
        self.stdout_print=stdout_print

    def print_out(self,s,new_line=True):
        '''
        类似于print，但是支持刷新和输出到文件
        :param s:所要读取的字符串
        :param new_line:是否写入新行
        '''
        if isinstance(s,bytes):#判断s的类型是否为bytes
            s=s.decode('utf-8')#如果是则用utf-8解码

        if self.out_file:#默认f=None不会输出，如果文件存在，则if判断为True
            self.out_file.write(s)
            if new_line:#默认为True
                self.out_file.write('\n')
        self.out_file.flush()#强行把缓冲区中的内容放到磁盘中，数据立刻写入到文件中，如果没有该函数，则只有程序运行完成才会把数据保存到文件中

        #stdout 标准输出和标准错误（通常缩写为stdout和stderr,当print某东西，结果输出到stdout管道(pipe)中,当程序崩溃并打印出调试信息时，结果输出到stderr管道中）
        if self.stdout_print:
            print(s,end='',file=sys.stdout)#file=可以设置为所要输出的文件名，默认为输出到控制台
            if new_line:
                sys.stdout.write('\n')#sys.stdout.write(obj+"\n)等价于print(obj).也就是说该函数是输出到控制台的.
            sys.stdout.flush()#立即写入

def initialize_task_settings(args,task):
    '''
    初始化任务设置
    :param args: 命令行参数对象
    :param task: 任务名称
    :return:
    args['task_name']=TaskParams.task_params.task_name
    args['input_dim'] = task_params.input_dim
    args['n_nodes'] = task_params.n_nodes
    args['decode_len'] = task_params.decode_len
    '''
    try:
        task_params=task_lst[task]#根据输入任务参数名在预先设置好的字典中查找并返回对应列表
    except:
        raise Exception('不支持该任务')#如果查找不到则抛出异常

    for name,value in task_params._asdict().items():#将对应列表转换为字典,并以键与值为元组，组合成列表的形式，返回字典的所有项,结果为dict_items([(键，对应的值),(另一个键,另一个键对应的值)])
        args[name]=value

    return args


def str2bool(v):
    '''
    如果输入字符串是'true'或者'1'，则返回true,否则返回False
    :param v: 所要判断的字符串
    :return: True 或 False
    '''
    return v.lower() in ('true','1')

def get_time():
    '''
    :return: 返回格式化的当前时间.例如：2019-08-27_20-55-47
    '''
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S') #y:年,m:月,d:日_H:时,M:分,S:秒

def parseParams():
    '''
    添加参数
    从命令行分析输入的参数
    有默认值
    :return:
    '''
    #实例化对象
    parser=argparse.ArgumentParser(description="用深度学习进行神经组合优化")

    #数据
    parser.add_argument('-task',default='vrp10',help='选择要解决的问题，例如vrp10')
    parser.add_argument('-batch_size',default=128,type=int,help='训练的批量大小')
    parser.add_argument('-n_train',default=2600,type=int,help='训练的步数')
    parser.add_argument('-test_size',default=100,type=int,help='在测试数据集中问题的数量')

    #网络
    parser.add_argument('-agent_type',default='attention',help='填写attention或pointer')
    parser.add_argument('-forget_bias',default=1.0,type=float,help='长短期记忆细胞的遗忘偏参数')
    parser.add_argument('-embedding_dim',default=128,type=int,help='输入嵌入层的维数')
    parser.add_argument('-hidden_dim',default=128,type=int,help='编码器或者解码器中隐藏层的维数')
    parser.add_argument('-n_process_blocks',default=3,type=int,help='在惩罚网络中运行的进程数')
    parser.add_argument('-rnn_layers',default=1,type=int,help='在编码器和解码器中的记忆层数')
    parser.add_argument('-decode_len',default=None,type=int,help='解码器在停止之前运行时间步的数量')
    parser.add_argument('-n_glimpses',default=0,type=int,help='在注意力机制中的glimpses数')#暂时没明白这是啥
    parser.add_argument('-tanh_exploration',default=10,type=float,help='在网络中通过浏览在softmax中的tanh激活函数实现超参数控制探索')
    parser.add_argument('-use_tanh',type=str2bool,default=False,help="true 或 1 (是字符串，例如'1')")
    parser.add_argument("-mask_glimpses",type=str2bool,default=True,help="true 或 1 (是字符串，例如'1')")
    parser.add_argument('-mask_pointer',type=str2bool,default=True,help="true 或 1 (是字符串，例如'1')")
    parser.add_argument('-dropout',default=0.1,type=float,help='丢弃概率')

    #训练
    parser.add_argument('-is_train',default=True,type=str2bool,help='训不训练，填True或False')
    parser.add_argument('-actor_net_lr',default=1e-4,type=float,help='行动者网络中的学习率')
    parser.add_argument('-critic_net_lr',default=1e-4,type=float,help='惩罚网络中de学习率')
    parser.add_argument('-random_seed',default=24601,type=int,help='选择随机数种子')
    parser.add_argument('-max_grad_norm',default=2.0,type=float,help='梯度裁剪')
    parser.add_argument('-entropy_coedd',default=0.0,type=float,help='熵正则化系数')
    #parser.add_argument('-loss_type',type=int,default=1,help='1,2,3(是数字)')

    #推论
    parser.add_argument('-infer_type',default='batch',help='填batch或single,表示一个个去推论或者一次性全部运行')
    parser.add_argument('-beam_width',default=10,type=int,help='束搜索的宽度')

    #其余参数
    parser.add_argument('-stdout_print',default=True,type=str2bool,help='是否输出到控制台')
    parser.add_argument('-gpu',default='0',type=str,help='使用第几块GPU，默认从0开始，如果输入"2",代表仅使用GPU2（也就是第三块GPU)，如果要指定多块，输入的前后顺序表示优先度，输入"2,1",表示优先使用2号设备')
    parser.add_argument('-log_interval',default=20,type=int,help='日志信息的间隔步数')
    parser.add_argument('-test_interval',default=20,type=int,help='测试的间隔步数')
    parser.add_argument('-save_interval',default=100,type=int,help='保存的间隔步数')
    parser.add_argument('-log_dir',type=str,default='logs',help='日志目录')
    parser.add_argument('-data_dir',type=str,default='data',help='数据目录')
    parser.add_argument('-model_dir',type=str,default='',help='模型目录')
    parser.add_argument('-load_path',type=str,default='',help='载入训练变量的路径')
    # parser.add_argument('-disable_tqdm',default=True,type=str2bool,help='是否使用tqdm库，tqdm库可以弄一个进度条,True或False')

    args,unknown=parser.parse_known_args()
    args=vars(args)#vars关键字返回一个对象属性与属性值的字典

    args['log_dir']="{}/{}-{}".format(args['log_dir'],args['task'],get_time())#设置日志目录，原本的log_dir参数/任务名称-当前时间
    if args['model_dir']=='':#如果模型目录未设定
        args['model_dir']=os.path.join(args['log_dir'],'model')#os.path.join可以拼接俩个字符串为路径，如：log_dir/model

    #控制输出写文件
    try:
        os.makedirs(args['log_dir'])#尝试创建日志目录,makedirs方法可以递归的创建目录，如果子目录的父级目录都不存在，也会自动创建
        os.makedirs(args['model_dir'])#尝试创建模型目录
    except:
        pass#如果报错不管了

    #创建打印处理程序
    out_file=open(os.path.join(args['log_dir'],'results.txt'),'w+')#w+表示如果存在文件则删除原有内容从头开始编辑，如果不存在则创建文件
    prt=printOut(out_file,args['stdout_print'])

    os.environ['CUDA_VISIBLE_DEVICES']=args['gpu']#根据参数设置当前使用的GPU设备，如gpu='0'则名称为'/gpu:0'

    args=initialize_task_settings(args,args['task'])

    #打印运行参数
    for key,value in sorted(args.items()):
        prt.print_out("{}: {}".format(key,value))#同时输出到控制台与日志文件中

    return args,prt




