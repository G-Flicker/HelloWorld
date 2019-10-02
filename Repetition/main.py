import numpy as np
import tensorflow as tf
import time

from config import parseParams
from agent import RLAgent

def load_task_specific_components(task):
    '''
    这个函数用来加载特定任务的库
    '''

    if task == 'vrp':
        from utils import DataGenerator,Env,reward_func
        from Attention import AttentionVRPActor,AttentionVRPCritic

        AttentionActor = AttentionVRPActor
        AttentionCritic = AttentionVRPCritic

    else:
        raise Exception('任务未实现') #抛出异常，任务未实现


    return DataGenerator, Env, reward_func, AttentionActor, AttentionCritic

def main(args, prt):
    config = tf.ConfigProto() #tensorflow的参数配置对象
    config.gpu_options.allow_growth = True #动态申请显存，需要多少就申请多少
    sess = tf.Session(config=config)

    # 加载任务特定的类
    DataGenerator, Env, reward_func, AttentionActor, AttentionCritic = \
        load_task_specific_components(args['task_name'])

    dataGen = DataGenerator(args)
    dataGen.reset()
    env = Env(args)

    #创建一个 RL（强化学习）代理
    agent = RLAgent(args,
                    prt,
                    env,
                    dataGen,
                    reward_func,
                    AttentionActor,
                    AttentionCritic,
                    is_train=args['is_train'])
    agent.Initialize(sess)

    #训练或评估
    start_time = time.time() # 计算时间用的，记录开始时间
    if args['is_train']:# 如果参数是训练
        prt.print_out('训练开始')
        train_time_beg = time.time() # 开始训练的时间
        for step in range(args['n_train']):
            summary = agent.run_train_step()
            _, _ , actor_loss_val, critic_loss_val, actor_gra_and_var_val, critic_gra_and_var_val,\
                R_val, v_val, logprobs_val,probs_val, actions_val, idxs_val= summary

            if step%args['save_interval'] == 0:
                agent.saver.save(sess,args['model_dir']+'/model.ckpt', global_step=step)#sess是之前tensorflow的初始化对象

            if step%args['log_interval'] == 0:
                train_time_end = time.time()-train_time_beg
                prt.print_out('训练步数: {} -- 时间: {} -- 训练奖励: {} -- 值: {}'\
                      .format(step,time.strftime("%H:%M:%S", time.gmtime(\
                        train_time_end)),np.mean(R_val),np.mean(v_val)))
                prt.print_out('    actor loss: {} -- critic loss: {}'\
                      .format(np.mean(actor_loss_val),np.mean(critic_loss_val)))
                train_time_beg = time.time()
            if step%args['test_interval'] == 0:
                agent.inference(args['infer_type'])

    else: # 否则就是在推论
        prt.print_out('评估开始')
        agent.inference(args['infer_type'])


    prt.print_out('总时间为： {}'.format(\
        time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))) # 运行完毕之后输出运行总时间

if __name__ == "__main__":
    args, prt = parseParams()#根据命令行输入的参数调整参数后返回，args是全部参数的字典,prt是一个实例化后的控制输出对象
    #随机初始化
    random_seed = args['random_seed']#从命令行获取参数：随机数种子
    if random_seed is not None and random_seed > 0:
        prt.print_out("随机数种子为:  %d" % random_seed)#同步输出到控制台与日志文件中
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
    tf.reset_default_graph()# 对图重置，tensorflow默认一个程序创建一个图

    main(args, prt)
