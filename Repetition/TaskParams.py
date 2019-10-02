from collections import namedtuple

#设置任务特定参数,namedtuple是命名元组，类似于一个列表
TaskVRP=namedtuple('TaskVRP',['task_name','input_dim','n_nodes','n_cust','decode_len','capacity','demand_max'])

task_lst={}#创建字典类型

#VRP10
vrp10=TaskVRP(task_name='vrp',input_dim=3,n_nodes=11,n_cust=10,decode_len=16,capacity=20,demand_max=9)#输入维为3，节点个数比任务数多1，顾客数等于任务数，容量比任务数多10，最大需求为9，解码长度数暂时不知道规律
task_lst['vrp10']=vrp10#给键值赋值，对应vrp10的列表

# VRP20
vrp20 = TaskVRP(task_name ='vrp',input_dim=3,n_nodes=21,n_cust = 20,decode_len=30,capacity=30,demand_max=9)
task_lst['vrp20'] = vrp20

# VRP50
vrp50 = TaskVRP(task_name = 'vrp',input_dim=3,n_nodes=51,n_cust = 50,decode_len=70,capacity=40,demand_max=9)
task_lst['vrp50'] = vrp50

# VRP100
vrp100 = TaskVRP(task_name = 'vrp',input_dim=3,n_nodes=101,n_cust = 100,decode_len=140,capacity=50,demand_max=9)
task_lst['vrp100'] = vrp100