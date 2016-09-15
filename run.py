import src.nntrain as nntrain
from importlib import reload

nntrain.train_nn_wrapper(3,n_iter=20)
nntrain.train_nn_wrapper(4,n_iter=20)

#     r_mlp.append(build_nn('regress',['Rectifier','Rectifier']            ,[500,500]    ,n_iter,batch_size,'momentum'))
#     r_mlp.append(build_nn('regress',['Rectifier','Rectifier','Rectifier'],[200,200,200],n_iter,batch_size,'momentum'))
#     r_mlp.append(build_nn('regress',['Tanh','Tanh','Tanh']               ,[100,100,100],n_iter,batch_size,'momentum'))
#     r_mlp.append(build_nn('regress',['Tanh','Tanh']                      ,[500,500]    ,n_iter,batch_size,'momentum'))
#     r_mlp.append(build_nn('regress',['Tanh','Tanh']                      ,[200,200]    ,n_iter,batch_size,'momentum'))
#     r_mlp.append(build_nn('regress',['Tanh']                             ,[500]        ,n_iter,batch_size,'momentum'))
#     r_mlp.append(build_nn('regress',['Tanh','Tanh']                      ,[500,500]    ,n_iter,batch_size,'momentum',learning_momentum=0.7))
#     r_mlp.append(build_nn('regress',['Tanh','Tanh']                      ,[500,500]    ,n_iter,batch_size,'sgd'))
