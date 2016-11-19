import src.nntrain as nntrain
import src.nnplot as nnplot

r_str=[None]*16

# Define preprocessor
x_ppi={'name':'MinMax','method':'individually'}
y_ppi={'name':'MaxAbs','method':'alltogether'}

xp1 = {'name':'StandardScaler','method':'individually'}
xp2 = {'name':'StandardScaler','method':'alltogether'}
xp3 = {'name':'MinMax','method':'individually'}
xp4 = {'name':'MinMax','method':'alltogether'}

yp1 = {'name':'StandardScaler','method':'individually'}
yp2 = {'name':'StandardScaler','method':'alltogether'}
yp3 = {'name':'MaxAbs','method':'individually'}
yp4 = {'name':'MaxAbs','method':'alltogether'}

r_str[0], x, y = nntrain.train_nn_wrapper(400, xp1, yp1, n_stable=75)
r_str[1], x, y = nntrain.train_nn_wrapper(400, xp1, yp2, n_stable=75)
r_str[2], x, y = nntrain.train_nn_wrapper(400, xp2, yp1, n_stable=75)
r_str[3], x, y = nntrain.train_nn_wrapper(400, xp2, yp2, n_stable=75)

r_str[4], x, y = nntrain.train_nn_wrapper(400, xp1, yp3, n_stable=75)
r_str[5], x, y = nntrain.train_nn_wrapper(400, xp1, yp4, n_stable=75)
r_str[6], x, y = nntrain.train_nn_wrapper(400, xp2, yp3, n_stable=75)
r_str[7], x, y = nntrain.train_nn_wrapper(400, xp2, yp4, n_stable=75)

r_str[8], x, y = nntrain.train_nn_wrapper(400, xp3, yp1, n_stable=75)
r_str[9], x, y = nntrain.train_nn_wrapper(400, xp3, yp2, n_stable=75)
r_str[10], x, y = nntrain.train_nn_wrapper(400, xp4, yp1, n_stable=75)
r_str[11], x, y = nntrain.train_nn_wrapper(400, xp4, yp2, n_stable=75)

r_str[12], x, y = nntrain.train_nn_wrapper(400, xp3, yp3, n_stable=75)
r_str[13], x, y = nntrain.train_nn_wrapper(400, xp3, yp4, n_stable=75)
r_str[14], x, y = nntrain.train_nn_wrapper(400, xp4, yp3, n_stable=75)
r_str[15], x, y = nntrain.train_nn_wrapper(400, xp4, yp4, n_stable=75)

for r in r_str:
    nnplot.plot_all_figs(r, x, y)
#nntrain.train_nn_wrapper(300,n_stable=50)
#nntrain.train_nn_wrapper(600,n_stable=50)

#     r_mlp.append(build_nn('regress',['Rectifier','Rectifier']            ,[500,500]    ,n_iter,batch_size,'momentum'))
#     r_mlp.append(build_nn('regress',['Rectifier','Rectifier','Rectifier'],[200,200,200],n_iter,batch_size,'momentum'))
#     r_mlp.append(build_nn('regress',['Tanh','Tanh','Tanh']               ,[100,100,100],n_iter,batch_size,'momentum'))
#     r_mlp.append(build_nn('regress',['Tanh','Tanh']                      ,[500,500]    ,n_iter,batch_size,'momentum'))
#     r_mlp.append(build_nn('regress',['Tanh','Tanh']                      ,[200,200]    ,n_iter,batch_size,'momentum'))
#     r_mlp.append(build_nn('regress',['Tanh']                             ,[500]        ,n_iter,batch_size,'momentum'))
#     r_mlp.append(build_nn('regress',['Tanh','Tanh']                      ,[500,500]    ,n_iter,batch_size,'momentum',learning_momentum=0.7))
#     r_mlp.append(build_nn('regress',['Tanh','Tanh']                      ,[500,500]    ,n_iter,batch_size,'sgd'))
