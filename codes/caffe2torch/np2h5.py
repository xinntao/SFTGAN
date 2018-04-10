import numpy as np
import h5py

weight_dict = np.load('../../models/OutdoorSceneSeg_bic_iter_30000.npy', encoding='latin1').item()
layers = set()
idx = 0
ori_idx = 0
for name, data in sorted(weight_dict.items()):
    ori_idx += 1
    layer = name.split('.')[0].split('_bn')[0]
    layers.add(layer)
for layer in layers:
    print(layer)
    key_weight = layer+'.weight'
    key_bias = layer+'.bias'
    key_bn_weight = layer+'_bn.weight'
    key_bn_bias = layer+'_bn.bias'
    key_running_mean = layer+'_bn.running_mean'
    key_running_var = layer+'_bn.running_var'
    if key_weight in weight_dict:
        h5f = h5py.File('./dump/' + layer + '.h5', 'w')
        h5f.create_dataset('weight', data=weight_dict[key_weight])
        idx += 1
    if key_bias in weight_dict:
        h5f.create_dataset('bias', data=weight_dict[key_bias])
        idx += 1
    if key_bn_weight in weight_dict:
        h5f.create_dataset('bn_weight', data=weight_dict[key_bn_weight])
        idx += 1
    if key_bn_bias in weight_dict:
        h5f.create_dataset('bn_bias', data=weight_dict[key_bn_bias])
        idx += 1
    if key_running_mean in weight_dict:
        h5f.create_dataset('running_mean', data=weight_dict[key_running_mean])
        idx += 1
    if key_running_var in weight_dict:
        h5f.create_dataset('running_var', data=weight_dict[key_running_var])
        idx += 1
    h5f.close()
print('Ori total params ' + str(ori_idx))
print('Total params '+str(idx))
