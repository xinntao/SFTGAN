require 'torch'
require 'nn'
-- require 'cudnn'
require 'cunn'
require 'nngraph'
require 'paths'
require 'image'
require './util/cudnn_convert_custom'
local matlab_functions = require './util/matlab_functions'
local utils = require './util/utils'

local model_name = 'sft-gan'
local test_img_folder = 'OST300'

local model_path = '../models/'..model_name..'.t7'
local test_img_path = '../data/'..test_img_folder
local seg_prob_path = '../data/'..test_img_folder..'_segprob'
local save_path = '../data/rlt_'..test_img_folder..'_'..model_name
if not paths.dirp(save_path) then paths.mkdir(save_path) end

local sft_gan = torch.load('../models/SFT-GAN.t7')
local condition_net = sft_gan['condition_net']
local sr_net = sft_gan['sr_net']
-- cudnn_convert_custom(condition_net, cudnn)
-- cudnn_convert_custom(sr_net, cudnn)
condition_net:replace(function(module)
   if torch.typename(module) == 'nn.LeakyReLU' then
      neg = module.negval
      print('replace leakyrelu with neg: '..neg)
      return nn.LeakyReLU(neg, true)
   elseif torch.typename(module) == 'nn.ReLU' then
      print('replace relu')
      return nn.ReLU(true)
   else
      return module
   end
end)

sr_net:replace(function(module)
   if torch.typename(module) == 'nn.LeakyReLU' then
      neg = module.negval
      print('replace leakyrelu with neg: '..neg)
      return nn.LeakyReLU(neg, true)
   elseif torch.typename(module) == 'nn.ReLU' then
      print('replace relu')
      return nn.ReLU(true)
   else
      return module
   end
end)

for i,node in ipairs(sr_net.forwardnodes) do
   local m = node.data.module -- nngraph module
   if m then -- cannot support nngraph containing nngraph
      -- print(m)
   end
end

-- local sft_gan_inplace = {}
-- sft_gan_inplace['condition_net'] = condition_net
-- sft_gan_inplace['sr_net'] = sr_net
-- torch.save('../models/SFT-GAN_inplace.t7', sft_gan_inplace)
print(sft_gan)