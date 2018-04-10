require 'torch'
require 'nn'
require 'cudnn'
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
cudnn_convert_custom(condition_net, cudnn)
cudnn_convert_custom(sr_net, cudnn)
condition_net:evaluate()
sr_net:evaluate()
condition_net = condition_net:cuda()
sr_net = sr_net:cuda()

local idx = 0
for f in paths.files(test_img_path, '.+%.%a') do
    idx = idx + 1
    local ext = paths.extname(f)
    local img_base_name = paths.basename(f, ext)
    local img_HR = image.load(paths.concat(test_img_path, f), 3, 'float')
    img_HR = utils.modcrop(img_HR, 8, 'CHW')
    local seg_input = torch.load(paths.concat(seg_prob_path, img_base_name..'_bic.t7'))
    -- generate LR image
    local img_LR = matlab_functions.imresize(img_HR, 1/4, true)
    local seg_input = seg_input:cuda()
    local img_input = img_LR:view(1, table.unpack(img_LR:size():totable())):cuda()
    local shared_condition = condition_net:forward(seg_input)
    local G_output = sr_net:forward({img_input, shared_condition}):squeeze():float()

    local new_name = img_base_name..'_'..model_name..'.png'
    image.save(paths.concat(save_path, new_name), G_output)
    print(idx, new_name)

    img_HR = nil;
    seg_input = nil; img_input = nil;
    shared_condition = nil; G_output = nil
    condition_net:clearState()
    sr_net:clearState()
    collectgarbage()
end