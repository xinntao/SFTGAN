-- test code for segmentation

require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'

require 'paths'
require 'image'
require './util/cudnn_convert_custom'

local model_path = '../models/OutdoorSceneSeg_bic_iter_30000.t7'
local test_img_folder = '../data/OutdoorSceneTest300_bicx4'
local save_prob_path = '../data/save_prob'
local save_byteimg_path = '../data/save_byteimg'
local save_colorimg_path = '../data/save_colorimg'

-- create folders
if not paths.dirp(save_prob_path) then
    paths.mkdir(save_prob_path)
end
if not paths.dirp(save_byteimg_path) then
    paths.mkdir(save_byteimg_path)
end
if not paths.dirp(save_colorimg_path) then
    paths.mkdir(save_colorimg_path)
end

-- load model
local net = torch.load(model_path)
cudnn_convert_custom(net, cudnn)
-- cudnn.SoftMax will result in wrong SoftMax behaviour
-- cudnn can reduce GPU memory compared with cunn
net:cuda()
net:evaluate()

local idx = 0

-- lookup_table is a double RGB tensor #categories * 3
local lookup_table = torch.Tensor({
   {153, 153, 153}, -- 0, background
   {0, 255, 255}, --1, sky
   {109, 158, 235}, --2, water
   {183, 225, 205}, --3, grass
   {153, 0, 255}, -- 4, mountain
   {17, 85, 204}, -- 5, building
   {106, 168, 79}, -- 6, plant
   {224, 102, 102}, -- 7, animal
   {255, 255, 255}, -- 8/255, void
   })
lookup_table = lookup_table / 255

for f in paths.files(test_img_folder, '.+%.%a+') do
    idx = idx + 1
    local ext = paths.extname(f)
    local img_base_name = paths.basename(f, ext)
    print(idx, img_base_name)
    local img = image.load(paths.concat(test_img_folder, f), 3, 'float')
    img = img:index(1,torch.LongTensor({3,2,1})) -- BGR
    img = img * 255
    img[1]:add(- 103.939)
    img[2]:add(- 116.779)
    img[3]:add(- 123.68)
    local input = img:view(1, table.unpack(img:size():totable())):cuda()
    local output = net:forward(input)
    output = output:float():contiguous()
    -- prob
    torch.save(paths.concat(save_prob_path, img_base_name..'.t7'), output)
    -- byte img
    output = output:squeeze()
    _, argmax = torch.max(output, 1)
    argmax = argmax:squeeze():add(-1):byte()
    image.save(paths.concat(save_byteimg_path, img_base_name..'.png'), argmax)
    -- color img
    local argmx = argmax:add(1)
    local im_h, im_w = argmx:size(1), argmx:size(2)
    local color = torch.Tensor(3, im_h, im_w):zero() -- black
    for i = 1,8 do -- have added 1
        local mask = torch.eq(argmx, i)
        color:select(1,1):maskedFill(mask, lookup_table[i][1]) -- R
        color:select(1,2):maskedFill(mask, lookup_table[i][2]) -- G
        color:select(1,3):maskedFill(mask, lookup_table[i][3]) -- B
    end
        -- void
    local mask = torch.eq(argmx, 256)
    color:select(1,1):maskedFill(mask, lookup_table[9][1]) -- R
    color:select(1,2):maskedFill(mask, lookup_table[9][2]) -- G
    color:select(1,3):maskedFill(mask, lookup_table[9][3]) -- B
    image.save(paths.concat(save_colorimg_path, img_base_name..'.png'), color)

    img = nil; input = nil; output = nil
    net:clearState()
    collectgarbage()
end