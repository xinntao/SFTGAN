require 'nn'
local hdf5 = require 'hdf5'
torch.setdefaulttensortype('torch.FloatTensor')
require 'image'

local function SpatialConvBatchNormReLU(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, has_ReLU, dilation)
   local has_ReLU = has_ReLU or 1
   local dilation = dilation or 0
   local std_epsilon = 1e-5
   local conv
   if dilation == 0 then
      conv = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   else
      conv = nn.SpatialDilatedConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, dilation, dilation)
   end
   conv:noBias()
   local module = nn.Sequential()
   module:add(conv)
   module:add(nn.SpatialBatchNormalization(nOutputPlane, std_epsilon, nil, true))
   if has_ReLU == 1 then
      module:add(nn.ReLU(true))
   end
   return module
end

local function Res131(nIn, nMid, nOut, dilation, stride)
   local dilation = dilation or 0
   local stride = stride or 1
   local convs = nn.Sequential()
   convs:add(SpatialConvBatchNormReLU(nIn, nMid, 1, 1, 1, 1))
   if dilation == 0 then
      convs:add(SpatialConvBatchNormReLU(nMid, nMid, 3, 3, stride, stride, 1, 1))
   elseif dilation == 2 then
      convs:add(SpatialConvBatchNormReLU(nMid, nMid, 3, 3, stride, stride, 2, 2, 1, 2))
   elseif dilation == 4 then
      convs:add(SpatialConvBatchNormReLU(nMid, nMid, 3, 3, stride, stride, 4, 4, 1, 4))
   end
   convs:add(SpatialConvBatchNormReLU(nMid, nOut, 1, 1, 1, 1, 0, 0, 0)) -- no ReLU
   local shortcut
   if nIn == nOut then
      shortcut = nn.Identity()
   else
      shortcut = nn.Sequential():add(SpatialConvBatchNormReLU(nIn, nOut, 1, 1, stride, stride, 0, 0, 0))
   end
   local returned_net =  nn.Sequential()
            :add(nn.ConcatTable()
               :add(convs)
               :add(shortcut))
            :add(nn.CAddTable())
            :add(nn.ReLU(true))
   return returned_net
end

local function OutdoorSceneSeg(nclass)
   local nclass = nclass or 8
   local net = nn.Sequential()
   -- conv1
   net:add(SpatialConvBatchNormReLU(3, 64, 3, 3, 2, 2, 1, 1)) -- /2
   net:add(SpatialConvBatchNormReLU(64, 64, 3, 3, 1, 1, 1, 1))
   net:add(SpatialConvBatchNormReLU(64, 128, 3, 3, 1, 1, 1, 1))
   local pool = nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0)
   pool:ceil()
   net:add(pool) -- /2
   -- -- conv2, 3 blocks
   net:add(Res131(128, 64, 256))
   net:add(Res131(256, 64, 256))
   net:add(Res131(256, 64, 256))
   -- conv3, 4 blocks
   net:add(Res131(256, 128, 512, 0, 2)) -- /2
   net:add(Res131(512, 128, 512))
   net:add(Res131(512, 128, 512))
   net:add(Res131(512, 128, 512))
   -- -- conv4 23 blocks
   net:add(Res131( 512, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   net:add(Res131(1024, 256, 1024, 2))
   -- conv5
   net:add(Res131(1024, 512, 2048, 4))
   net:add(Res131(2048, 512, 2048, 4))
   net:add(Res131(2048, 512, 2048, 4))
   net:add(SpatialConvBatchNormReLU(2048, 512, 3, 3, 1, 1, 1, 1))
   net:add(nn.Dropout(0.1))
   -- conv6
   net:add(nn.SpatialConvolution(512, 8, 1, 1))
   -- deconv
   -- Torch cannot support group deconv, therefore we use initialization to achieve it.
   local deconv = nn.SpatialFullConvolution(8, 8, 16, 16, 8, 8, 4, 4)
   deconv:noBias()
   net:add(deconv)
   -- softmax
   net:add(nn.SoftMax())
   return net
end

----------------
-- Load --
----------------

local function load_conv2d(module, name)
   -- local name = name or 'Conv2d_1a_3x3'
   local h5f = hdf5.open('dump/'..name..'.h5', 'r')

   local conv = module:get(1) -- Spatial Convolution
   local weights = h5f:read("weight"):all()
   conv.weight:copy(weights)

   local bn = module:get(2) -- Spatial Batch Normalization
   local bn_weight = h5f:read("bn_weight"):all()
   bn.weight:copy(bn_weight)
   local bn_bias = h5f:read("bn_bias"):all()
   bn.bias:copy(bn_bias)
   local running_mean = h5f:read("running_mean"):all()
   bn.running_mean:copy(running_mean)
   local running_var = h5f:read("running_var"):all()
   bn.running_var:copy(running_var)

   h5f:close()
end

local function load_conv2d_nobn(module, name)
   -- local name = name or 'Conv2d_1a_3x3'
   local h5f = hdf5.open('dump/'..name..'.h5', 'r')
   local conv = module -- Spatial Convolution

   local weights = h5f:read("weight"):all()
   conv.weight:copy(weights)
   local biases = h5f:read("bias"):all()
   conv.bias:copy(biases)

   h5f:close()
end

local function load_res131(module, name, has_proj)
   local has_proj = has_proj or 0
   load_conv2d(module:get(1):get(1):get(1), name..'_1x1_reduce')
   load_conv2d(module:get(1):get(1):get(2), name..'_3x3')
   load_conv2d(module:get(1):get(1):get(3), name..'_1x1_increase')
   if has_proj == 1 then
      load_conv2d(module:get(1):get(2):get(1), name..'_1x1_proj')
   end
end

local function load(net)
   -- conv 1
   load_conv2d(net:get(1), 'conv1_1_3x3_s2')
   load_conv2d(net:get(2), 'conv1_2_3x3')
   load_conv2d(net:get(3), 'conv1_3_3x3')
   -- -- conv2
   load_res131(net:get(5), 'conv2_1', 1)
   load_res131(net:get(6), 'conv2_2')
   load_res131(net:get(7), 'conv2_3')
   -- conv3
   load_res131(net:get(8), 'conv3_1', 1)
   load_res131(net:get(9), 'conv3_2')
   load_res131(net:get(10), 'conv3_3')
   load_res131(net:get(11), 'conv3_4')
   -- conv4
   load_res131(net:get(12), 'conv4_1', 1)
   for i=2, 23 do
      load_res131(net:get(11+i), 'conv4_'..i)
   end
   -- conv5
   load_res131(net:get(35), 'conv5_1', 1)
   load_res131(net:get(36), 'conv5_2')
   load_res131(net:get(37), 'conv5_3')
   load_conv2d(net:get(38), 'conv5_4')
   -- conv6
   load_conv2d_nobn(net:get(40), 'conv6_Outdoor')

   -- deconv
   local h5f = hdf5.open('dump/deconv_Outdoor.h5', 'r')
   local conv = net:get(41)

   local weights = h5f:read("weight"):all()
   conv.weight:fill(0)
   for i=1, 8 do
      conv.weight[{{i},{i},{},{}}]:copy(weights[{{i},{1},{},{}}])
   end
   h5f:close()
end

------------------------------------------------------------------
-- Main
------------------------------------------------------------------

local function main()
   local opt = {
      cuda = true
   }
   local net = OutdoorSceneSeg()
   print(net)
   load(net)
   print('loaded')

   -- test
   require 'cunn'
   require 'cutorch'
   require 'cudnn'
   net:cuda()
   net:evaluate()
--    local img = image.load('test.png', 3, 'float')
--    img = img:index(1,torch.LongTensor({3,2,1}))
--    img = img * 255
--    img[1]:add(- 103.939)
--    img[2]:add(- 116.779)
--    img[3]:add(- 123.68)
--    -- RGB to BGR
--    local input = img:view(1, table.unpack(img:size():totable())):cuda()
--    local output = net:forward(input)
--    torch.save('test.t7', output:float())
   -- print(output)
   torch.save('../../models/OutdoorSceneSeg_bic_iter_30000.t7', net:clearState():float())
end

main()