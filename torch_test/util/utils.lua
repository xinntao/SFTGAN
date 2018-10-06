--
-- utils
-- PSNR, SSIM, modcrop
-- Xintao Wang

require 'image'
local utils = {}
function utils.list_nngraph_modules(g)
   local om = {}
   for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module -- nngraph module
      if m then -- cannot support nngraph containing nngraph
         if m.modules then -- nngraph module is a nn.Sequential container
            local seq = m
            local seq_list = seq:listModules() -- containing all the containers(sub-modules) and the minimum unit-layer
            for k,v in ipairs(seq_list) do
               table.insert(om, v)
            end
         else -- pure nngraph module
            table.insert(om, m)
         end
      end
   end
   return om
end

function utils.listModules(net)
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end

function utils.netParams(net)
   local N_layer = 0 -- number of layers with params
   local module_list = utils.listModules(net)
   local total_params = 0
   for k,v in ipairs(module_list) do
      if v.weight then
         N_layer = N_layer + 1
         total_params = total_params + v.weight:nElement()
      end
      if v.bias then
         total_params = total_params + v.bias:nElement()
      end
   end
   return total_params, N_layer
end

function utils.calculate_PSNR(img1, img2)
   -- can use for gray(2 or 3 channels) and color(3 channels) images
   -- input range: 1 or 255
   assert(img1:dim()>=2 and img2:dim()>=2, 'input must be 2(gray), 3(gray or color) or 4(with batch size 1) dims.')
   img1, img2 = img1:double(), img2:double()
   local diff = torch.add(img1, -img2)
   local rmse = math.sqrt(torch.mean(torch.pow(diff,2)))
   local psnr
   -- if torch.max(img1) > 2 then
   --    psnr = 20*math.log10(255/rmse)
   -- else
      psnr = 20*math.log10(1/rmse)
   -- end
   return psnr
end

function utils.calculate_PSNR_batch(b1, b2)
   assert(b1:dim() == 4, 'not a batch')
   assert(b1:size(1) == b2:size(1), 'two batches have different batch sizes: '..b1:size(1)..' vs '..b2:size(1))
   local batch_sz = b1:size(1)
   local PSNR_sum = 0
   for i = 1, batch_sz do
      PSNR_sum = utils.calculate_PSNR(b1[i], b2[i]) + PSNR_sum
   end
   return PSNR_sum/batch_sz
end

function utils.SSIM(img1, img2)
  --[[
   %This is an implementation of the algorithm for calculating the
   %Structural SIMilarity (SSIM) index between two images. Please refer
   %to the following paper:
   %
   %Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
   %quality assessment: From error visibility to structural similarity"
   %IEEE Transactios on Image Processing, vol. 13, no. 4, pp.600-612,
   %Apr. 2004.
   %
   %Input : (1) img1: the first image being compared
   %        (2) img2: the second image being compared
   %        (3) K: constants in the SSIM index formula (see the above
   %            reference). defualt value: K = [0.01 0.03]
   %        (4) window: local window for statistics (see the above
   %            reference). default widnow is Gaussian given by
   %            window = fspecial('gaussian', 11, 1.5);
   %        (5) L: dynamic range of the images. default: L = 255
   %
   %Output:     mssim: the mean SSIM index value between 2 images.
   %            If one of the images being compared is regarded as
   %            perfect quality, then mssim can be considered as the
   %            quality measure of the other image.
   %            If img1 = img2, then mssim = 1.]]
--[[
   if img1:size(1) > 2 then
    img1 = image.rgb2y(img1)
    img1 = img1[1]
    img2 = image.rgb2y(img2)
    img2 = img2[1]
   end
--]]
   -- place images between 0 and 255.
   if torch.max(img1) < 2 then
      img1:mul(255)
      img2:mul(255)
   end
   img1 = img1:double()
   img2 = img2:double()

   local K1 = 0.01;
   local K2 = 0.03;
   local L = 255;

   local C1 = (K1*L)^2;
   local C2 = (K2*L)^2;
   local window = image.gaussian(11, 1.5/11,0.0708, true):double();
   window = window:div(torch.sum(window));

   local mu1 = image.convolve(img1, window, 'valid')
   local mu2 = image.convolve(img2, window, 'valid')

   local mu1_sq = torch.cmul(mu1,mu1);
   local mu2_sq = torch.cmul(mu2,mu2);
   local mu1_mu2 = torch.cmul(mu1,mu2);

   local sigma1_sq = image.convolve(torch.cmul(img1,img1),window,'valid')-mu1_sq
   local sigma2_sq = image.convolve(torch.cmul(img2,img2),window,'valid')-mu2_sq
   local sigma12 =  image.convolve(torch.cmul(img1,img2),window,'valid')-mu1_mu2

   local ssim_map = torch.cdiv( torch.cmul((mu1_mu2*2 + C1),(sigma12*2 + C2)) ,
     torch.cmul((mu1_sq + mu2_sq + C1),(sigma1_sq + sigma2_sq + C2)));
   local mssim = torch.mean(ssim_map);
   return mssim
end

function utils.calculate_SSIM(img1_in, img2_in)
   local img1 = img1_in:clone()
   local img2 = img2_in:clone()
   local ssim_sum_perimg = 0
   if img1:dim() == 2 and img2:dim() == 2  then -- gray
      return utils.SSIM(img1, img2)
   elseif img1:dim() == 3 and img2:dim() ==3 then -- color mean
      local n_channel = img1:size(1)
      for i=1,n_channel do
         local ssim = utils.SSIM(img1[i], img2[i])
         ssim_sum_perimg = ssim_sum_perimg + ssim
      end
      return ssim_sum_perimg/n_channel
   end
end

function utils.calculate_SSIM_batch(b1_in, b2_in)
   local b1 = b1_in:clone()
   local b2 = b2_in:clone()
   assert(b1:dim() == 4, 'not a batch')
   assert(b1:size(1) == b2:size(1), 'two batches have different batch sizes: '..b1:size(1)..' vs '..b2:size(1))
   local batch_sz = b1:size(1)
   local SSIM_sum = 0
   for i = 1, batch_sz do
      SSIM_sum = utils.calculate_SSIM(b1[i], b2[i]) + SSIM_sum
   end
   return SSIM_sum/batch_sz
end

function utils.modcrop(t, scale, mode)
   -- t: H*W tensor / C*H*W (H*W*C) tensor, range doesnot matter
   -- scale: number
   -- mode: CHW or HWC
   -- return cropped tensor
   mode = mode or 'HWC'
   local new_t
   if t:nDimension() == 2 then -- H*W tensor
      local H, W = t:size(1), t:size(2)
      local H_r, W_r = H % scale, W % scale
      new_t = t[{{1,H-H_r},{1,W-W_r}}]
   elseif t:nDimension() == 3 then
      if 'CHW' == mode then -- C*H*W
         local H, W = t:size(2), t:size(3)
         local H_r, W_r = H % scale, W % scale
         new_t = t[{{},{1,H-H_r},{1,W-W_r}}]
      elseif 'HWC' == mode then -- H*W*C
         local H, W = t:size(1), t:size(2)
         local H_r, W_r = H % scale, W % scale
         new_t = t[{{1,H-H_r},{1,W-W_r},{}}]
      else
         error('invalid mode: '..mode)
      end
   else
      error('invalid dimensions of input tensor:  ' .. t:nDimension())
   end
   return new_t
end
function utils.rm_border(t, border)
   -- t: C*H*W tensor or B*C*H*W tensor
   if t:dim() == 3 then
      local tmp = t:narrow(2, border+1, t:size(2)-2*border)
      return tmp:narrow(3, border+1, tmp:size(3)-2*border):contiguous()
   elseif t:dim() == 4 then
      local tmp = t:narrow(3, border+1, t:size(3)-2*border)
      return tmp:narrow(4, border+1, tmp:size(4)-2*border):contiguous()
   elseif t:dim() == 2 then
      local tmp = t:narrow(1, border+1, t:size(1)-2*border)
      return tmp:narrow(2, border+1, tmp:size(2)-2*border):contiguous()
   end
end
local function _img_derivative(img)
-- input 1*H*W
-- filter 4*1*3*3
-- output F*H*W
   local filter = torch.Tensor(4,1,3,3)
   filter[1] = torch.Tensor({{-1,-2,-1},{0,0,0},{1,2,1}}) -- sobel operator, second derivative
   filter[2] = torch.Tensor({{-1,0,1},{-2,0,2},{-1,0,1}})
   filter[3] = torch.Tensor({{1,0,-1},{1,0,-1},{1,0,-1}}) -- prewitt operator, first derivative
   filter[4] = torch.Tensor({{1,1,1},{0,0,0},{-1,-1,-1}})
   local rlt = torch.conv2(img, filter, 'F')
   rlt = rlt:narrow(2, 2, img:size(2)):narrow(3, 2, img:size(3))
   return rlt
end
local function _img_derivative_SP(img)
-- input H*W
-- output F*H*W
   local f1_h = torch.Tensor({{1,0,-1}})
   local f1_v = torch.Tensor({{1},{0},{-1}})
   local f2_h = torch.Tensor({{1,0,-2,0,1}})
   local f2_v = torch.Tensor({{1},{0},{-2},{0},{1}})

   local r1_h = torch.conv2(img, f1_h, 'F')
   r1_h = r1_h:view(1, table.unpack(r1_h:size():totable()))
   r1_h = r1_h:narrow(3, 2, img:size(2))
   local r1_v = torch.conv2(img, f1_v, 'F')
   r1_v = r1_v:view(1, table.unpack(r1_v:size():totable()))
   r1_v = r1_v:narrow(2, 2, img:size(1))
   local r2_h = torch.conv2(img, f2_h, 'F')
   r2_h = r2_h:view(1, table.unpack(r2_h:size():totable()))
   r2_h = r2_h:narrow(3, 3, img:size(2))
   local r2_v = torch.conv2(img, f2_v, 'F')
   r2_v = r2_v:view(1, table.unpack(r2_v:size():totable()))
   r2_v = r2_v:narrow(2, 3, img:size(1))
   local rlt = torch.cat({r1_h, r1_v, r2_h, r2_v},1)
   return rlt
end
function utils.batch_img_derivative(input)
   -- input B*1*H*W
   -- output B*F*H*W
   local B, C, H, W = table.unpack(input:size():totable())
   local output = torch.Tensor(B, 4, H, W)
   for i=1, B do
      output[i] = _img_derivative(input[i])
   end
   return output
end
function utils.batch_img_derivative_SP(input)
   -- input B*1*H*W
   -- output B*F*H*W
   local B, C, H, W = table.unpack(input:size():totable())
   local output = torch.Tensor(B, 4, H, W)
   for i=1, B do
      output[i] = _img_derivative_SP(input[i][1])
   end
   return output
end
function utils.rotation(img, degree)
   if degree == 90 then
      return image.hflip(img:transpose(2,3):contiguous()):contiguous()
   elseif degree == 180 then
      return image.vflip(image.hflip(img):contiguous()):contiguous()
   elseif degree == 270 then
      return image.vflip(img:transpose(2,3):contiguous()):contiguous()
   else
      return img
   end
end

function utils.get_bilinear_kernel(k)
   -- return a bilinear kernel
   -- reference: LapSRN code
   local factor = math.floor((k + 1) / 2)
   local center
   if k % 2 == 1 then
      center = factor
   else
      center = factor + 0.5
   end
   local C = torch.range(1, k)
   local vector = torch.Tensor(k):fill(1) - torch.abs(C-center)/factor
   local kernel = torch.Tensor(k,k):zero()
   return kernel:addr(vector,vector):view(1,1,k,k)
end

function utils.get_bilinear_kernel_channels(k, channels)
   local kernels = torch.Tensor(channels, channels, k, k):zero()
   for i = 1, channels do
      kernels[i][i]:copy(utils.get_bilinear_kernel(k))
   end
   return kernels
end

function clip_gradients(t, thres, t_name)
   -- square sum
   local l2norm = torch.sum(torch.pow(t,2))--/t:nElement()
   l2norm = torch.sqrt(l2norm)
   if l2norm <= thres then
      return t
   else
      print(t_name..' clip gradiens. L2_norm is '.. l2norm)
      return torch.mul(t, thres/l2norm)
   end
end
function search_node(net)
   for k, v in ipairs(net.forwardnodes) do
      if v.data then
         if v.data.annotations.name == 'BilinearDeconv' then print(k)  end
      end
   end
end
return utils
