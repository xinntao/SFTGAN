-- implement the matlab sytle functions
   -- imresize (now only support bicubic)
      -- the speed is not optimized, (x10~x100 slower than matlab built-in function)
   -- rgb2ycbcr & ycbcr2rgb

local matlab_functions = {}

local function cubic(x)
   -- x is a H*W tensor
   local absx = torch.abs(x)
   local absx2 = torch.pow(absx,2)
   local absx3 = torch.pow(absx,3)
   local add_item_1 = torch.cmul((1.5*absx3 - 2.5*absx2 + 1), torch.le(absx,1):typeAs(absx))
   local add_item_2 = (-0.5*absx3 + 2.5*absx2 - 4*absx +2):cmul(torch.gt(absx,1):typeAs(absx)):cmul(torch.le(absx,2):typeAs(absx))
   return add_item_1 + add_item_2
end

local function calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing)
   if (scale < 1) and (antialiasing) then
      -- Use a modified kernel to simultaneously interpolate and antialias.
      -- @(x) scale * kernel(scale * x);
      kernel_width = kernel_width / scale
   end
   -- Output-space coordinates
   local x = torch.linspace(1, out_length, out_length)
   -- Input-space coordinates. (center and proper interval)
   local u = x/scale + 0.5 * (1 - 1/scale)
   -- What is the left-most pixel that can be involved in the computation?
   local left = torch.floor(u - kernel_width/2)
   -- What is the maximum number of pixels that can be involved in the
   -- computation?  Note: it's OK to use an extra pixel here; if the
   -- corresponding weights are all zero, it will be eliminated at the end
   -- of this function.
   local P = math.ceil(kernel_width) + 2
   local indices = torch.add(torch.expand(left:view(out_length,1),out_length,P), torch.linspace(0,P-1,P):view(1,P):expand(out_length,P))
   local weights_pre = torch.add(u:view(out_length,1):expand(out_length,P), -indices)
   -- apply cubic kernel
   local weights
   if (scale < 1) and (antialiasing) then
      weights = scale * cubic(weights_pre*scale)
   else
      weights = cubic(weights_pre)
   end
   -- Normalize the weights matrix so that each row sums to 1.
   local weights_sum = torch.sum(weights,2)
   weights = torch.cdiv(weights, weights_sum:expand(out_length,P))
   -- Mirror out-of-bounds indices; equivalent of doing symmetric padding
   -- different from the Matlab code
   -- less than 0
   indices = torch.abs(indices) + torch.le(indices, 0):typeAs(indices)
   -- larger than in_length
   local larger_tmp = torch.add(indices, -in_length):clamp(0, math.huge)
   indices = indices - 2*larger_tmp + torch.gt(larger_tmp,0):typeAs(indices)
   -- can only consider the first and last column
   local weights_zero_tmp = torch.sum(torch.eq(weights,0), 1)
   if weights_zero_tmp[1] ~= 0 then indices = indices:narrow(2,2,P-1); weights = weights:narrow(2,2,P-1) end
   if weights_zero_tmp[-1] ~= 0 then indices = indices:narrow(2,1,P-2); weights = weights:narrow(2,1,P-2) end
   weights = weights:contiguous()
   indices = indices:contiguous()
   return weights, indices
end

function matlab_functions.imresize(img, scale, is_antialiasing)
   -- Now only support scale, and the scale should be the same in H and W dimension
   local antialiasing = is_antialiasing or true
   -- input: img: CHW RGB [0,1]
   img = img * 255
   local in_C, in_H, in_W = table.unpack(img:size():totable())
   local out_C, out_H, out_W = in_C, math.ceil(in_H*scale), math.ceil(in_W*scale)
   local kernel_width = 4
   local kernel = 'cubic'

   -- Return the desired dimension order for performing the resize.  The
   -- strategy is to perform the resize first along the dimension with the
   -- smallest scale factor.
   -- Now we do not support this.

   -- get weights and indices
   -- local timer = torch.Timer();timer:reset()
   local weights_H, indices_H = calculate_weights_indices(in_H, out_H, scale, kernel, kernel_width, antialiasing)
   local weights_W, indices_W = calculate_weights_indices(in_W, out_W, scale, kernel, kernel_width, antialiasing)
   -- print(timer:time().real)

   -- imresizemex job
   --[[
   -- torch, vanilla  (1356*2040, 0.25x, 91.13s VS matlab 0.02s. Too too slow)
   -- process H dimension
   local out_1 = torch.Tensor(3, out_H, in_W)
   kernel_width = weights_H:size(2)
   local pixel_input = torch.Tensor(kernel_width):fill(0)
   for i = 1, in_C do
      for j = 1, in_W do
         for k = 1, out_H do
            for m = 1, kernel_width do
               local pixel_idx = indices_H[k][m]
               pixel_input[m] = img[i][pixel_idx][j]
            end
            out_1[i][k][j] = torch.sum(torch.cmul(pixel_input, weights_H[k]))
         end
      end
   end
   -- process W dimension
   local out_2 = torch.Tensor(3, out_H, out_W)
   kernel_width = weights_W:size(2)
   local pixel_input = torch.Tensor(kernel_width):fill(0)
   for i = 1, in_C do
      for j = 1, out_H do
         for k = 1, out_W do
            for m = 1, kernel_width do
               local pixel_idx = indices_W[k][m]
               pixel_input[m] = out_1[i][j][pixel_idx]
            end
            out_2[i][j][k] = torch.sum(torch.cmul(pixel_input, weights_W[k]))
         end
      end
   end
   --]]

   --[[
   -- torch ffi () (1356*2040, 0.25x, 5.26s VS matlab 0.02s. Still too slow)
   -- process H dimension
   local out_1 = torch.Tensor(3, out_H, in_W)
   kernel_width = weights_H:size(2)
   local pixel_input = torch.Tensor(kernel_width):fill(0)

   local out_1_p = torch.data(out_1:contiguous())
   local pixel_input_p = torch.data(pixel_input:contiguous())
   local indices_H_p = torch.data(indices_H:contiguous())
   local img_p = torch.data(img:contiguous())

   for i = 1, in_C do
      for j = 1, in_W do
         for k = 1, out_H do
            for m = 1, kernel_width do
               local pixel_idx = indices_H_p[(k-1)*kernel_width+m-1]
               pixel_input_p[m-1] = img_p[((i-1)*in_H+pixel_idx-1)*in_W+j-1]
            end
            out_1_p[((i-1)*out_H+k-1)*in_W+j-1] = torch.sum(torch.cmul(pixel_input, weights_H[k]))

         end
      end
   end
   -- process W dimension
   local out_2 = torch.Tensor(3, out_H, out_W)
   kernel_width = weights_W:size(2)
   local pixel_input = torch.Tensor(kernel_width):fill(0)

   local out_2_p = torch.data(out_2:contiguous())
   local pixel_input_p = torch.data(pixel_input:contiguous())
   local indices_W_p = torch.data(indices_W:contiguous())

   for i = 1, in_C do
      for j = 1, out_H do
         for k = 1, out_W do
            for m = 1, kernel_width do
               local pixel_idx = indices_W_p[(k-1)*kernel_width+m-1]
               pixel_input_p[m-1] = out_1_p[((i-1)*out_H+j-1)*in_W+pixel_idx-1]
            end
            out_2_p[((i-1)*out_H+j-1)*out_W+k-1] = torch.sum(torch.cmul(pixel_input, weights_W[k]))
         end
      end
   end
   --]]

   ---[[
   local out_2
   if scale == 1/4 or scale == 1/8 or scale == 1/2 then
      -- can only for x1/8, x1/4, x1/2 (w/ or w/o modcrop), not for x 1/3. (1/(2^n) can be OK, but should modify the code.)
      -- torch tensor patch (1356*2040, 0.25x, 0.25s VS matlab 0.025s. ) not stable 1.3s, 0.33s
      local sym_len, stride
      local stride = 1/scale
      if scale == 1/4 then sym_len = 6
      elseif scale == 1/8 then sym_len = 12
      else sym_len = 3
      end

      local sym_len_2 = out_H/scale - in_H
      local img_aug = torch.Tensor(in_C, in_H+2*sym_len+sym_len_2, in_W)
      img_aug:narrow(2, sym_len+1, in_H):copy(img)
      -- do symmetric copy
      for i = 1, sym_len do
         img_aug[{{},{i},{}}] = img[{{},{sym_len-i+1},{}}]
         img_aug[{{},{in_H+sym_len+i},{}}] = img[{{},{in_H-i+1},{}}]
      end
      if sym_len_2 ~= 0 then
         for i = sym_len+1, sym_len+sym_len_2 do
            img_aug[{{},{in_H+sym_len+i},{}}] = img[{{},{in_H-i+1},{}}]
         end
      end

      -- process H dimension
      local out_1 = torch.Tensor(3, out_H, in_W)
      kernel_width = weights_H:size(2)
      for i = 1, out_H do
         local tmp = torch.cmul(img_aug:narrow(2,(i-1)*stride+1,kernel_width), weights_H[i]:view(1,kernel_width,1):expand(in_C,kernel_width,in_W))
         out_1[{{},{i},{}}] = torch.sum(tmp, 2)
      end

      sym_len_2 = out_W/scale - in_W
      local out_1_aug = torch.Tensor(in_C, out_H, in_W+2*sym_len+sym_len_2)
      out_1_aug:narrow(3, sym_len+1, in_W):copy(out_1)
      -- do symmetric copy
      for i = 1, sym_len do
         out_1_aug[{{},{},{i}}] = out_1[{{},{},{sym_len-i+1}}]
         out_1_aug[{{},{},{in_W+sym_len+i}}] = out_1[{{},{},{in_W-i+1}}]
      end
      if sym_len_2 ~= 0 then
         for i = sym_len+1, sym_len+sym_len_2 do
            out_1_aug[{{},{},{in_W+sym_len+i}}] = out_1[{{},{},{in_W-i+1}}]
         end
      end
      --print(out_1_aug);os.exit()
      -- process W dimension
      out_2 = torch.Tensor(3, out_H, out_W)
      kernel_width = weights_W:size(2)
      for i = 1, out_W do
         local tmp = torch.cmul(out_1_aug:narrow(3,(i-1)*stride+1,kernel_width), weights_W[i]:view(1,1,kernel_width):expand(in_C,out_H,kernel_width))
         out_2[{{},{},{i}}] = torch.sum(tmp, 3)
      end

   else
      -- torch tensor index version
      -- have tested x2, x3, x4, x8, x1/3, x1/8(w/o modcrop), x1/4(w/o modcrop), theoretically for any version (xX and w/o modcrop )
      -- torch tensor index (1356*2040, 0.25x, 2.09s VS matlab 0.02s. Still too slow)
      -- process H dimension
      local out_1 = torch.Tensor(3, out_H, in_W)
      kernel_width = weights_H:size(2)
      local tensor_input = torch.Tensor(in_C, kernel_width, in_W)

      for i = 1, out_H do
         for j = 1, kernel_width do
            local idx = indices_H[i][j]
            tensor_input[{{},{j},{}}] = img[{{},{idx},{}}]
            local tmp = torch.cmul(tensor_input, weights_H[i]:view(1,kernel_width,1):expand(in_C,kernel_width,in_W))
            out_1[{{},{i},{}}] = torch.sum(tmp, 2)
         end
      end
      -- process W dimension
      out_2 = torch.Tensor(3, out_H, out_W)
      kernel_width = weights_W:size(2)
      local tensor_input = torch.Tensor(in_C, out_H, kernel_width)
      for i = 1, out_W do
         for j = 1, kernel_width do
           local idx = indices_W[i][j]
            tensor_input[{{},{},{j}}] = out_1[{{},{},{idx}}]
            local tmp = torch.cmul(tensor_input, weights_W[i]:view(1,1,kernel_width):expand(in_C,out_H,kernel_width))
            out_2[{{},{},{i}}] = torch.sum(tmp, 3)
         end
      end
   end
   return out_2:double()/255
end

function matlab_functions.test_imresize()
   require 'image'
   local img = image.load('baboonx4.png', 3, 'double')
   scale = 4
   local timer = torch.Timer();timer:reset()
   local img_out = matlab_functions.imresize(img, scale, true)
   print(timer:time().real)
   image.save('test.png', img_out)
end

function matlab_functions.rgb2ycbcr(img)
   -- img: [0,1], CHW, RGB
   -- return ycbcr: [0,1], CHW, YCbCr
   local img_ycbcr = torch.Tensor(img:size())
   local origT = torch.Tensor({{65.481, 128.553, 24.966},
      {-37.797, -74.203, 112},
      {112, -93.786, -18.214}})
   local origOffset = torch.Tensor({16, 128, 128})
   local scale_T, scale_offset = 1/255, 1/255
   local T = origT*scale_T
   local offset = origOffset*scale_offset
   for p = 1, 3 do
      img_ycbcr[p] = img[1]*T[p][1]+img[2]*T[p][2]+img[3]*T[p][3]+offset[p]
   end
   return img_ycbcr
end

function matlab_functions.ycbcr2rgb(img)
   -- img: [0,1], CHW, YCbCr
   -- return rgb: [0,1], CHW, RGB
   local img_rgb = torch.Tensor(img:size())
   local origT = torch.Tensor({{0.00456621, 0, 0.00625893},
      {0.00456621, -0.00153632, -0.00318811},
      {0.00456621, 0.00791071, 0}})
   local origOffset = torch.Tensor({0.87420242, -0.53166827, 1.08563257})
   local scale_T, scale_offset = 255, 1
   local T = origT*scale_T
   local offset = origOffset*scale_offset
   for p = 1, 3 do
      img_rgb[p] = img[1]*T[p][1]+img[2]*T[p][2]+img[3]*T[p][3]-offset[p]
   end
   return img_rgb
end

return matlab_functions