local sigmoid_cross_entropy_with_logits, parent = torch.class('nn.sigmoid_cross_entropy_with_logits', 'nn.Criterion')

function sigmoid_cross_entropy_with_logits:__init()
   parent.__init(self)
end

function sigmoid_cross_entropy_with_logits:updateOutput(input, target)
   local tmp = torch.log(1 + torch.exp(-torch.abs(input)))
   local Tensor_sum = torch.sum(torch.clamp(input, 0, math.huge) - torch.cmul(input, target) + tmp)
   self.output = Tensor_sum/(input:nElement())
   return self.output
end

function sigmoid_cross_entropy_with_logits:updateGradInput(input, target)
   local sign = torch.sign(input)
   local exp_sign_x = torch.exp(-torch.cmul(sign, input))
   local tmp = torch.cdiv(-torch.cmul(sign, exp_sign_x), 1+exp_sign_x)
   self.gradInput = (0.5+0.5*sign-target+tmp)/input:nElement()
   return self.gradInput
end
