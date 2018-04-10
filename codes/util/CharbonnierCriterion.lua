local CharbonnierCriterion, parent = torch.class('nn.CharbonnierCriterion', 'nn.Criterion')

function CharbonnierCriterion:__init(eps)
   parent.__init(self)
   self.eps = eps
end

function CharbonnierCriterion:updateOutput(input, target)
   local Tensor_sum = torch.sum(torch.sqrt(torch.pow(torch.add(input, -target), 2)+self.eps*self.eps))
   self.output = Tensor_sum/(input:nElement())
   return self.output
end

function CharbonnierCriterion:updateGradInput(input, target)
   local denominator = torch.add(input, -target)
   local numerator = torch.sqrt(torch.pow(denominator,2)+self.eps*self.eps)
   self.gradInput = torch.cdiv(denominator, numerator)/input:nElement()
   return self.gradInput
end
