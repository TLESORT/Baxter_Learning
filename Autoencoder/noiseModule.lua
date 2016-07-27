local noiseModule, parent = torch.class('nn.noiseModule', 'nn.Module')

function noiseModule:__init()
  parent.__init(self)
end

function noiseModule:updateOutput(input)
	self.output:resizeAs(input)
	self.output:copy(input)
	self.Noise=torch.rand(input:size()):cuda()
	self.output:add(self.Noise)
	return self.output
end

function noiseModule:updateGradInput(input, gradOutput)
  if self.gradInput then
      self.gradInput:resizeAs(gradOutput)
      self.gradInput:copy(gradOutput)
    return self.gradInput
  end
end

