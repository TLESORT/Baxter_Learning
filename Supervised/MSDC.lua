local MSDCriterion, parent = torch.class('nn.MSDCriterion','nn.Criterion')

function MSDCriterion:__init(sizeAverage)
  parent.__init(self)
  self.buffer = torch.Tensor()
  self.sizeAverage = sizeAverage or sizeAverage==nil
  self.mask = nil
  self.maskView = torch.Tensor()
end

function MSDCriterion:updateOutput(inputs)
  assert(#inputs==2 and inputs[1]:isSameSizeAs(inputs[2]), 'input must be a table of 2 tensors of identical dimension')
  
  self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, inputs)

  if self.mask then  
    if not self.maskView:isSameSizeAs(inputs[1]) then
      self.maskView = self.mask.new(self.mask)
      for i=1,inputs[1]:dim()-self.mask:dim() do  
        self.maskView = nn.utils.addSingletonDimension(self.maskView) 
      end
      self.maskView = self.maskView:expandAs(inputs[1])
    end
    
    self.buffer:resizeAs(inputs[1]):add(inputs[1],-1,inputs[2])  
    self.gradInput[1]:cmul(self.buffer,self.maskView)  
    self.buffer:cmul(self.gradInput[1])        
    self.output = self.buffer:sum()  
  else  
    self.gradInput[1]:add(inputs[1],-1,inputs[2])  
    self.buffer:resizeAs(inputs[1]):pow(self.gradInput[1],2)
    self.output = self.buffer:sum()
  end

  if self.sizeAverage then
    local n = inputs[1]:nElement()
    self.output = self.output/n
  end

  return self.output
end

function MSDCriterion:updateGradInput(inputs)
  -- This function assumes it is called after forward has been called (self.gradInput calculus starts in forward) for optimization purposes
  if self.sizeAverage then
    local n = inputs[1]:nElement()
    self.gradInput[1]:mul(2/n)
  else
    self.gradInput[1]:mul(2)
  end
  self.gradInput[2]:mul(self.gradInput[1],-1)
  return self.gradInput
end


function MSDCriterion:clearState()
  nn.utils.clear(self, 'buffer', 'maskView')
  return parent.clearState(self)
end

