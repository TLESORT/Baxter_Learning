function doStuff_temp(Model,criterion, gradParameters,im1, im2)
	State2=Model:forward(im2):clone()--should be before State1
	State1=Model:forward(im1)
	loss=criterion:forward({State1,State2})
	GradOutputs=criterion:backward({State1,State2})

	-- calculer les gradients pour les deux images
	Model:backward(im1,GradOutputs[1])
	return loss, gradParameters
end

function doStuff_Caus(Model,criterion, gradParameters,im1,im2)
	
	State1=Model:forward(im1)
	State2=Model2:forward(im2)
	loss=criterion:forward({State1,State2})
	GradOutputs=criterion:backward({State1,State2})

	-- calculer les gradients pour les deux images
	Model:backward(im1,(-1)*GradOutputs[1]*math.exp(-loss))
	return math.exp(-loss), gradParameters
end

function doStuff_Prop(Model,criterion,gradParameters,im1,im2,im3,im4)

	State2=Model:forward(im2):clone()
	State3=Model:forward(im3):clone()
	State4=Model:forward(im4):clone()
	State1=Model:forward(im1)

	Module= nn.Sequential()
	Module:add(nn.CSubTable())
	Module:add(nn.Square())
	Module:add(nn.Sum(1))
	Module=Module:cuda()

	Module2=Module:clone()

	delta1=Module:forward({State1,State2})
	delta2=Module2:forward({State3,State4})

	loss=criterion:forward({delta1,delta2})
	GradOutputs=criterion:backward({delta1,delta2})
	GradOutputs2=Module:backward({State1,State2},GradOutputs[1])

	-- calculer les gradients pour les deux images
	Model:backward(im1,GradOutputs2[1])

	return loss, gradParameters
end

function doStuff_Rep(Model,criterion,gradParameters,im1,im2,im3,im4)


	State2=Model:forward(im2):clone()
	State3=Model:forward(im3):clone()
	State4=Model:forward(im4):clone()
	State1=Model:forward(im1)

	Module= nn.Sequential()
	Module:add(nn.CSubTable())
	Module:add(nn.Square())
	Module:add(nn.Sum(1))
	Module=Module:cuda()

	Module2=Module:clone()

	delta1=Module:forward({State1,State2})
	delta2=Module2:forward({State3,State4})

	criterion2=criterion:clone()

	loss=criterion:forward({delta1,delta2})
	loss2=criterion2:forward({State1,State3})
	loss2=math.exp(-loss2)

	GradS=criterion2:backward({State1,State3})

	-- BACKWARD
	GradOutputs=criterion:backward({delta1,delta2})
	GradOutputs2=Module:backward({State1,State2},GradOutputs[1])
	GradS1=GradOutputs2[1]*loss2+GradS[1]*loss

	Model:backward(im1, GradS1)

	return loss2*loss, gradParameters
end
