function doStuff_temp(Model,criterion, gradParameters,im1,im2)
	
	Model2=Model:clone('weight','bias','gradWeight',
			'gradBias','running_mean','running_std')
	State1=Model:forward(im1)
	State2=Model2:forward(im2)
	loss=criterion:forward({State1,State2})
	GradOutputs=criterion:backward({State1,State2})

	-- calculer les gradients pour les deux images
	Model:backward(im1,GradOutputs[1])
	Model2:backward(im2,GradOutputs[2])
	return loss, gradParameters
end

function doStuff_Caus(Model,criterion, gradParameters,im1,im2)
	
	Model2=Model:clone('weight','bias','gradWeight',
			'gradBias','running_mean','running_std')
	State1=Model:forward(im1)
	State2=Model2:forward(im2)
	loss=criterion:forward({State1,State2})
	GradOutputs=criterion:backward({State1,State2})

	-- calculer les gradients pour les deux images
	Model:backward(im1,(-1)*GradOutputs[1]*math.exp(-loss))
	Model2:backward(im2,(-1)*GradOutputs[2]*math.exp(-loss))
	return math.exp(-loss), gradParameters
end

function doStuff_Prop(Model,criterion,gradParameters,im1,im2,im3,im4)

	Model2=Model:clone('weight','bias','gradWeight',
			'gradBias','running_mean','running_std')
	Model3=Model:clone('weight','bias','gradWeight',
			'gradBias','running_mean','running_std')
	Model4=Model:clone('weight','bias','gradWeight',
			'gradBias','running_mean','running_std')


	State1=Model:forward(im1)
	State2=Model2:forward(im2)
	State3=Model3:forward(im3)
	State4=Model4:forward(im4)


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
	GradOutputs3=Module2:backward({State3,State4},GradOutputs[2])

	-- calculer les gradients pour les deux images
	Model:backward(im1,GradOutputs2[1])
	Model2:backward(im2,GradOutputs2[2])
	Model3:backward(im3,GradOutputs3[1])
	Model4:backward(im4,GradOutputs3[2])
	return loss, gradParameters
end

function doStuff_Rep(Model,criterion,gradParameters,im1,im2,im3,im4)

	Model2=Model:clone('weight','bias','gradWeight',
			'gradBias','running_mean','running_std')
	Model3=Model:clone('weight','bias','gradWeight',
			'gradBias','running_mean','running_std')
	Model4=Model:clone('weight','bias','gradWeight',
			'gradBias','running_mean','running_std')

	State1=Model:forward(im1)
	State2=Model2:forward(im2)
	State3=Model3:forward(im3)
	State4=Model4:forward(im4)

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

	-- BACKWARD

	GradOutputs=criterion:backward({delta1,delta2})
	GradS=criterion2:backward({State1,State3})

	GradOutputs2=Module:backward({State1,State2},GradOutputs[1])
	GradOutputs3=Module2:backward({State3,State4},GradOutputs[2])

	GradS1=GradOutputs2[1]*loss2+GradS[1]*loss
	GradS2=loss2*GradOutputs2[2]
	GradS3=GradOutputs3[1]*loss2+GradS[2]*loss
	GradS4=loss2*GradOutputs3[2]

	--print(GradS4)

	-- calculer les gradients pour les deux images
	Model:backward(im1, GradS1)
	Model2:backward(im2,GradS2)
	Model3:backward(im3,GradS3)
	Model4:backward(im4,GradS4)
	return loss2*loss, gradParameters
end
