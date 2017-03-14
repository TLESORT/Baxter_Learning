function doStuff_temp(models,criterion,Batch,coef)
	
	local coef= coef or 1

	im1=Batch[1]:cuda()
	im2=Batch[2]:cuda()
	
	model=models.model1
	model2=models.model2

	State1=model:forward(im1)
	State2=model2:forward(im2)

	criterion=criterion:cuda()
	loss=criterion:forward({State2,State1})
	GradOutputs=criterion:backward({State2,State1})

	-- calculer les gradients pour les deux images
	model:backward(im1,coef*GradOutputs[2])
	model2:backward(im2,coef*GradOutputs[1])
	return loss, coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

function doStuff_Caus(models,criterion,Batch,coef)

	local coef= coef or 1
	im1=Batch[1]:cuda()
	im2=Batch[2]:cuda()
	
	model=models.model1
	model2=models.model2

	State1=model:forward(im1)
	State2=model2:forward(im2)

	criterion=criterion:cuda()
	output=criterion:updateOutput({State1, State2})
	--we backward with a starting gradient initialized at 1
	GradOutputs=criterion:updateGradInput({State1, State2}, torch.ones(1))

	-- calculer les gradients pour les deux images
	model:backward(im1,coef*GradOutputs[1]/Batch[1]:size(1))
	model2:backward(im2,coef*GradOutputs[2]/Batch[1]:size(1))
	return output:mean(), coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

function doStuff_Prop(models,criterion,Batch, coef)
	
	local coef= coef or 1

	im1=Batch[1]:cuda()
	im2=Batch[2]:cuda()
	im3=Batch[3]:cuda()
	im4=Batch[4]:cuda()

	model=models.model1
	model2=models.model2
	model3=models.model3
	model4=models.model4


	State1=model:forward(im1)
	State2=model2:forward(im2)
	State3=model3:forward(im3)
	State4=model4:forward(im4)


	criterion=criterion:cuda()
	output=criterion:updateOutput({State1, State2, State3, State4})

	--we backward with a starting gradient initialized at 1
	GradOutputs=criterion:updateGradInput({State1, State2, State3, State4},torch.ones(1))

	model:backward(im1,coef*GradOutputs[1]/Batch[1]:size(1))
	model2:backward(im2,coef*GradOutputs[2]/Batch[1]:size(1))
	model3:backward(im3,coef*GradOutputs[3]/Batch[1]:size(1))
	model4:backward(im4,coef*GradOutputs[4]/Batch[1]:size(1))

	return output:mean(), coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

function doStuff_Rep(models,criterion,Batch, coef)
	
	local coef= coef or 1

	im1=Batch[1]:cuda()
	im2=Batch[2]:cuda()
	im3=Batch[3]:cuda()
	im4=Batch[4]:cuda()


	model=models.model1
	model2=models.model2
	model3=models.model3
	model4=models.model4

	State1=model:forward(im1)
	State2=model2:forward(im2)
	State3=model3:forward(im3)
	State4=model4:forward(im4)

	criterion=criterion:cuda()
	output=criterion:updateOutput({State1, State2, State3, State4})

	--we backward with a starting gradient initialized at 1
	GradOutputs=criterion:updateGradInput({State1, State2, State3, State4}, torch.ones(1))


	model:backward(im1,coef*GradOutputs[1]/Batch[1]:size(1))
	model2:backward(im2,coef*GradOutputs[2]/Batch[1]:size(1))
	model3:backward(im3,coef*GradOutputs[3]/Batch[1]:size(1))
	model4:backward(im4,coef*GradOutputs[4]/Batch[1]:size(1))

	return output:mean(), coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

function get_Rep_criterion()
	h1 = nn.Identity()()
	h2 = nn.Identity()()
	h3 = nn.Identity()()
	h4 = nn.Identity()()

	h_h1 = nn.CSubTable()({h2,h1})
	h_h2 = nn.CSubTable()({h4,h3})

	madd = nn.CSubTable()({h_h2,h_h1})
	sqr=nn.Square()(madd)
	out1 = nn.Sum(1,1)(sqr)

	norm2= nn.Sum(1,1)(nn.Square()(nn.CSubTable()({h3,h1})))
	out2=nn.Exp()(nn.MulConstant(-1)(norm2))

	outTot=nn.Sum(1,1)(nn.CMulTable()({out1, out2}))
	gmod = nn.gModule({h1, h2, h3, h4}, {outTot})
	return gmod
end

function get_Prop_criterion()
	h1 = nn.Identity()()
	h2 = nn.Identity()()
	h3 = nn.Identity()()
	h4 = nn.Identity()()

	h_h1 = nn.CSubTable()({h2,h1})
	h_h2 = nn.CSubTable()({h4,h3})

	norm=nn.Sqrt()(nn.Sum(1,1)(nn.Square()(h_h1)))
	norm2=nn.Sqrt()(nn.Sum(1,1)(nn.Square()(h_h2)))

	madd = nn.CSubTable()({norm,norm2})
	sqr=nn.Square()(madd)
	out = nn.Sum(1,1)(sqr)

	gmod = nn.gModule({h1, h2, h3, h4}, {out})
	return gmod
end

function get_Caus_criterion()
	h1 = nn.Identity()()
	h2 = nn.Identity()()

	h_h1 = nn.CSubTable()({h2,h1})

	norm=nn.Sum(1,1)(nn.Square()(h_h1))
	exp=nn.Exp()(nn.MulConstant(-1)(norm))
	out = nn.Sum(1,1)(exp)

	gmod = nn.gModule({h1, h2}, {out})
	return gmod
end








