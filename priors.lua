function doStuff_temp(Models,criterion,Batch)

	im1=Batch[1]:cuda()
	im2=Batch[2]:cuda()
	
	Model=Models.Model1
	Model2=Models.Model2

	State1=Model:forward(im1)
	State2=Model2:forward(im2)

	criterion=criterion:cuda()
	loss=criterion:forward({State1,State2})
	GradOutputs=criterion:backward({State1,State2})

	-- calculer les gradients pour les deux images
	Model:backward(im1,GradOutputs[1])
	Model2:backward(im2,GradOutputs[2])

	return loss
end

function doStuff_Caus(Models,criterion,Batch)

	im1=Batch[1]:cuda()
	im2=Batch[2]:cuda()
	
	Model=Models.Model1
	Model2=Models.Model2

	State1=Model:forward(im1)
	State2=Model2:forward(im2)


	criterion=criterion:cuda()
	output=criterion:updateOutput({State1, State2})

	--we backward with a starting gradient initialized at 1
	criterion:updateGradInput({State1, State2}, torch.ones(1))

	-- calculer les gradients pour les deux images
	Model:backward(im1,criterion.gradInput[1])
	Model2:backward(im2,criterion.gradInput[2])
	return output
end

function doStuff_Prop(Models,criterion,Batch)
	im1=Batch[1]:cuda()
	im2=Batch[2]:cuda()
	im3=Batch[3]:cuda()
	im4=Batch[4]:cuda()

	Model=Models.Model1
	Model2=Models.Model2
	Model3=Models.Model3
	Model4=Models.Model4


	State1=Model:forward(im1)
	State2=Model2:forward(im2)
	State3=Model3:forward(im3)
	State4=Model4:forward(im4)


	criterion=criterion:cuda()
	output=criterion:updateOutput({State1, State2, State3, State4})

	--we backward with a starting gradient initialized at 1
	GradOutputs=criterion:updateGradInput({State1, State2, State3, State4}, torch.ones(1))

	Model:backward(im1,GradOutputs[1])
	Model2:backward(im2,GradOutputs[2])
	Model3:backward(im3,GradOutputs[3])
	Model4:backward(im4,GradOutputs[4])

	return output[1]
end

function doStuff_Rep(Models,criterion,Batch)

	im1=Batch[1]:cuda()
	im2=Batch[2]:cuda()
	im3=Batch[3]:cuda()
	im4=Batch[4]:cuda()


	Model=Models.Model1
	Model2=Models.Model2
	Model3=Models.Model3
	Model4=Models.Model4

	State1=Model:forward(im1)
	State2=Model2:forward(im2)
	State3=Model3:forward(im3)
	State4=Model4:forward(im4)
-------------------------------------------------------------------------------------------

	criterion=criterion:cuda()
	output=criterion:updateOutput({State1, State2, State3, State4})

	--we backward with a starting gradient initialized at 1



	GradOutputs=criterion:updateGradInput({State1, State2, State3, State4}, torch.ones(1)) 

--!!!!!!!!!!!!!!!!!!!!!!!!!!! coef added -> grad *2
	local coef=2

	Model:backward(im1,coef*GradOutputs[1])
	Model2:backward(im2,coef*GradOutputs[2])
	Model3:backward(im3,coef*GradOutputs[3])
	Model4:backward(im4,coef*GradOutputs[4])

	return output[1]
end

function doStuff_Energie(Models,criterion,Batch)
	im=Batch[1]:cuda()

	Model=Models.Model1

	State=Model:forward(im)
	FM=Model:get(19)
	
	gmod = nn.gModule({h1}, {res})

	criterion=criterion:cuda()
	output=criterion:updateOutput({State})
	GradOutputs=criterion:updateGradInput({State}, torch.ones(1))

	-- this is a test:
	-- the idea is to put a gradient on the top feature map before the MLP
	-- then backwarding a nul gradient from the top the model
	-- the weight wont change util reaching the top feature map
	-- then the accumulation of gradient will take the gradient of the feature map and backarding through the network
	FM:updateGradInput(Model:get(18).output,GradOutputs)
	Model:backward(im1,State*0)

	return ouput[1]
end

function fake_energie_criterion()
	
	h1=nn.Identity()()
	view=nn.View(100)(h1)
	mean=nn.Mean()(view)
	diff=nn.AddConstant(nn.MulConstant(-1)(mean.output[1]))(view)
	
	sqrt=nn.Square()(diff)

	std=nn.Mean()(sqrt)
	diff2=nn.AddConstant(nn.MulConstant(-1)(std.output[1]))(sqrt)
	sqrt2=nn.Square()(diff2)
	res=nn.Mean()(sqrt2)

	gmod = nn.gModule({h1}, {res})
	return gmod
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








