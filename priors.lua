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

	-- normalisation dégeulasse des gradients
--[[
	local somme_1=0
	local somme_2=0
	for i=1, GradOutputs[1]:size(1) do
	somme_1=somme_1+math.abs(GradOutputs[1][i][1])
	somme_2=somme_2+math.abs(GradOutputs[2][i][1])
	end
	for i=1, criterion.gradInput[1]:size(1) do
	if somme_1~=0 then GradOutputs[1][i]=GradOutputs[1][i]/somme_1 end	
	if somme_2~=0 then GradOutputs[2][i]=GradOutputs[2][i]/somme_2 end
	end
--]]



	-- calculer les gradients pour les deux images
	Model:backward(im1,GradOutputs[1])
	Model2:backward(im2,GradOutputs[2])

	--grad= (torch.ones(5,1)*0.5):cuda()
	--Model:backward(im1, -1*grad)
	--Model2:backward(im2,grad)

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

	-- normalisation dégeulasse des gradients
--[[
	local somme_1=0
	local somme_2=0
	local somme_3=0
	local somme_4=0

	for i=1, GradOutputs[1]:size(1) do
	somme_1=somme_1+math.abs(GradOutputs[1][i][1])
	somme_2=somme_2+math.abs(GradOutputs[2][i][1])
	somme_3=somme_3+math.abs(GradOutputs[3][i][1])
	somme_4=somme_4+math.abs(GradOutputs[4][i][1])
	end
	for i=1, criterion.gradInput[1]:size(1) do
	if somme_1~=0 then GradOutputs[1][i]=GradOutputs[1][i]/somme_1 end	
	if somme_2~=0 then GradOutputs[2][i]=GradOutputs[2][i]/somme_2 end
	if somme_3~=0 then GradOutputs[3][i]=GradOutputs[3][i]/somme_3 end
	if somme_4~=0 then GradOutputs[4][i]=GradOutputs[4][i]/somme_4 end
	end
--]]
	Model:backward(im1,GradOutputs[1])
	Model2:backward(im2,GradOutputs[2])
	Model3:backward(im3,GradOutputs[3])
	Model4:backward(im4,GradOutputs[4])
--[[	
	grad= (torch.ones(5,1)*0.25):cuda()
	Model:backward(im1, grad)
	Model2:backward(im2,-1*grad)
	Model3:backward(im3, grad)
	Model4:backward(im4,-1*grad)
]]--

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

	-- normalisation dégeulasse des gradients
--[[
	local somme_1=0
	local somme_2=0
	local somme_3=0
	local somme_4=0

	for i=1, GradOutputs[1]:size(1) do
	somme_1=somme_1+math.abs(GradOutputs[1][i][1])
	somme_2=somme_2+math.abs(GradOutputs[2][i][1])
	somme_3=somme_3+math.abs(GradOutputs[3][i][1])
	somme_4=somme_4+math.abs(GradOutputs[4][i][1])
	end
	for i=1, criterion.gradInput[1]:size(1) do
	if somme_1~=0 then GradOutputs[1][i]=GradOutputs[1][i]/somme_1 end	
	if somme_2~=0 then GradOutputs[2][i]=GradOutputs[2][i]/somme_2 end
	if somme_3~=0 then GradOutputs[3][i]=GradOutputs[3][i]/somme_3 end
	if somme_4~=0 then GradOutputs[4][i]=GradOutputs[4][i]/somme_4 end
	end
--]]
	Model:backward(im1,GradOutputs[1])
	Model2:backward(im2,GradOutputs[2])
	Model3:backward(im3,GradOutputs[3])
	Model4:backward(im4,GradOutputs[4])

--[[	grad= (torch.ones(5,1)*0.25):cuda()
	Model:backward(im1, grad)
	Model2:backward(im2,-1*grad)
	Model3:backward(im3, grad)
	Model4:backward(im4,-1*grad)--]]

	return output[1]
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






