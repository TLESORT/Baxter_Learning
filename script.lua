

require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'cunn'

require 'MSDC'
require "mini_model"
require "Get_HeadCamera_HeadMvt"

function Temp_Loss(Model,criterion,image1, image2)
	local LR=0.1

	local Data1={data=torch.Tensor(1, 3, 200, 200),size = function() return 1 end}
	local Data2={data=torch.Tensor(1, 3, 200, 200),size = function() return 1 end}
	
	Data1.data[{1}]=image1:cuda()
	Data2.data=image2:cuda()
	print(Data1.data:size())
	State1=Model:forward({Data1.data})
	Model2=Model:clone('weight','bias','gradWeight',
			'gradBias','weight','bias','running_mean','running_std')

	State2=Model2:forward(Data2.data)
	--Loss_Rico= computeGradient(State1,State2, Mode)
	print(" 1 : "..Model.output[1])
	print(" 2 : "..Model2.output[1])

	print(Model:get(9).weight)

	loss=criterion:forward({State1,State2})
	GradOutputs=criterion:backward({State1,State2})
	print(GradOutputs)

	-- calculer les gradients pour les deux images
	Model:backward(Data1.data,GradOutputs[1])
	Model2:backward(Data2.data,GradOutputs[2])
	
	-- met à jour les parmetres avec les 2 gradients
	Model:updateParameters(LR)
	
end

--TODO
function getImages(list, indice)
	local image1=image.load(list[indice],3,'byte')
	local image2=image.load(list[indice+1],3,'byte')
	local img1_rsz=image.scale(image1,"200x200")
	local img2_rsz=image.scale(image2,"200x200")
	
	local Mode="Temp"

	return img1_rsz, img2_rsz, Mode
end

--TODO
function Rico(model,criterion,Mode,image1, image2, image3, image4)
--Mode : Simplicity, Temporal Coherence, proportionnality, Causality, Repeatability
	local Mode= Mode or 'Prop'
	if Mode=='Simpl' then print("Simpl")
	elseif Mode=="Temp" then 
		Temp_Loss(model,criterion,image1, image2)
	elseif Mode=="Prop" then print("Prop")
	elseif Mode=="Caus" then print("Caus")
	elseif Mode=="Rep" then print("Rep")	
	else print("Wrong Mode")
	end
end

--load the two images
function train_epoch(Model)
	local criterion=nn.MSDCriterion()	
	criterion=criterion:cuda()
	local list1=images_Paths(list_folders_images[1])
	local max=1
	for i=1, #list1-1 do
		image1, image2, Mode=getImages(list1,i)
		Rico(Model,criterion,"Temp",image1, image2)
	end
end

list_folders_images=Get_HeadCamera_HeadMvt()

local image_width=200
local image_height=200
local Model=getNet(image_width,image_height)
Model=Model:cuda()
train_epoch(Model)

