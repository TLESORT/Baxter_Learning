

require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'cunn'
require 'nngraph'

require 'MSDC'
require 'functions.lua'
require "Get_HeadCamera_HeadMvt"
require 'priors'

function Rico_Training(Models, Mode,batch, criterion)
	local LR=0.001
	local mom=0.9
        local coefL2=0,0

	      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
		 -- just in case:
		 collectgarbage()

		 -- get new parameters
		 if x ~= parameters then
		    parameters:copy(x)
		 end

		 -- reset gradients
		gradParameters:zero()
	--	loss=doStuff_Energie(Models,fake_energie_criterion(),batch)
		if Mode=='Simpl' then print("Simpl")
		elseif Mode=='Temp' then loss=doStuff_temp(Models,criterion, batch)
		elseif Mode=='Prop' then loss=doStuff_Prop(Models,criterion,batch)	
		elseif Mode=='Caus' then loss=doStuff_Caus(Models,criterion,batch)
		elseif Mode=='Rep' then loss=doStuff_Rep(Models,criterion,batch)
		else print("Wrong Mode")
		end
         	return loss,gradParameters
	end
	-- met à jour les parmetres avec les 2 gradients
	-- Perform SGD step:
        sgdState = sgdState or { learningRate = LR, momentum = mom,learningRateDecay = 5e-7,weightDecay=coefL2 }

	--state=state or {learningRate = LR,paramVariance=nil, weightDecay=0.0005 }
	--config=config or {}
	--parameters, loss=optim.adagrad(feval, parameters,config, state)

	parameters, loss=optim.sgd(feval, parameters, sgdState)
	return loss[1] -- table of one value transformed in value
end


function train_Epoch(Models, list_folders_images, list_txt,use_simulate_images)
	local BatchSize=12
	nbEpoch=50
	local NbBatch=100

	local REP_criterion=get_Rep_criterion()
	local PROP_criterion=get_Prop_criterion()
	local CAUS_criterion=get_Caus_criterion()
	local TEMP_criterion=nn.MSDCriterion()

	local Temp_loss_list={}
	local Prop_loss_list={}
	local Rep_loss_list={}
	local Caus_loss_list={}

	local Temp_loss_list_test={}
	local Prop_loss_list_test={}
	local Rep_loss_list_test={}
	local Caus_loss_list_test={}

	
	nbList= #list_folders_images
	local list_truth=images_Paths(list_folders_images[nbList])

	imgs_test=load_list(list_truth)
	txt_test=list_txt[nbList]

	local arrondit=false
	local truth=getTruth(txt_test,use_simulate_images,arrondit)
	show_figure(truth, Log_Folder..'The_Truth.Log')
	Print_performance(Models, imgs_test,txt_test,"First_Test",Log_Folder,use_simulate_images)

	real_temp_loss,real_prop_loss,real_rep_loss=real_loss(txt_test,use_simulate_images)
	print("temp loss : "..real_temp_loss)
	print("prop loss : "..real_prop_loss[1])
	print("rep loss : "..real_rep_loss[1])
	imgs={}
	for i=1, nbList-1 do
		list=images_Paths(list_folders_images[i])
		table.insert(imgs,load_list(list,200,200))
	end

			
	for epoch=1, nbEpoch do
		print('--------------Epoch : '..epoch..' ---------------')
		print(nbList..' : sequences')

		local Temp_loss=0
		local Prop_loss=0
		local Rep_loss=0
		local Caus_loss=0

		for numBatch=1, NbBatch do

			indice1=torch.random(1,nbList-1)
			repeat indice2=torch.random(1,nbList-1) until (indice1 ~= indice2)

			txt1=list_txt[indice1]
			txt2=list_txt[indice2]

			imgs1=imgs[indice1]
			imgs2=imgs[indice2]

			Batch_Temp=getRandomBatchFromSeparateList(imgs1, imgs2, txt1,txt2, BatchSize, image_width, image_height, "Temp", use_simulate_images)
			Batch_Prop=getRandomBatchFromSeparateList(imgs1, imgs2, txt1,txt2, BatchSize, image_width, image_height, "Prop", use_simulate_images)
			Batch_Caus=getRandomBatchFromSeparateList(imgs1, imgs2, txt1,txt2, BatchSize, image_width, image_height, "Caus", use_simulate_images)

			Temp_loss=Temp_loss+Rico_Training(Models, 'Temp',Batch_Temp, TEMP_criterion)
			Prop_loss=Prop_loss+Rico_Training(Models, 'Prop',Batch_Prop, PROP_criterion)
			Rep_loss=Rep_loss+Rico_Training(Models,'Rep',Batch_Prop, REP_criterion)
			Caus_loss=Caus_loss+Rico_Training(Models, 'Caus',Batch_Caus, CAUS_criterion)
			xlua.progress(numBatch, NbBatch)
		end
		save_model(Models.Model1,name_save)
		
		local id=name..epoch -- variable used to not mix several log files
		Temp_test,Prop_test,Rep_test=Print_performance(Models, 		imgs_test,txt_test,id.."_Test",Log_Folder,use_simulate_images)

		table.insert(Temp_loss_list,Temp_loss/NbBatch)
		table.insert(Prop_loss_list,Prop_loss/NbBatch)
		table.insert(Rep_loss_list,Rep_loss/NbBatch)		
		table.insert(Caus_loss_list,Caus_loss/NbBatch)

		table.insert(Temp_loss_list_test,Temp_test)
		table.insert(Prop_loss_list_test,Prop_test)
		table.insert(Rep_loss_list_test,Rep_test)
		table.insert(Caus_loss_list_test,Caus_test)

		Print_Loss(Temp_loss_list,Prop_loss_list,Rep_loss_list,
			Temp_loss_list_test,Prop_loss_list_test,Rep_loss_list_test,
			Log_Folder)
	end
end

name='Save29_08_4'
name_save='./Save/'..name..'.t7'
name_load='./Save/'..name..'.t7'

Log_Folder='./Log/29_08/Everything/confirmation/'

local use_simulate_images=true
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt(use_simulate_images)
local reload=false
local TakeWeightFromAE=false
local UseSecondGPU= false
local model_file='./models/topUniqueFM_Deeper'

torch.manualSeed(123)

image_width=200
image_height=200

if UseSecondGPU then
	cutorch.setDevice(2) 
end

if reload then
	Model = torch.load(name_load):double()
elseif TakeWeightFromAE then
	require './Autoencoder/noiseModule'
	require(model_file)
	Model=getModel(image_width,image_height)
	AE= torch.load('./Save/AE_3x3_1TopFM.t7'):double()
	print('AE\n' .. AE:__tostring());
	Model=copy_weight(Model, AE)
else
	require(model_file)
	Model=getModel(image_width,image_height)	
end
Model=Model:cuda()
parameters,gradParameters = Model:getParameters()
Model2=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
Model3=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
Model4=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')

Models={Model1=Model,Model2=Model2,Model3=Model3,Model4=Model4}

train_Epoch(Models, list_folders_images, list_txt,use_simulate_images)

