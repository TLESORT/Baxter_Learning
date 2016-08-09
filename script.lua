

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

function copy_weight(model, AE)
	model:get(1).weight:copy(AE:get(1).weight)
	model:get(4).weight:copy(AE:get(5).weight)
	return model
end

function Rico_Training(Models, Mode,batch, criterion)
	local LR=0.01
	local mom=0.9
        local coefL2=0,0

	parameters,gradParameters = Models.Model1:getParameters()
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

	state=state or {learningRate = LR,paramVariance=nil, weightDecay=0.0005 }
	config=config or {}
	parameters, loss=optim.adagrad(feval, parameters,config, state)

	--parameters, loss=optim.sgd(feval, parameters, sgdState)
	return loss[1] -- table of one value transformed in value
end



--load the two images
function train_epoch(Models, list_folders_images, list_txt,use_simulate_images)
	local BatchSize=12
	nbEpoch=3
	local REP_criterion=get_Rep_criterion()
	local PROP_criterion=get_Prop_criterion()
	local CAUS_criterion=get_Caus_criterion()
	local TEMP_criterion=nn.MSDCriterion()

	local Temp_loss_list={}
	local Prop_loss_list={}
	local Rep_loss_list={}
	
	local NbBatch=1000
	local last_indice=#list_folders_images
	local imgs_test=load_list(images_Paths(list_folders_images[last_indice]))
	local truth=getTruth(list_txt[last_indice],use_simulate_images)
	show_figure(truth, './Log/The_Truth.Log')
	Print_performance(Models.Model1, imgs_test,"First_Test")

			
	for epoch=1, nbEpoch do
		print('--------------Epoch : '..epoch..' ---------------')
		print(#list_folders_images..' : sequences')
		nbList= #list_folders_images

		local NbBatchForLossEstimation=1000

		for l=1,nbList-1 do

			local Temp_loss=0
			local Prop_loss=0
			local Rep_loss=0

			local Temp_loss_tot=0
			local Prop_loss_tot=0
			local Rep_loss_tot=0

			list=images_Paths(list_folders_images[l])
			imgs=load_list(list)
			print(#imgs..' : images')
			truth=getTruth(list_txt[l],use_simulate_images)
			show_figure(truth, './Log/Truth_list_'..l..'.Log')


			for numBatch=1, NbBatch do
				Batch_Temp=getRandomBatch(imgs, list, list_txt[l], BatchSize, 	200, 200, "Temp", use_simulate_images)
				Batch_Prop=getRandomBatch(imgs, list, list_txt[l], BatchSize, 200, 200, "Prop", use_simulate_images)
				Temp_loss=Rico_Training(Models, 'Temp',Batch_Temp, TEMP_criterion)
				Prop_loss=Rico_Training(Models, 'Prop',Batch_Prop, PROP_criterion)
				Rep_loss=Rico_Training(Models,'Rep',Batch_Prop, REP_criterion)
				xlua.progress(numBatch, NbBatch)

				Temp_loss_tot=Temp_loss_tot+Temp_loss
				Prop_loss_tot=Prop_loss_tot+Prop_loss
				Rep_loss_tot=Rep_loss_tot+Rep_loss

			end
			save_model(Models.Model1,name_save)
			
			local id=name..l -- variable used to not mix several log files
			Print_performance(Models.Model1,imgs,id.."train")
			print(#imgs_test)
			Print_performance(Models.Model1, imgs_test,id.."_Test")
			table.insert(Temp_loss_list,Temp_loss_tot/NbBatch)
			table.insert(Prop_loss_list,Prop_loss_tot/NbBatch)
			table.insert(Rep_loss_list,Rep_loss_tot/NbBatch)

			Print_Loss(Temp_loss_list,Prop_loss_list,Rep_loss_list, id)
			xlua.progress(l, #list_folders_images)
		end
	end
end

name='Save08_08_NoTrick'
name_save='./Save/'..name..'.t7'
name_load='./Save/'..name..'.t7'

local use_simulate_images=true
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt(use_simulate_images)
local reload=false
local TakeWeightFromAE=false
local UseSecondGPU= true
local model_file='./models/topUniqueFM_Deeper'

local image_width=200
local image_height=200

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
Model2=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
Model3=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
Model4=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')

Models={Model1=Model,Model2=Model2,Model3=Model3,Model4=Model4}

train_epoch(Models, list_folders_images, list_txt,use_simulate_images)

