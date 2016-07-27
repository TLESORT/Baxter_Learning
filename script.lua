

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
require './Autoencoder/noiseModule'


function copy_weight(model, AE)
	model:get(1).weight:copy(AE:get(1).weight)
	model:get(4).weight:copy(AE:get(5).weight)
	model:get(7).weight:copy(AE:get(8).weight)
	return model
end

function Rico_Training(Models, Mode,batch, criterion)
	local LR=0.01
	local mom=0.0
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
	optim.adagrad(feval, parameters,config, state)


	--parameters, loss=optim.sgd(feval, parameters, sgdState)
end



--load the two images
function train_epoch(Models, list_folders_images, list_txt)
	local BatchSize=5
	local list_t=images_Paths(list_folders_images[2])
	nbEpoch=10
	local REP_criterion=get_Rep_criterion()
	local PROP_criterion=get_Prop_criterion()
	local CAUS_criterion=get_Caus_criterion()
	local TEMP_criterion=nn.MSDCriterion()
			
	for epoch=1, nbEpoch do
		print('--------------Epoch : '..epoch..' ---------------')
		print(#list_folders_images..' : sequences')
		nbList= #list_folders_images
		
		nbList=1-------------------------------!!!!----------------
		
		for l=1,nbList do
			list=images_Paths(list_folders_images[l])
			imgs=load_list(list)
			list_Prop, list_Temp=create_Head_Training_list(list, list_txt[1])
			NbPass=#list_Prop.Mode+#list_Temp.Mode
			NbBatch=math.floor((#list_Prop.Mode+#list_Temp.Mode)/BatchSize)
			for numBatch=1, NbBatch do
				Batch_Temp=getBatch(imgs,list_Temp, numBatch, BatchSize, 200, 200,"Temp")
				Batch_Prop=getBatch(imgs,list_Prop, numBatch, BatchSize, 200, 200,"Prop")
				Rico_Training(Models, 'Temp',Batch, TEMP_criterion)
				Rico_Training(Models, 'Prop',Batch_Prop, PROP_criterion)
				Rico_Training(Models,'Rep',Batch_Prop, REP_criterion)
				xlua.progress(numBatch, NbBatch)
			end
			xlua.progress(l, #list_folders_images)
		end
		save_model(Models.Model1,'./Save/Save07_07.t7')
		Print_performance(Models.Model1,imgs,epoch)
	end
end

local list_folders_images, list_txt=Get_HeadCamera_HeadMvt()
local reload=false
local TakeWeightFromAE=true

local image_width=200
local image_height=200

if reload then
	Model = torch.load('./Save/Save07_07.t7'):double()
elseif TakeWeightFromAE then
	require "./models/mini_model"
	Model=getModel(image_width,image_height)
	AE= torch.load('./Save/AE_3x3.t7'):double()
	print('AE\n' .. AE:__tostring());
	Model=copy_weight(Model, AE)
else
	require "./models/mini_model"
	Model=getModel(image_width,image_height)	
end
Model=Model:cuda()
Model2=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
Model3=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
Model4=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')

Models={Model1=Model,Model2=Model2,Model3=Model3,Model4=Model4}

train_epoch(Models, list_folders_images, list_txt)

