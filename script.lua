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
require 'printing.lua'
require "Get_HeadCamera_HeadMvt"
require 'priors'

function Rico_Training(Models, Mode,batch, criterion, coef, LR)
	local LR=LR or 0.001
	local mom=0.0--9
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
		if Mode=='Simpl' then print("Simpl")
		elseif Mode=='Temp' then loss,grad=doStuff_temp(Models,criterion, batch,coef)
		elseif Mode=='Prop' then loss,grad=doStuff_Prop(Models,criterion,batch,coef)
		elseif Mode=='Caus' then loss,grad=doStuff_Caus(Models,criterion,batch,coef)
		elseif Mode=='Rep' then loss,grad=doStuff_Rep(Models,criterion,batch,coef)
		else print("Wrong Mode")
		end
         	return loss,gradParameters
	end
	-- met à jour les parmetres avec les 2 gradients
	-- Perform SGD step:
--        sgdState = sgdState or { learningRate = LR, momentum = mom,learningRateDecay = 5e-7,weightDecay=coefL2 }
--	parameters, loss=optim.sgd(feval, parameters, sgdState)

optimState={learningRate=LR}
parameters, loss=optim.adagrad(feval, parameters, optimState)

	 -- loss[1] table of one value transformed in just a value
	 -- grad[1] we use just the first gradient to print the figure (there are 2 or 4 gradient normally)
	return loss[1], grad:mean()
end


function train_Epoch(Models,list_folders_images,list_txt,Prior_Used,Log_Folder,use_simulate_images,LR)
	local BatchSize=12
	local nbEpoch=200 --15 is ok too
	local NbBatch=10
	
	local name='Save'..day
	local name_save=Log_Folder..name..'.t7'

	local REP_criterion=get_Rep_criterion()
	local PROP_criterion=get_Prop_criterion()
	local CAUS_criterion=get_Caus_criterion()
	local TEMP_criterion=nn.MSDCriterion()

	local Temp_loss_list, Prop_loss_list, Rep_loss_list, Caus_loss_list = {},{},{},{}
	local Temp_loss_list_test,Prop_loss_list_test,Rep_loss_list_test,Caus_loss_list_test = {},{},{},{}
	local Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list = {},{},{},{}		
	local list_errors, List_corr, sum_loss_train, sum_loss_test={}, {}, {}, {}

	local Prop=Have_Todo(Prior_Used,'Prop')
	local Temp=Have_Todo(Prior_Used,'Temp')
	local Rep=Have_Todo(Prior_Used,'Rep')
	local Caus=Have_Todo(Prior_Used,'Caus')
print(Prop)
print(Temp)
print(Rep)
print(Caus)

	local coef_Temp=1
	local coef_Prop=1
	local coef_Rep=1
	local coef_Caus=1
	local coef_list={coef_Temp,coef_Prop,coef_Rep,coef_Caus}

	local best=0
	local list_truth=images_Paths(list_folders_images[nbList])

	imgs_test=load_list(list_truth,image_width,image_height,not WODA) --DA added to validation test
	txt_test=list_txt[nbList]

	local truth=getTruth(txt_test,use_simulate_images)
	show_figure(truth, Log_Folder..'The_Truth.log')
	Temp_test,Prop_test,Rep_test,Caus_test, corr=Print_performance(Models, imgs_test,txt_test,"First_Test",Log_Folder,use_simulate_images,truth)
	table.insert(List_corr,corr)

	real_temp_loss,real_prop_loss,real_rep_loss, real_caus_loss=real_loss(txt_test,use_simulate_images)
	print("temp loss : "..real_temp_loss)
	print("prop loss : "..real_prop_loss[1])
	print("rep loss : "..real_rep_loss[1])	
	print("caus loss : "..real_caus_loss[1])


	printParamInAFile(Log_Folder,coef_list, LR, "adagrad", BatchSize, nbEpoch, NbBatch)
			
	for epoch=1, nbEpoch do
		print('--------------Epoch : '..epoch..' ---------------')
		print(nbList..' : sequences')

		local Temp_loss=0
		local Prop_loss=0
		local Rep_loss=0
		local Caus_loss=0

		local Grad_Temp=0
		local Grad_Prop=0
		local Grad_Rep=0
		local Grad_Caus=0


		if not WODA and torch.random(1,30)==30 then --time to time we reload every images to change the dataAugmentation
			imgs={}
			for i=1, nbList-1 do
				list=images_Paths(list_folders_images[i])
				table.insert(imgs,load_list(list,image_width,image_height,true))
			end
		end

		for numBatch=1, NbBatch do

			indice1=torch.random(1,nbList-1)
			repeat indice2=torch.random(1,nbList-1) until (indice1 ~= indice2)

			txt1=list_txt[indice1]
			txt2=list_txt[indice2]

			imgs1=imgs[indice1]
			imgs2=imgs[indice2]
			
			if Temp then
				Batch_Temp=getRandomBatchFromSeparateList(imgs1,imgs2,txt1,txt2,BatchSize,image_width,image_height,"Temp", use_simulate_images)
				Loss,Grad=Rico_Training(Models,'Temp',Batch_Temp, TEMP_criterion, coef_Temp,LR)
				Grad_Temp=Grad_Temp+Grad
 				Temp_loss=Temp_loss+Loss
			end
			if Prop then 
				Batch_Prop=getRandomBatchFromSeparateList(imgs1,imgs2,txt1,txt2,BatchSize,image_width,image_height,"Prop", use_simulate_images)
				Loss,Grad=Rico_Training(Models, 'Prop',Batch_Prop, PROP_criterion, coef_Prop,LR)
				Grad_Prop=Grad_Prop+Grad
				Prop_loss=Prop_loss+Loss
			end
			if Rep then 
				Batch_Rep=getRandomBatchFromSeparateList(imgs1,imgs2,txt1,txt2,BatchSize,image_width,image_height,"Rep", use_simulate_images)
				Loss,Grad=Rico_Training(Models,'Rep',Batch_Rep, REP_criterion, coef_Rep,LR)
				Grad_Rep=Grad_Rep+Grad
				Rep_loss=Rep_loss+Loss
			end
			if Caus then 
				Batch_Caus=getRandomBatchFromSeparateList(imgs1,imgs2,txt1,txt2,BatchSize,image_width,image_height,"Caus", use_simulate_images)
				Loss,Grad=Rico_Training(Models, 'Caus',Batch_Caus, CAUS_criterion, coef_Caus,LR)
				Grad_Caus=Grad_Caus+Grad
				Caus_loss=Caus_loss+Loss
			end
			xlua.progress(numBatch, NbBatch)
		end
		
		local id=name..epoch -- variable used to not mix several log files
		Temp_test,Prop_test,Rep_test,Caus_test, corr=Print_performance(Models, imgs_test,txt_test,id.."_Test",Log_Folder,use_simulate_images,truth)

		table.insert(Temp_loss_list,Temp_loss/(NbBatch*BatchSize))
		table.insert(Prop_loss_list,Prop_loss/(NbBatch*BatchSize))
		table.insert(Rep_loss_list,Rep_loss/(NbBatch*BatchSize))		
		table.insert(Caus_loss_list,Caus_loss/(NbBatch*BatchSize))

		table.insert(Temp_loss_list_test,Temp_test)
		table.insert(Prop_loss_list_test,Prop_test)
		table.insert(Rep_loss_list_test,Rep_test)
		table.insert(Caus_loss_list_test,Caus_test)


		table.insert(sum_loss_train,(Temp_loss+Prop_loss+Rep_loss+Caus_loss)/(NbBatch*BatchSize))
		table.insert(sum_loss_test,Temp_test+Prop_test+Rep_test+Caus_test)

		table.insert(Temp_grad_list,Grad_Temp/NbBatch)
		table.insert(Prop_grad_list,Grad_Prop/NbBatch)
		table.insert(Rep_grad_list,Grad_Rep/NbBatch)
		table.insert(Caus_grad_list,Grad_Caus/NbBatch)
		table.insert(List_corr,corr)

		Print_Loss(Temp_loss_list,Prop_loss_list,Rep_loss_list,Caus_loss_list,
		Temp_loss_list_test,Prop_loss_list_test,Rep_loss_list_test,Caus_loss_list_test,
			Log_Folder)

show_loss(sum_loss_train, sum_loss_test, Log_Folder..'Sum_loss.log')
		Print_Grad(Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list,Log_Folder)
		
		show_figure(List_corr, Log_Folder..'Correlation.log', 'corr', '-')
		save_model(Models.Model1,name_save)
		if corr^2>best then
			best=corr^2
		 	save_model(Models.Model1,Log_Folder..name..'_best.t7')
		end
	end

end


day="3_11_reload_211woda"
local UseSecondGPU= true
local LR=0.00005
WODA=false

Tests_Todo={{"Prop","Temp","Caus","Rep"}}
--[[{"Prop","Temp","Caus","Rep"}},
{"Rep","Caus"},
{"Prop","Caus"},
{"Temp","Caus"},
{"Temp","Prop"},
{"Rep","Prop"},
{"Rep","Temp"},
{"Rep"},
{"Temp"},
{"Caus"},
{"Prop"},
{"Rep","Caus","Prop"},
{"Rep","Caus","Temp"},
{"Rep","Prop","Temp"},
{"Prop","Caus","Temp"},
}--]]

local Log_Folder='./Log/'..day..'/'


name_load='./Log/Save/'..day..'.t7'

local use_simulate_images=true
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt(use_simulate_images)
local reload=true
local TakeWeightFromAE=false
local model_file='./models/topUniqueFM_Deeper'--'./models/OneLayer' './models/topUniqueFM_Deeper'


image_width=200
image_height=200

if UseSecondGPU then
	cutorch.setDevice(2) 
end


torch.manualSeed(123)
nbList= #list_folders_images
imgs={}
for i=1, nbList-1 do
	list=images_Paths(list_folders_images[i])
	table.insert(imgs,load_list(list,image_width,image_height,not WODA))
end

--image.display(imgs[1][1])
for nb_test=1, #Tests_Todo do
	if reload then
		Model = torch.load('./Log/2_11_WODA/Everything/Save2_11_WODA.t7'):double()
	elseif TakeWeightFromAE then
		require './Autoencoder/noiseModule'
		require(model_file)
		Model=getModel(image_width,image_height)
		AE= torch.load('./Log/13_09_adagrad4_coef1/Everything/Save13_09_adagrad4_coef1.t7'):double()
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



	local Priors=Tests_Todo[nb_test]
	local Log_Folder=Get_Folder_Name(Log_Folder,Priors)
	print("Test actuel : "..Log_Folder)
	train_Epoch(Models,list_folders_images,list_txt,Priors,Log_Folder,use_simulate_images,LR)
end

imgs={} --memory is free!!!!!
