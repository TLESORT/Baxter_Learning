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
require "Get_Baxter_Files"
require 'priors'

REP_criterion=get_Rep_criterion()
PROP_criterion=get_Prop_criterion()
CAUS_criterion=get_Caus_criterion()
TEMP_criterion=nn.MSDCriterion()

image_width=200
image_height=200

function Rico_Training(Models, Mode,batch, coef, LR)
	local LR=LR or 0.001
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
		if Mode=='Temp' then loss,grad=doStuff_temp(Models,TEMP_criterion, batch,coef)
		elseif Mode=='Prop' then loss,grad=doStuff_Prop(Models,PROP_criterion,batch,coef)
		elseif Mode=='Caus' then loss,grad=doStuff_Caus(Models,CAUS_criterion,batch,coef)
		elseif Mode=='Rep' then loss,grad=doStuff_Rep(Models,REP_criterion,batch,coef)
		else print("Wrong Mode")
		end
         	return loss,gradParameters
	end
	optimState={learningRate=LR}
	parameters, loss=optim.adagrad(feval, parameters, optimState)
end


function train_Epoch(Models,list_folders_images,list_txt,Log_Folder,use_simulate_images,LR)
	local BatchSize=12
	local nbEpoch=100
	local NbBatch=10
	local name_save=Log_Folder..'Save.t7'
	local coef_Temp=1
	local coef_Prop=1
	local coef_Rep=1
	local coef_Caus=1
	local coef_list={coef_Temp,coef_Prop,coef_Rep,coef_Caus}
	local list_corr={}
	Model=getModel()

	nbList= #list_folders_images
	imgs={}
	for i=1, nbList do
		list=images_Paths(list_folders_images[i])
		table.insert(imgs,load_list(list,image_width,image_height,dataAugmentation))
	end
	
	for CrossValStep=1, nbList do
		------------------------------------------------------------*
		torch.manualSeed(123)
		Model=Model:cuda()
		parameters,gradParameters = Model:getParameters()
		Model2=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
		Model3=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
		Model4=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
		Models={Model1=Model,Model2=Model2,Model3=Model3,Model4=Model4}
		------------------------------------------------------------*
		Log_Folder=Log_Folder..'CrossVal'..CrossValStep..'/' --*
		-- we put the test set at the end
		imgs[CrossValStep],imgs[nbList]=imgs[nbList],imgs[CrossValStep]--*


		-- we use last list as test 
		--local list_truth=images_Paths(list_folders_images[nbList])
		local list_truth=images_Paths(list_folders_images[CrossValStep])--*
		local imgs_test=load_list(list_truth,image_width,image_height,false)
		--local txt_test=list_txt[nbList]
		local txt_test=list_txt[CrossValStep]--*

		local truth=getTruth(txt_test,use_simulate_images)
		show_figure(truth,Log_Folder..'GroundTruth.log')
		corr=Print_performance(Models, imgs_test,txt_test,"First_Test",Log_Folder,truth)
		table.insert(list_corr,corr)

		for epoch=1, nbEpoch do
			print('--------------Epoch : '..epoch..' ---------------')
			print(nbList..' : sequences')


			for numBatch=1, NbBatch do

				indice1=torch.random(1,nbList-1)
				repeat indice2=torch.random(1,nbList-1) until (indice1 ~= indice2)

				txt1=list_txt[indice1]
				txt2=list_txt[indice2]

				imgs1=imgs[indice1]
				imgs2=imgs[indice2]


				Batch_Temp=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Temp")
				Batch_Prop=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Prop")
				Batch_Rep=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Rep")
				Batch_Caus=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Caus")
				Rico_Training(Models,'Temp',Batch_Temp, coef_Temp,LR)
				Rico_Training(Models, 'Prop',Batch_Prop, coef_Prop,LR)
				Rico_Training(Models,'Rep',Batch_Rep, coef_Rep,LR)
				Rico_Training(Models, 'Caus',Batch_Caus, coef_Caus,LR)

				xlua.progress(numBatch, NbBatch)
			end
			corr=Print_performance(Models, imgs_test,txt_test,"Test",Log_Folder,truth)
			table.insert(list_corr,corr)

			save_model(Models.Model1,name_save)
		end
		show_figure(list_corr,Log_Folder..'correlation.log','-')
		-- the list is put in the original order
		imgs[CrossValStep],imgs[nbList]=imgs[nbList],imgs[CrossValStep]--*
	end --*
end


local LR=0.0001
local dataAugmentation=false
local Log_Folder='./Log/'
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt()

require('./models/convolutionnal')
Model=getModel()



torch.manualSeed(123)
	
Model=Model:cuda()
parameters,gradParameters = Model:getParameters()
Model2=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
Model3=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
Model4=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')

Models={Model1=Model,Model2=Model2,Model3=Model3,Model4=Model4}




train_Epoch(Models,list_folders_images,list_txt,Log_Folder,use_simulate_images,LR)


imgs={} --memory is free!!!!!
