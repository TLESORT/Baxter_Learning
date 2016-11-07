

require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'cunn'

require 'MSDC'
require '../functions.lua'
require "../Get_HeadCamera_HeadMvt"
require "../printing"

function Print_performance_sup(Model,imgs,txt, name, Log_Folder, use_simulate_images,truth)

	local list_out1={}

	for i=1, #imgs do
		image1=imgs[i]
		Data1=torch.Tensor(1,3,200,200)
		Data1[1]=image1
		
		Model:forward(Data1:cuda())
		local State1=Model.output[1]	

		table.insert(list_out1,State1)
	end

	Variable_Name="out"
	local scale= 1000
	accLogger = optim.Logger(Log_Folder..'state'..name..'.log')
	for i=1, #list_out1 do
	-- update logger
		accLogger:add{[Variable_Name] = list_out1[i]*scale,
				["Truth"] = truth[i]*scale,}
	end
	-- plot logger
	accLogger:style{[Variable_Name] = '-',["Truth"]='-'}
	accLogger.showPlot = false
	accLogger:plot()
end

function supervised_Training(Model,image1, label)
	local LR=0.001
	local mom=0
        local coefL2=0
	criterion=nn.MSDCriterion()
	
	res=0
	criterion=criterion:cuda()
	
	im1=image1:cuda()

	parameters,gradParameters = Model:getParameters()
	 local feval = function(x)
		 -- just in case:
		 collectgarbage()

		 -- get new parameters
		 if x ~= parameters then
		    parameters:copy(x)
		 end

		 -- reset gradients
		gradParameters:zero()
		output=Model:forward(im1)
		if arrondit(output[1])==arrondit(label[1]) then res=1 end
		loss2=criterion:forward({output, label})
		Model:zeroGradParameters()
		grad_crit=criterion:backward({output, label})
		Model:backward(im1, grad_crit[1])
         return loss2,gradParameters
	end
	--Model:updateParameters(LR)
	sgdState = sgdState or { learningRate = LR, momentum = mom,learningRateDecay = 5e-7,weightDecay=coefL2 }
	parameters, loss=optim.sgd(feval, parameters, sgdState)
	return res, loss2
end

function Dumb_Batch(imgs, label)
	Batch=torch.Tensor(1,3, 200, 200)
	Label=torch.Tensor(1)
	Batch[1]=imgs	
	Label[1]=label

	return Batch:cuda(), Label:cuda()
end

--load the two images
function train_epoch(Model,list_folders_images,list_txt,Priors,Log_Folder,use_simulate_images,LR)

	local last_indice=#list_folders_images
	local list_test=images_Paths(list_folders_images[last_indice])
	local name='Save'..day
	local name_save=Log_Folder..name..'.t7'

	imgs_test=load_list(list_test)
	local truth_test=getTruth(list_txt[last_indice],use_simulate_images)
	show_figure(truth_test, './Log/The_Truth.Log', 1000/0.8)
	imgs={}
	for i=1, #list_folders_images-1 do
		list=images_Paths(list_folders_images[i])
		table.insert(imgs,load_list(list,200,200))
	end
	Print_performance_sup(Model, imgs_test,txt_test,"First_Test",Log_Folder,use_simulate_images,truth_test)

	nbEpoch=10
	local NbPass=100
	for epoch=1, nbEpoch do
		
		local stat=0
		local lossToT=0
		local NbPassToT=0
		print('--------------Epoch : '..epoch..' ---------------')
		print(#list_folders_images..' : sequences')
		nbList= #list_folders_images

		for l=1,NbPass do

			indice1=math.random(1,nbList-1)
			txt1=list_txt[indice1]
			imgs1=imgs[indice1]
			truth=getTruth(txt1,use_simulate_images,true)
			

			for i=1, #truth do
				Batch, Label=Dumb_Batch(imgs1[i], truth[i])
				--Label=Label/0.8
				res, loss=supervised_Training(Model,Batch, Label)
				lossToT=lossToT+loss2
				stat=stat+res
			end
			NbPassToT=NbPassToT+ #truth 
			xlua.progress(l, NbPass)
		end
		print("precision : "..stat/NbPassToT)
		print("loss : "..lossToT/NbPassToT)
		print("Nb images : "..NbPassToT)
		local id=name..epoch
		save_model(Model,name_save)
		Print_performance_sup(Model, imgs_test,txt_test,id.."_Test",Log_Folder,use_simulate_images,truth_test)
	end
end

day="15_09_supervised"
local UseSecondGPU= false
local LR=0.002

local use_simulate_images=true
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt(use_simulate_images)
local reload=false
local UseSecondGPU= true
local model_file='../models/topUniqueFM_Deeper'
local Log_Folder='./Log/'..day..'/'
image_width=200
image_height=200

if UseSecondGPU then
	cutorch.setDevice(2) 
end

nbList= #list_folders_images
imgs={}
for i=1, nbList-1 do
	list=images_Paths(list_folders_images[i])
	table.insert(imgs,load_list(list,image_width,image_height,true))
end

if reload then
	Model = torch.load(name_load):double()
else
	require(model_file)
	Model=getModel(image_width,image_height)	
end
Model=Model:cuda()
Tests_Todo={{"Prop","Temp","Caus","Rep"}}
local Priors=Tests_Todo[1]
local Log_Folder=Get_Folder_Name(Log_Folder,Priors)
train_epoch(Model,list_folders_images,list_txt,Priors,Log_Folder,use_simulate_images,LR)

