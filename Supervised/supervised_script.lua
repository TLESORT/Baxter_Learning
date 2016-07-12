

require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'cunn'

require 'MSDC'
require 'functions.lua'
require "Get_HeadCamera_HeadMvt"

function supervised_Training(Model,image1, joint)
	local LR=0.005
	local mom=0
        local coefL2=0.5
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
		if arrondit(output[1])==joint then res=1 end
		print(arrondit(output[1]).." vs "..joint)
		label=torch.Tensor(1)
		--label=label:copy(output):cuda()
		label[1]=joint
		label=label:cuda()
		loss2=criterion:forward({output, label})
		--print(loss)
		Model:zeroGradParameters()
		grad_crit=criterion:backward({output, label})
		Model:backward(im1, grad_crit[1])
         return loss2,gradParameters
	end
	--Model:updateParameters(LR)
	sgdState = sgdState or { learningRate = LR, momentum = mom,learningRateDecay = 5e-7,weightDecay=coefL2 }
	loss, parameters=optim.sgd(feval, parameters, sgdState)
	return res, loss2
end



--load the two images
function train_epoch(Model, list_folders_images, list_txt)
	
	local list_t=images_Paths(list_folders_images[1])
	list_1=create_Im_Training_list(list_t, list_txt[1])
	nbEpoch=30
	for epoch=1, nbEpoch do
		
		local stat=0
		local lossToT=0
		local NbPassToT=0
		print('--------------Epoch : '..epoch..' ---------------')
		print(#list_folders_images..' : sequences')
		nbList= #list_folders_images
		
		-- for each list (ie : each list of images in a folder)
		nbList=1
		for l=1,nbList do
			list_im=images_Paths(list_folders_images[l])
			list=create_Im_Training_list(list_im, list_txt[l])
			--list=shuffleList(list)
			NbPass=#list.im
			NbPassToT=NbPassToT+NbPass
			-- for each image in the folder	
			for i=1, NbPass do
				im=getImage(list.im[i])
				res, loss2=supervised_Training(Model,im, list.joint[i])
				lossToT=lossToT+loss2
				stat=stat+res
			end
			xlua.progress(l, #list_folders_images)
		end
		print("precision : "..stat/NbPassToT)
		print("loss : "..lossToT/NbPassToT)
		print("Nb images : "..NbPassToT)
		save_model(Model,'../Save/SupervisedSave07_07_2.t7')
		Print_performance(Model,list_1,epoch)
	end
end

local list_folders_images, list_txt=Get_HeadCamera_HeadMvt()
local reload=true

local image_width=200
local image_height=200

if reload then
	Model = torch.load('../Save/SupervisedSave07_07.t7'):double()
else
	require "mini_model"
	Model=getModel(image_width,image_height)	
end
Model=Model:cuda()

train_epoch(Model, list_folders_images, list_txt)

--print(list_folders_images[1])
--list_t=images_Paths(list_folders_images[1])
--Print_performance(Model,list_t,1)

