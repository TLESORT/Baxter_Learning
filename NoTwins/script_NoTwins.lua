

require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'cunn'

require '../MSDC'
require '../functions.lua'
require "../Get_HeadCamera_HeadMvt"
require 'priors_NoTwins'

function Rico_Training(Model, Mode,batch)
	local LR=0.01
	local mom=0.9
        local coefL2=0,5
	local criterion=nn.MSDCriterion()
	criterion=criterion:cuda()

	parameters,gradParameters = Model:getParameters()

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
	elseif Mode=='Temp' then
	     loss,gradParameters=doStuff_temp(Model,criterion,gradParameters, batch)
	elseif Mode=='Prop' then
	     loss,gradParameters=doStuff_Prop(Model,criterion,gradParameters,batch)	
	elseif Mode=='Caus' then 
	     --coefL2=0.5  -- unstable in other case
	     loss,gradParameters=doStuff_Caus(Model,criterion,gradParameters,batch)
	elseif Mode=='Rep' then
	     --coefL2=1  -- unstable in other case
	     loss,gradParameters=doStuff_Rep(Model,criterion,gradParameters,batch)
	else print("Wrong Mode")end
         return loss,gradParameters
	end
	-- met à jour les parmetres avec les 2 gradients
	         -- Perform SGD step:
        sgdState = sgdState or { learningRate = LR, momentum = mom,learningRateDecay = 5e-7,weightDecay=coefL2 }

	state=state or {learningRate = LR,paramVariance=nil, weightDecay=0.0005 }
	config=config or {}
	optim.adagrad(feval, parameters,config, state)
	--Model:updateParameters(LR)

	--parameters, loss=optim.sgd(feval, parameters, sgdState)
end



--load the two images
function train_epoch(Model, list_folders_images, list_txt)
	local BatchSize=5
	local list_t=images_Paths(list_folders_images[1])
	nbEpoch=10
	for epoch=1, nbEpoch do
		print('--------------Epoch : '..epoch..' ---------------')
		print(#list_folders_images..' : sequences')
		nbList= #list_folders_images
		
		nbList=1-------------------------------!!!!----------------
		
		for l=1,nbList do
			list=images_Paths(list_folders_images[l])
			imgs=load_list(list)
			list_Prop, list_Temp=create_Head_Training_list(list, list_txt[l])
			NbBatch=math.floor((#list_Prop.Mode+#list_Temp.Mode)/BatchSize)
			NbBatch=10*10*10
			for numBatch=1, NbBatch do
				i=math.random(1,#list_Temp.Mode)
				Batch_Temp=getBatch(imgs,list_Temp, i, BatchSize, 200, 200,"Temp")
				Rico_Training(Model, 'Temp',Batch_Temp)

				i=math.random(1,#list_Prop.Mode)
				Batch_Prop=getBatch(imgs,list_Prop, i, BatchSize, 200, 200,"Prop")
				Rico_Training(Model, 'Prop',Batch_Prop)

				xlua.progress(numBatch, NbBatch)
			end
			xlua.progress(l, #list_folders_images)
		end
		save_model(Model,'../Save/SaveNoTwins18_07.t7')
		Print_performance(Model,imgs,epoch)
	end
end

local list_folders_images, list_txt=Get_HeadCamera_HeadMvt()
local reload=true

local image_width=200
local image_height=200

if reload then
	Model = torch.load('../Save/SaveNoTwins08_07.t7'):double()
else
	require "../models/mini_model"
	Model=getModel(image_width,image_height)	
end
Model=Model:cuda()

train_epoch(Model, list_folders_images, list_txt)

