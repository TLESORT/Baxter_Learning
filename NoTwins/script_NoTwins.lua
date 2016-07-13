

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

function Rico_Training(Model, Mode,image1, image2, image3, image4)
	local LR=0.01
	local mom=0.9
        local coefL2=0,5
	local criterion=nn.MSDCriterion()
	criterion=criterion:cuda()
	
	if image1 then im1=image1:cuda() end
	if image2 then im2=image2:cuda() end
	if image3 then im3=image3:cuda() end
	if image4 then im4=image4:cuda() end

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
	     loss,gradParameters=doStuff_temp(Model,criterion,gradParameters, im1,im2)
	elseif Mode=='Prop' then
	     loss,gradParameters=doStuff_Prop(Model,criterion,gradParameters,im1,im2,im3,im4)	
	elseif Mode=='Caus' then 
	     --coefL2=0.5  -- unstable in other case
	     loss,gradParameters=doStuff_Caus(Model,criterion,gradParameters,im1,im2,im3,im4)
	elseif Mode=='Rep' then
	     --coefL2=1  -- unstable in other case
	     loss,gradParameters=doStuff_Rep(Model,criterion,gradParameters,im1,im2,im3,im4)
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
	
	local list_t=images_Paths(list_folders_images[1])
	nbEpoch=10
	for epoch=1, nbEpoch do
		print('--------------Epoch : '..epoch..' ---------------')
		print(#list_folders_images..' : sequences')
		nbList= #list_folders_images
		
		nbList=1-------------------------------!!!!----------------
		
		for l=1,nbList do
			--list=images_Paths(list_folders_images[l])
			list_Prop, list_Temp=create_Head_Training_list(images_Paths(list_folders_images[l]), list_txt[l])
			NbPass=#list_Prop.Mode+#list_Temp.Mode
			--NbPass=20
			for k=1, NbPass do
				i=math.random(1,#list_Temp.Mode)
				im1=getImage(list_Temp.im1[i])
				im2=getImage(list_Temp.im2[i])
				Rico_Training(Model, 'Temp',im1, im2)

				i=math.random(1,#list_Prop.Mode)
				im1=getImage(list_Prop.im1[i])
				im2=getImage(list_Prop.im2[i])
				im3=getImage(list_Prop.im3[i])
				im4=getImage(list_Prop.im4[i])
				--image.display{image=({im1,im2,im3,im4}), zoom=1}
				Rico_Training(Model, 'Prop',im1, im2,im3,im4)
				--Rico_Training(Model,'Rep',im1,im2,im3,im4)

				xlua.progress(k, NbPass)
			end
			xlua.progress(l, #list_folders_images)
		end
		save_model(Model,'./Save/SaveNoTwins08_07.t7')
		Print_performance(Model,list_t,epoch)
	end
end

local list_folders_images, list_txt=Get_HeadCamera_HeadMvt()
local reload=false

local image_width=200
local image_height=200

if reload then
	Model = torch.load('./Save/SaveNoTwins08_07.t7'):double()
else
	require "../models/mini_model"
	Model=getModel(image_width,image_height)	
end
Model=Model:cuda()

train_epoch(Model, list_folders_images, list_txt)

--print(list_folders_images[1])
--list_t=images_Paths(list_folders_images[1])
--Print_performance(Model,list_t,1)
