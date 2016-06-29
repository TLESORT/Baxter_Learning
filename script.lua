

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
require 'priors'

function Rico_Training(Model, criterion, Mode,image1, image2, image3, image4)
	local LR=0.01
	local mom=0.9
        local coefL2=0
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
	elseif Mode=="Temp" then
	     loss,gradParameters=doStuff_temp(Model,criterion,gradParameters, im1,im2)
	elseif Mode=="Prop" then
	     loss,gradParameters=doStuff_Prop(Model,criterion,gradParameters,im1,im2,im3,im4)	
	elseif Mode=="Caus" then 
	     loss,gradParameters=doStuff_Caus(Model,criterion,gradParameters,im1,im2,im3,im4)
	elseif Mode=="Rep" then
	     loss,gradParameters=doStuff_Rep(Model,criterion,gradParameters,im1,im2,im3,im4)
	else print("Wrong Mode")
	end

         return loss,gradParameters
	end
	-- met à jour les parmetres avec les 2 gradients
	         -- Perform SGD step:
        sgdState = sgdState or { learningRate = LR, momentum = mom,learningRateDecay = 5e-7,weightDecay=coefL2 }
	parameters, loss=optim.sgd(feval, parameters, sgdState)
end



--load the two images
function train_epoch(Model, list_folders_images, list_txt)
	
	local list_t=images_Paths(list_folders_images[1])
	nbEpoch=1
	for epoch=1, nbEpoch do
		print('--------------Epoch : '..epoch..' ---------------')
		print(#list_folders_images..' : sequences')
		nbList= #list_folders_images
		nbList=1
		
		for l=1,nbList do
			list=images_Paths(list_folders_images[l])
			for i=1, #list-1 do
				Mode='Temp'
				im1=getImage(list,i)
				im2=getImage(list,i+1)
				Rico_Training(Model, criterion, Mode,im1, im2)
				if i<#list-3 then
					im3=getImage(list,i+2)
					im4=getImage(list,i+3)
					Mode='Prop'
					Rico_Training(Model, criterion, Mode,im1, im2,im3, im4)
					Mode='Rep'
					Rico_Training(Model, criterion, Mode,im1, im2,im3, im4)
				end
			end
			xlua.progress(l, #list_folders_images)
		end
		save_model(Model,'./Save/Save29.t7')
		Print_performance(Model,list_t,epoch)
	end
end

local list_folders_images, list_txt=Get_HeadCamera_HeadMvt()

print("hello")
print(list_txt)

local image_width=200
local image_height=200
require "mini_model"
local Model=getModel(image_width,image_height)
Model=Model:cuda()
--train_epoch(Model, list_folders_images)

tensor, label=tensorFromTxt(list_txt[1])

create_Training_list(images_Paths(list_folders_images[1]), list_txt[1])



