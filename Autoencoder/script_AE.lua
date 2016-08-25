

require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'cunn'
require 'cutorch'


require '../functions.lua'
require "../Get_HeadCamera_HeadMvt"

cutorch.setDevice(2) 


local use_simulate_images=true
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt(use_simulate_images)

require "../models/autoencoder_mini_model"
AE=getAE()
AE=AE:cuda()

nbEpoch=10
LR=0.001
for epoch=1, nbEpoch do
	print('--------------Epoch : '..epoch..' ---------------')
	print(#list_folders_images..' : sequences')
	nbList= #list_folders_images
	
	for l=1,nbList do
		list=images_Paths(list_folders_images[l])
		imgs=load_list(list)

		for i=1, #imgs do
			--train
			input=imgs[i]:cuda()
			parameters,gradParameters = AE:getParameters()
			AE:zeroGradParameters()
			output=AE:forward(input)
			criterion= nn.MSECriterion():cuda()
			criterion:forward(input,output)
			grad=criterion:backward(input,output)
			AE:backward(input,grad*-1)
			AE:updateParameters(LR)
		end
		xlua.progress(l, #list_folders_images)
	end

	
	image.display({input,output})
	save_model(AE,'../Save/AE_3x3_1TopFM.t7')
	
end


