
require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'cutorch'

require 'MSDC'
require 'functions.lua'
require "Get_HeadCamera_HeadMvt"
require 'priors'


--net = torch.load('model-test.t7'):double()
net = torch.load('./Save/SupervisedSave07_07.t7'):double()
print('net\n' .. net:__tostring());
--net=net:cuda()


list_folders_images, list_txt=Get_HeadCamera_HeadMvt()

local list1=images_Paths(list_folders_images[1])

--[[
image1=getImage(list1[17])
local Data1=image1:cuda()
net:forward(Data1)
local State1=net.output[1]	
--]]	


--image.display(net.get(1).weight)
--print(net:get(1).weight:size())
--print(net:get(1).weight)

--print(net:get(1).weight[1][1][{}][{}])
--print(net:get(1).weight[1][{}][1][{}])
--image.display{image=net:get(1).weight[{}][{}][{}][{}], nrow=8,  zoom=100, legend="image"}
--image.display(net:get(1).output)
--image.display(net:get(8).output)






for i=1, #list1 do
	image1=getImage(list1[i])
	Data1=image1--:cuda()
	net:forward(Data1)
	image.display{image=net:get(8).output[1], nrow=8,  zoom=1, legend="image"..i}
	--image.display{image=net:get(15).output[1][10], nrow=16,  zoom=5}
	--image.display{image=net:get(16).output[1], nrow=16,  zoom=20}
end


--[[
path_test="/home/lesort/TrainTorch/Kaggle/PreprocessedData/Train/epoch0/c0/img_208.jpg"

image1=getImage(path_test)
net:forward(Data1)
image.display{image=net:get(14).output[1], nrow=8,  zoom=4, legend="image"}
--]]


