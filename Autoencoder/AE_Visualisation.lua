
require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'cunn'

require 'noiseModule'

require '../functions.lua'
require "../Get_HeadCamera_HeadMvt"

net = torch.load('../Save/AE_3x3.t7'):double()
net=net:cuda()

net2 = torch.load('../Save/AE_3x3_NoWeightSharing.t7'):double()
net2=net2:cuda()

require "../models/autoencoder_mini_model"
temoin=getAE()
temoin=temoin:cuda()

print('net\n' .. net:__tostring());

local list_folders_images, list_txt=Get_HeadCamera_HeadMvt()
local list=images_Paths(list_folders_images[1])

imgs=load_list(list)
input=imgs[1]:cuda()
output=net:forward(input)
output=net2:forward(input)
output_temoin=temoin:forward(input)


BatchSize=2

im_width=100
im_height=100


print(net:get(3).weight:size())
--image.display({image=net:get(3).weight[1],zoom=10, legend='trained'})
--image.display({image=net2:get(3).weight[1],zoom=10, legend='NoSharing'})
--image.display({image=temoin:get(3).weight[1],zoom=10, legend='temoin'})
image.display({image=input, legend='input'})
image.display({image=net:get(16).output, legend='Reconstruction WeightSharing'})
image.display({image=net2:get(16).output, legend='Reconstruction  NoSharing'})--NoSharing

image.display({image=net:get(10).output[1], legend='WeightSharing'})
image.display({image=net2:get(10).output[1], legend='NoSharing'})--NoSharing
image.display({image=temoin:get(10).output[1], legend='temoin'})--temoin

image.display({image=net:get(11).output[1], legend='Noise'})

--print(net:get(6).weight:size())

--[[
image0=PreTraitement(testData0,1)
image.display{image=image0.data[1],  zoom=4, legend="image0"}
local prediction=net:forward(image0.data)
maxs, indices = torch.max(prediction,1)
print("truth = " .. testData0.label[1])
print("classe = " .. indices[1]-1)
output0=net:get(15).output[1]
image.display{image=image0.data,  zoom=4, legend="image0"}
image.display{image=output0, nrow=16,  zoom=2, legend="image0"}


image1=PreTraitement(testData1,1)
local prediction=net:forward(image1.data)
maxs, indices = torch.max(prediction,1)
print("truth = " .. testData1.label[1])
print("classe = " .. indices[1]-1)
output1=net:get(15).output[1]
image.display{image=image1.data,  zoom=4, legend="image1"}
image.display{image=output1, nrow=16,  zoom=2, legend="image1"}
--]]