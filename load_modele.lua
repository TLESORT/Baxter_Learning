
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
require "Get_HeadCamera_HeadMvt"
require 'priors'

function rescale_10_200(im)
	img_rescale=torch.Tensor(3,200, 200)
	for i=1, 3 do
		for j=1,200 do
			for l=1,200 do
				img_rescale[i][j][l]=im[math.ceil(j/20)][math.ceil(l/20)]
			end
		end
	end

	return img_rescale
end

local function clampImage(tensor)
   if tensor:type() == 'torch.ByteTensor' then
      return tensor
   end
    
   local a = torch.Tensor():resize(tensor:size()):copy(tensor)
   min=a:min()
   max=a:max()
   a:add(-min)
   a:mul(1/(max-min))         -- remap to [0-1]
   return a
end

--net = torch.load('model-test.t7'):double()
net = torch.load('./Save/Save24_08_NoTrick.t7'):double()
print('net\n' .. net:__tostring());
--net=net:cuda()

local use_simulate_images=true
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt(use_simulate_images)
local last_indice=#list_folders_images
local list=images_Paths(list_folders_images[last_indice])

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


nbIm=#list
imgs=load_list(list, 200, 200)
for i=1, nbIm do
	Batch=torch.Tensor(1,3, 200, 200)
	Batch[1]=imgs[i]
	net:forward(Batch)
	--image.display{image=net:get(19).output[1], nrow=16,  zoom=10, legend="image"..i}
	im=net:get(19).output[1]
	--local format="200x200"
	--local img_rsz=image.scale(im,format)

	img_rsz=rescale_10_200(im[1])
	salience=torch.cmul(img_rsz,Batch[1])




	--image.display{image=net:get(15).output[1][10], nrow=16,  zoom=5}
	--image.display{image=net:get(16).output[1], nrow=16,  zoom=20}
	img=torch.cat(torch.cat(img_rsz,salience,3),Batch[1],3)
	tensor= clampImage(img)
	filename=paths.home.."/Bureau/Resultat_non_supervise/Temp_only/image"..i..".jpg"
	image.save(filename,tensor)
end
im=im[1]
min=im:min()
max=im:max()
im:add(-min)
im:mul(256/(max-min))
	
prob=torch.zeros(256)
for j=1,256 do
	for a=1, im:size(1) do
		for b=1, im:size(2) do
			if a~=1 and im[a][b]-im[a-1][b]==j then
				prob[j]=prob[j]+1
			end
			if b~=1 and im[a][b]-im[a][b-1]==j then
				prob[j]=prob[j]+1
			end
			if a~=im:size(1) and im[a][b]-im[a+1][b]==j then
				prob[j]=prob[j]+1
			end
			if b~=im:size(2) and im[a][b]-im[a][b+1]==j then
				prob[j]=prob[j]+1
			end
		end
	end
end
prob=prob/(im:size(1)*im:size(2))

entropy=0
for j=1, 256 do
	if prob[j] ~=0 then
		entropy=entropy-prob[j]*math.log(prob[j])
	end
end
print("entropy")
print(entropy)

--[[
path_test="/home/lesort/TrainTorch/Kaggle/PreprocessedData/Train/epoch0/c0/img_208.jpg"

image1=getImage(path_test)
net:forward(Data1)
image.display{image=net:get(14).output[1], nrow=8,  zoom=4, legend="image"}
--]]


