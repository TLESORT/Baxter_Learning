
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
require "Get_HeadCamera_HeadMvt"
require 'priors'

local function get_output(net,imgs,imgs_color)
	nbIm=#imgs
	local list_color, list_out={},{}
	for i=1, nbIm do
		Batch=torch.Tensor(2,3, 200, 200)
		Batch[1]=imgs[i]
		Batch[2]=imgs_color[i]
		out=net:forward(Batch)
		table.insert(list_out,out[1]:clone())
		table.insert(list_color,out[2]:clone())
	end

	return list_out, list_color
end


torch.manualSeed(123)
name_deep='./Log/3_11_reload_2114/Everything/Save3_11_reload_2114_best.t7'
--name_deep='./Log/3_11_reload_211woda/Everything/Save3_11_reload_211woda_best.t7'
name_deep_WODA='./Log/2_11_WODA/Everything/Save2_11_WODA.t7'
name_linear='./Log/2_11_1L_WODA/Everything/Save2_11_1L_WODA_best.t7'

local net = torch.load(name_deep):double()
print('net\n' .. net:__tostring());

cutorch.setDevice(2) 

local use_simulate_images=true
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt(use_simulate_images)
local last_indice=#list_folders_images
local sum_corr=0
local sum_corr_color=0

for i=1, last_indice do
	list=images_Paths(list_folders_images[i])
	truth=getTruth(list_txt[i],true)
	imgs=load_list(list, 200, 200, false)
	imgs_color=load_list(list, 200, 200, true)
	list_out,list_color=get_output(net,imgs,imgs_color)
	corr=ComputeCorrelation(list_out, truth,1)	
	sum_corr=sum_corr+corr
	corr_color=ComputeCorrelation(list_color, truth,1)
	xlua.progress(i,last_indice)
	sum_corr_color=sum_corr_color+corr_color
end
print("Mean correlation : ")
print(sum_corr/last_indice)
print("Mean correlation color : ")
print(sum_corr_color/last_indice)


