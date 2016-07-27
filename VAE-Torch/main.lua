-- Joost van Amersfoort - <joost@joo.st>
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'image'

nngraph.setDebug(false)

local VAE = require 'VAE_conv'
require 'KLDCriterion'
require 'GaussianCriterion'
require 'Sampler2'

--For loading data files
require 'load'

require '../functions.lua'
require "../Get_HeadCamera_HeadMvt"

local continuous = true
--data = load(continuous)

list_folders_images, list_txt=Get_HeadCamera_HeadMvt()


batch_size = 2
 height=100
 lenght=100
channel=1

torch.manualSeed(1)

local input_size = height*lenght*channel
--local input_size = data.train:size(2)
local latent_variable_size = 30
local hidden_layer_size = 900

local encoder = VAE.get_encoder(input_size, hidden_layer_size, latent_variable_size)
local decoder = VAE.get_decoder(input_size, hidden_layer_size, latent_variable_size, continuous)

local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})

local reconstruction, reconstruction_var, model
if continuous then
    reconstruction, reconstruction_var = decoder(z):split(2)
    model = nn.gModule({input},{reconstruction, reconstruction_var, mean, log_var})
    criterion = nn.GaussianCriterion()
else
    reconstruction = decoder(z)
    model = nn.gModule({input},{reconstruction, mean, log_var})
    criterion = nn.BCECriterion()
    criterion.sizeAverage = false
end

-- Some code to draw computational graph
--dummy_x = torch.rand(input_size)
--model:forward({dummy_x})

-- Uncomment to get structure of the Variational Autoencoder
 --graph.dot(.fg, 'Variational Autoencoder', 'VA')

KLD = nn.KLDCriterion()

local parameters, gradients = model:getParameters()

local config = {
    learningRate = 0.001
}

local state = {}

local maxEpoch=10
for epoch = 0, maxEpoch do
epoch = epoch + 1
for l=1, #list_folders_images do

	local list=images_Paths(list_folders_images[l])

	img=load_list(list,height,lenght)

	data={train={}}
	for j=1, #img do
		table.insert(data.train,img[j])
	end

    local lowerbound = 0
    local tic = torch.tic()

    local shuffle = torch.randperm(#data.train)

    -- This batch creation is inspired by szagoruyko CIFAR example.
    local indices = torch.randperm(#data.train):long():split(batch_size)
    indices[#indices] = nil
    local N = #indices * batch_size

    local tic = torch.tic()
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)
	local inputs = torch.Tensor(batch_size, input_size)
	inputs[1]=data.train[t][1]:resize(input_size)
	inputs[2]=data.train[t][1]:resize(input_size)

        local opfunc = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end

            model:zeroGradParameters()
            local reconstruction, reconstruction_var, mean, log_var
            if continuous then
                reconstruction, reconstruction_var, mean, log_var = unpack(model:forward(inputs))
                reconstruction = {reconstruction, reconstruction_var}
            else
                reconstruction, mean, log_var = unpack(model:forward(inputs))
            end
if l%10==1 and t==1 then
im1=inputs[1]
im2=torch.Tensor(reconstruction[1]:size())
im2:copy(reconstruction[1])
image.display({image=im1:resize(channel,height,lenght), zoom=4})
image.display({image=im2:resize(channel,height,lenght), zoom=4})
end
	
inputs[1]=inputs[1]:resize(channel,height,lenght)
inputs[2]=inputs[2]:resize(channel,height,lenght)

            local err = criterion:forward(reconstruction, inputs)
            local df_dw = criterion:backward(reconstruction, inputs)

            local KLDerr = KLD:forward(mean, log_var)
            local dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(mean, log_var))

            if continuous then
                error_grads = {df_dw[1], df_dw[2], dKLD_dmu, dKLD_dlog_var}
            else
                error_grads = {df_dw, dKLD_dmu, dKLD_dlog_var}
            end

            model:backward(inputs, error_grads)

            local batchlowerbound = err + KLDerr

            return batchlowerbound, gradients
        end

        x, batchlowerbound = optim.adam(opfunc, parameters, config, state)

        lowerbound = lowerbound + batchlowerbound[1]
    end

    print("Epoch: " .. epoch .. " Lowerbound: " .. lowerbound/N .. " time: " .. torch.toc(tic)) 

    if lowerboundlist then
        lowerboundlist = torch.cat(lowerboundlist,torch.Tensor(1,1):fill(lowerbound/N),1)
    else
        lowerboundlist = torch.Tensor(1,1):fill(lowerbound/N)
    end

    if epoch % 2 == 0 then
        torch.save('save/parameters.t7', parameters)
        torch.save('save/state.t7', state)
        torch.save('save/lowerbound.t7', torch.Tensor(lowerboundlist))
    end
end
end
