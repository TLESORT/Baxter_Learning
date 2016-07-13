
require 'nn'

-- network-------------------------------------------------------
function getModel(image_width,image_height)
	nbFilter=64

	Timnet = nn.Sequential()
	Timnet:add(nn.SpatialConvolution(3, nbFilter, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(nbFilter))
	Timnet:add(nn.ReLU())
	Timnet:add(nn.SpatialMaxPooling(2,2,2,2,1,1))
	Timnet:add(nn.SpatialConvolution(nbFilter, 32, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(32)) 
	Timnet:add(nn.ReLU())
	Timnet:add(nn.SpatialMaxPooling(2,2,2,2,1,1))
	Timnet:add(nn.SpatialConvolution(32, 16, 1, 1))
	Timnet:add(nn.SpatialBatchNormalization(16)) 
	Timnet:add(nn.ReLU())
	Timnet:add(nn.SpatialMaxPooling(10,10,10,10))
	Timnet:add(nn.View(16*5*5))                    -- reshapes  3D tensor into 1D tensor 
	Timnet:add(nn.Linear(16*5*5, 100))
	Timnet:add(nn.ReLU())                    
	Timnet:add(nn.Linear(100, 1))                   -- 10 is the number of outputs of the network 
	--Timnet=Timnet:cuda()

	-- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
	local method = 'xavier'
	local Timnet = require('weight-init')(Timnet, method)
	print('Timnet\n' .. Timnet:__tostring());
	return Timnet
end