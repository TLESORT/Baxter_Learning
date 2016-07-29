
require 'nn'


-- network-------------------------------------------------------
function getModel()
	nbFilter=32

	Timnet = nn.Sequential()
	Timnet:add(nn.SpatialConvolution(3, nbFilter, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(nbFilter))
	Timnet:add(nn.ReLU())
	Timnet:add(nn.SpatialConvolution(nbFilter, 32, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(nbFilter)) 
	Timnet:add(nn.ReLU())
	Timnet:add(nn.SpatialMaxPooling(5,5,5,5))
	Timnet:add(nn.SpatialConvolution(nbFilter, 1, 1, 1))
	Timnet:add(nn.SpatialBatchNormalization(1)) 
	Timnet:add(nn.ReLU())
	Timnet:add(nn.View(39*39))                
	Timnet:add(nn.Linear(39*39, 500))
	Timnet:add(nn.ReLU())                    
	Timnet:add(nn.Linear(500, 1))

	-- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
	local method = 'xavier'
	local Timnet = require('weight-init')(Timnet, method)
	print('Timnet\n' .. Timnet:__tostring());
	return Timnet
end
