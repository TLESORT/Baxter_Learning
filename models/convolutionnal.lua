B
require 'nn'

-- network-------------------------------------------------------
function getModel()
	nbFilter=32

	Timnet = nn.Sequential()

	Timnet:add(nn.SpatialConvolution(3, nbFilter, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(nbFilter))
	Timnet:add(nn.ReLU())	
	Timnet:add(nn.SpatialMaxPooling(2,2,2,2))

	Timnet:add(nn.SpatialConvolution(nbFilter, 2*nbFilter, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(2*nbFilter)) 
	Timnet:add(nn.ReLU())	
	Timnet:add(nn.SpatialMaxPooling(2,2,2,2))

	Timnet:add(nn.SpatialConvolution(2*nbFilter, 4*nbFilter, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(4*nbFilter)) 
	Timnet:add(nn.ReLU())	
	Timnet:add(nn.SpatialMaxPooling(2,2,2,2))

	Timnet:add(nn.SpatialConvolution(4*nbFilter, 8*nbFilter, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(8*nbFilter)) 
	Timnet:add(nn.ReLU())	
	Timnet:add(nn.SpatialMaxPooling(2,2,2,2))

	Timnet:add(nn.SpatialConvolution(8*nbFilter, 1, 1, 1))
	Timnet:add(nn.SpatialBatchNormalization(1)) 
	Timnet:add(nn.ReLU())
	Timnet:add(nn.View(100))                
	Timnet:add(nn.Linear(100, 500))
	Timnet:add(nn.ReLU())                    
	Timnet:add(nn.Linear(500, 1))

	-- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
	local method = 'xavier'
	local Timnet = require('weight-init')(Timnet, method)
	print('Timnet\n' .. Timnet:__tostring());
	return Timnet
end
