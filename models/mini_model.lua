
require 'nn'


-- network-------------------------------------------------------
function getModel(image_width,image_height)
	nbFilter=32
	nbFilter2=32--16

	Timnet = nn.Sequential()
	Timnet:add(nn.SpatialConvolution(3, nbFilter, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(nbFilter))
	Timnet:add(nn.ReLU())
	--Timnet:add(nn.SpatialMaxPooling(2,2,2,2,1,1))
	Timnet:add(nn.SpatialConvolution(nbFilter, 32, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(32)) 
	Timnet:add(nn.ReLU())
	--Timnet:add(nn.SpatialMaxPooling(2,2,2,2,1,1))
	Timnet:add(nn.SpatialConvolution(32, nbFilter2, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(nbFilter2)) 
	Timnet:add(nn.ReLU())
	Timnet:add(nn.SpatialMaxPooling(10,10,10,10))
	Timnet:add(nn.View(nbFilter2*19*19))  --nbFilter2*5*5                  
	Timnet:add(nn.Linear(nbFilter2*19*19, 100))
	Timnet:add(nn.ReLU())                    
	Timnet:add(nn.Linear(100, 1))                   -- 10 is the number of outputs of the network 
	--Timnet=Timnet:cuda()

	-- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
	local method = 'xavier'
	local Timnet = require('weight-init')(Timnet, method)
	print('Timnet\n' .. Timnet:__tostring());
	return Timnet
end
