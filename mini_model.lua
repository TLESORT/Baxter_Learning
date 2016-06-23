
require 'nn'

-- network-------------------------------------------------------
function getNet(image_width,image_height)
	nbFilter=32

	Timnet = nn.Sequential()

	Timnet:add(nn.SpatialConvolution(3, nbFilter, 3, 3))
	
	Timnet:add(nn.SpatialBatchNormalization(nbFilter*200*200)) 
	Timnet:add(nn.ReLU())
	Timnet:add(nn.SpatialMaxPooling(20,20,20,20))

	width=math.floor((image_width-2)/20)
	height=math.floor((image_height-2)/20)
	print(height.." : height")
	image_size=width*height 
	print(image_size.." : image size")

	Timnet:add(nn.View(nbFilter*width*height))                    -- reshapes  3D tensor into 1D tensor 
	Timnet:add(nn.Linear(nbFilter*width*height, 100))             -- fully connected layer 
	Timnet:add(nn.ReLU())  
	--Timnet:add(nn.BatchNormalization(100))                  
	Timnet:add(nn.Linear(100, 100)) 
	Timnet:add(nn.ReLU())                       
	Timnet:add(nn.Linear(100, 1))                   -- 10 is the number of outputs of the network 
	--Timnet=Timnet:cuda()

	-- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
	local method = 'xavier'
	local Timnet = require('weight-init')(Timnet, method)
	print('Timnet\n' .. Timnet:__tostring());
	return Timnet
end
