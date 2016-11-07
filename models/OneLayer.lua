
require 'nn'

-- network-------------------------------------------------------
function getModel()
	Timnet = nn.Sequential()
	Timnet:add(nn.View(200*200*3))                
	Timnet:add(nn.Linear(200*200*3, 1))

	-- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
	local method = 'xavier'
	local Timnet = require('weight-init')(Timnet, method)
	print('Timnet\n' .. Timnet:__tostring());
	return Timnet
end
