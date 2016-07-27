
require 'nn'
require 'torch'
require "noiseModule"
-- network-------------------------------------------------------

ni=3--input feature map
no=24--output feature map
kw=3 --dim noyau
kh=3 -- dim noyau - height
pw=2--padding
ph=2--padding

encoder=nn.SpatialConvolution(ni,no,kw,kh,1,1,pw,ph)
encoder_2=nn.SpatialConvolution(no,no,kw,kh,1,1,pw,ph)

decoder=nn.SpatialFullConvolution(no,ni,kw,kh,1,1,pw,ph)
decoder_2=nn.SpatialFullConvolution(no,no,kw,kh,1,1,pw,ph)

decoder:share(encoder,'weight','gradWeight')
decoder_2:share(encoder_2,'weight','gradWeight')

AE = nn.Sequential()
AE:add(encoder)
AE:add(nn.ReLU())
AE:add(encoder_2)
AE:add(nn.ReLU())
AE:add(nn.noiseModule())
AE:add(nn.SpatialDropout(0.5))
AE:add(decoder_2)
AE:add(decoder)



function getAE()
	return AE
end
