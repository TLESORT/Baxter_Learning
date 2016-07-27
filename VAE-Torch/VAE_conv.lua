require 'torch'
require 'nn'

local VAE = {}

ni=3--input feature map
no=24--output feature map
kw=3 --dim noyau
kh=3 -- dim noyau - height
pw=2--padding
ph=2--padding

encoder=nn.SpatialConvolution(ni,no,kw,kh,1,1,pw,ph)
encoder_2=nn.SpatialConvolution(ni,no,kw,kh,1,1,pw,ph)

decoder=nn.SpatialFullConvolution(no,ni,kw,kh,1,1,pw,ph)
decoder_2=nn.SpatialFullConvolution(no,ni,kw,kh,1,1,pw,ph)

decoder:share(encoder,'weight','gradWeight')
decoder_2:share(encoder_2,'weight','gradWeight')

function VAE.get_encoder(input_size, hidden_layer_size, latent_variable_size)
     -- The Encoder
    local encoder = nn.Sequential()
    
    mean_logvar = nn.ConcatTable()
    mean_logvar:add(encoder)
    mean_logvar:add(encoder_2)

    encoder:add(mean_logvar)
    
    return encoder
end

function VAE.get_decoder(input_size, hidden_layer_size, latent_variable_size, continuous)
    -- The Decoder
    local decoder = nn.Sequential()

    if continuous then
        mean_logvar = nn.ConcatTable()
        mean_logvar:add(decoder)
        mean_logvar:add(decoder_2)
        decoder:add(mean_logvar)
    else
        decoder:add(nn.Linear(hidden_layer_size, input_size))
        decoder:add(nn.Sigmoid(true))
    end

    return decoder
end

return VAE
