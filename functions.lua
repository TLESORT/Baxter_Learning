require 'image'

function save_model(model,path)
   --print("Saved at : "..path)
   model:cuda()
   parameters, gradParameters = model:getParameters()
   local lightModel = model:clone():float()
   lightModel:clearState()
   torch.save(path,lightModel)
end
function load_list(list, train)
   local im={}
   local lenght=image_width or 200
   local height=image_height or 200
   for i=1, #list do
      table.insert(im,getImage(list[i],lenght,height,train))
   end 
   return im
end

function getImage(im,length,height, train)
   if im=='' or im==nil then return nil end
   local image1=image.load(im,3,'byte')
   local format=length.."x"..height
   local img1_rsz=image.scale(image1,format)
   return img1_rsz:float()
end

function meanAndStd(imgs)
   mean = {0,0,0}
   std = {0,0,0}
   totImg = 0

   numPix = imgs[1][1][1]:size(1)*imgs[1][1][1]:size(2)

   -- Don't forget the '-1' to calculate std only on the train set
   for i=1, nbList-1 do
      for j=1,#(imgs[i]) do
         mean[1] = mean[1] + imgs[i][j][{1,{},{}}]:sum()
         mean[2] = mean[2] + imgs[i][j][{2,{},{}}]:sum()
         mean[3] = mean[3] + imgs[i][j][{3,{},{}}]:sum()
         totImg = totImg+1
      end
   end

   mean[1] = mean[1] / (totImg*numPix)
   mean[2] = mean[2] / (totImg*numPix)
   mean[3] = mean[3] / (totImg*numPix)

   -- Don't forget the '-1' to calculate std only on the train set
   for i=1, nbList-1 do
      for j=1,#(imgs[i]) do
         std[1] = std[1] + torch.pow(imgs[i][j][{1,{},{}}] - mean[1],2):sum()
         std[2] = std[2] + torch.pow(imgs[i][j][{2,{},{}}] - mean[2],2):sum()
         std[3] = std[3] + torch.pow(imgs[i][j][{3,{},{}}] - mean[3],2):sum()
      end
   end

   std[1] = math.sqrt(std[1] / (totImg*numPix))
   std[2] = math.sqrt(std[2] / (totImg*numPix))
   std[3] = math.sqrt(std[3] / (totImg*numPix))

   return mean,std
end

function normalize(im,mean,std)

   for i=1,3 do
      im[{i,{},{}}] = (im[{i,{},{}}] - mean[i])/std[i]
   end
   return im

end

function preprocessingTest(imgs,mean,std)

   for i=1,#imgs do
      im = imgs[i]
      imgs[i] = normalize(im,mean,std)
   end

   return imgs
end
   
function preprocessing(imgs,mean,std)

   mean, std = meanAndStd(imgs)
   numSeq = #imgs-1
   
   for i=1,numSeq do
      for j=1,#(imgs[i]) do
         im = imgs[i][j]
         imgs[i][j] =  dataAugmentation(im, mean,std)
      end
   end

   return imgs
   
end

local function gamma(im)
   local Gamma= torch.Tensor(3,3)
   local channels = {'r','g','b'}
   local mean = {}
   local std = {}
   for i,channel in ipairs(channels) do
      for j,channel in ipairs(channels) do
         if i==j then Gamma[i][i] = im[{i,{},{}}]:var()
         else
            chan_i=im[{i,{},{}}]-im[{i,{},{}}]:mean()
            chan_j=im[{j,{},{}}]-im[{j,{},{}}]:mean()
            Gamma[i][j]=(chan_i:t()*chan_j):mean()
         end
      end
   end
   return Gamma
end

local function transformation(im, v,e)
   local transfo=torch.Tensor(3,200,200)
   local Gamma=torch.mv(v,e)
   for i=1, 3 do
      transfo[i]=im[i]+Gamma[i]
      io.read()
   end
   return transfo
end

function dataAugmentation(im, mean, std)
   local channels = {'r','g','b'}
   local noiseReductionFactor = 2 -- the bigger, less noise
   local length = im:size(2)
   local width = im:size(3)
   local maxShift = 50

   for i=1,3 do
      colorShift = torch.uniform(-maxShift,maxShift)
      im[{i,{},{}}] = im[{i,{},{}}] + colorShift
   end
   
   -- Adding Gaussian noise to the data
   noise=torch.rand(3,length,width)/noiseReductionFactor
   noise = noise - 0.5/noiseReductionFactor --center noise

   im = normalize(im):add(noise:float())

   return im

end

function normalize(data)
   -- Name channels for convenience
   local channels = {'r','g','b'}
   local mean = {}
   local std = {}
   
   for i,channel in ipairs(channels) do
      -- normalize each channel globally:
      mean[i] = data[i]:mean()
      std[i] = data[{i,{},{}}]:std()
      data[{i,{},{}}]:add(-mean[i])
      data[{i,{},{}}]:div(std[i])
   end

   return data
end

function getRandomBatch(imgs1, imgs2, txt1, txt2, lenght, Mode)

   local width=image_width or 200
   local height=image_height or 200

   if Mode=="Prop" or Mode=="Rep" then
      Batch=torch.Tensor(4, lenght,3, width, height)
   else
      Batch=torch.Tensor(2, lenght,3, width, height)
   end
   
   for i=1, lenght do
      if Mode=="Prop" or Mode=="Rep" then
         if txt1==txt2 then Set=get_one_random_Prop_Set(txt1)
         else Set=get_two_Prop_Pair(txt1, txt2) end
         Batch[1][i]=imgs1[Set.im1]
         Batch[2][i]=imgs1[Set.im2]
         Batch[3][i]=imgs2[Set.im3]
         Batch[4][i]=imgs2[Set.im4]
      elseif Mode=="Temp" then
         Set=get_one_random_Temp_Set(#imgs1)
         Batch[1][i]=imgs1[Set.im1]
         Batch[2][i]=imgs1[Set.im2]
      elseif Mode=="Caus" then
         Set=get_one_random_Caus_Set(txt1, txt2)
         Batch[1][i]=imgs1[Set.im1]
         Batch[2][i]=imgs2[Set.im2]
      else
         print "getRandomBatch Wrong mode "
      end
   end
   return Batch

end


