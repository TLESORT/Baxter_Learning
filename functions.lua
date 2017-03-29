 function createModels()

   if LOADING then
      print("Loading Model")
      model = torch.load(Log_Folder..'20e.t7')
   else
      model=getModel()
   end

   model=model:cuda()
   parameters,gradParameters = model:getParameters()
   model2=model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
   model3=model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
   model4=model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')

   models={model1=model,model2=model2,model3=model3,model4=model4}
   return models

end

function loadTrainTest(list_folders_images, crossValStep)

   imgs = {}
   print("Loading Images")

   if not file_exists('saveImgsRaw.t7') then
      print("nbList",nbList)
      for i=1,nbList do
         list=images_Paths(list_folders_images[i])
         table.insert(imgs,load_list(list,image_width,image_height,false))
      end
      torch.save('saveImgsRaw.t7',imgs)
   else
      imgs = torch.load('saveImgsRaw.t7')
   end

   -- switch value, because all functions consider the last element to be the test element
   imgs[crossValStep], imgs[#imgs] = imgs[#imgs], imgs[crossValStep]
   print("Preprocessing")
   imgs,mean,std = preprocessing(imgs)

   imgs_test = imgs[#imgs]
   return imgs, imgs_test
   
end

function save_model(model,path)
   --print("Saved at : "..path)
   model:float()
   parameters, gradParameters = model:getParameters()
   local lightModel = model:clone():float()
   lightModel:clearState()
   torch.save(path,lightModel)
end

function load_list(list)
   local im={}
   local lenght=image_width or 200
   local height=image_height or 200
   for i=1, #list do
      table.insert(im,getImage(list[i]))
   end 
   return im
end

function getImage(im)
   if im=='' or im==nil then return nil end
   local image1=image.load(im,3,'byte')
   return image1
   -- local format=length.."x"..height
   -- local img1_rsz=image.scale(image1,format)
   -- return img1_rsz:float()
end

function meanAndStd(imgs)

   local length,height = imgs[1][1][1]:size(1), imgs[1][1][1]:size(2)

   local mean = {torch.zeros(length,height),torch.zeros(length,height),torch.zeros(length,height)}
   local std = {torch.zeros(length,height),torch.zeros(length,height),torch.zeros(length,height)}

   for i=1,3 do
      mean[i] = mean[i]:float()
      std[i] = std[i]:float()
   end
   
   local numSeq = #imgs-1
   local totImg = 0

   for i=1,numSeq do
      for j=1,#(imgs[i]) do
         mean[1] = mean[1]:add(imgs[i][j][{1,{},{}}]:float())
         mean[2] = mean[2]:add(imgs[i][j][{2,{},{}}]:float())
         mean[3] = mean[3]:add(imgs[i][j][{3,{},{}}]:float())
         totImg = totImg+1
      end
   end

   mean[1] = mean[1] / totImg
   mean[2] = mean[2] / totImg
   mean[3] = mean[3] / totImg

   for i=1,numSeq do
      for j=1,#(imgs[i]) do
         std[1] = std[1]:add(torch.pow(imgs[i][j][{1,{},{}}]:float() - mean[1],2))
         std[2] = std[2]:add(torch.pow(imgs[i][j][{2,{},{}}]:float() - mean[2],2))
         std[3] = std[3]:add(torch.pow(imgs[i][j][{3,{},{}}]:float() - mean[3],2))
      end
   end

   std[1] = torch.sqrt(std[1] / totImg)
   std[2] = torch.sqrt(std[2] / totImg)
   std[3] = torch.sqrt(std[3] / totImg)

   torch.save('Log/meanStdImages.t7',{mean,std})
   return mean,std
end

function normalize(im,mean,std)
   for i=1,3 do
      im[{i,{},{}}] = (im[{i,{},{}}]:add(-mean[i])):cdiv(std[i])
   end
   return im
end

function preprocessingTest(imgs,mean,std)
   --Normalizing all images
   for i=1,#imgs do
      im = imgs[i]
      imgs[i] = normalize(im,mean,std)
   end
   return imgs
end

function preprocessing(imgs,meanStd)
   -- Calculate reformat imgs, mean and std for images in train set
   -- normalize train set and apply to test
   
   imgs = scaleAndCrop(imgs)
   if not meanStd then
      mean, std = meanAndStd(imgs)
   else
      mean, std = meanStd[1], meanStd[2]
   end
   
   numSeq = #imgs-1
   for i=1,numSeq do
      for j=1,#(imgs[i]) do
         im = imgs[i][j]
         imgs[i][j] =  dataAugmentation(im, mean,std)
      end
   end
   imgs[#imgs] = preprocessingTest(imgs[#imgs], mean,std)

   return imgs, mean, std
end


function scaleAndCrop(imgs, length, height)
   -- Why do i scale and crop after ? Because this is the way it's done under python,
   -- so we need to do the same conversion

   local lengthBeforeCrop = 320
   local lengthAfterCrop = length or 200
   local height = height or 200
   local formatBefore=lengthBeforeCrop.."x"..height

   for s=1,#imgs do
      for i=1,#imgs[s] do
         local img=image.scale(imgs[s][i],formatBefore)
         local img= image.crop(img, 'c', lengthAfterCrop, height)
         imgs[s][i] = img:float()
         -- image.display(img)
         -- io.read()
      end
   end
   return imgs
end

function scaleAndRandomCrop(imgs, length, height)
   local length = length or 200
   local height = height or 200
   local cropSize = 32
   
   for s=1,#imgs do
      -- Apply random modification on the images for the whole sequence
      local format=length+cropSize.."x"..height+cropSize
      local posX, posY = torch.random(cropSize),torch.random(cropSize)

      for i=1,#imgs[s] do
         local img1_rsz=image.scale(imgs[s][i],format)
         local img = image.crop(img1_rsz, posX, posY, posX+length, posY+height)
         imgs[s][i] = img:float()
         -- image.display(img)
         -- io.read()
      end
   end
   return imgs
end
   

function dataAugmentation(im, mean, std)
   local channels = {'r','g','b'}
   local noiseReductionFactor = 4 -- the bigger, less noise
   local length = im:size(2)
   local width = im:size(3)
   local maxShift = 1

   im = normalize(im, mean, std)
   return im

   -- for i=1,3 do
   --    colorShift = torch.uniform(-maxShift,maxShift)
   --    im[{i,{},{}}] = im[{i,{},{}}] + colorShift
   -- end
   
   -- -- Adding Gaussian noise to the data
   -- noise=torch.rand(3,length,width)/noiseReductionFactor
   -- noise = noise - 0.5/noiseReductionFactor --center noise

   -- im = normalize(im, mean, std):add(noise:float())
   -- return im
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

function saveMeanAndStdRepr(imgs, show, model)
   local Log_Folder='./Log/'
   local allRepr = {}
   local totImgs = 0
   local mean = nil
   local std = nil

   -- ===== Uncomment if you want to display images (and use qlua instead of th)
   if show then
      w = image.display(image.lena()) -- with positional arguments mode
   end

   
   for s=1,#imgs do
      local imgs_test = imgs[s]
      for i=1,#imgs_test do
         local img = torch.zeros(1,3,200,200)
         img[1] = imgs_test[i]

         if model then
            allRepr[#allRepr+1] = model:forward(img:float())
         else
            allRepr[#allRepr+1] = models.model1:forward(img:cuda()):float()
         end
         --====== Printing the state corresponding to the image =====
         -- ====== don't forget to uncomment the line 'w = image ... " above
         if show then
            image.display{image=img, win=w}
            print(allRepr[#allRepr][1])
            io.read()
         end

         if mean then
            mean = torch.add(mean, allRepr[#allRepr])
         else
            mean = allRepr[#allRepr]
         end
      end
   end

   mean = mean / #allRepr
   -- print("mean",mean)
   -- print("allRepr",allRepr[5],allRepr[150],allRepr[400],allRepr[250])
   
   for i=1,#allRepr do
      if std then
         std = std:add(torch.pow(allRepr[i] - mean,2))
      else
         std = torch.pow(allRepr[i] - mean,2)
      end
   end
   std = torch.sqrt(std / #allRepr)
   -- print("sumStd", std)
   torch.save(Log_Folder..'meanStdRepr.t7',{mean,std})
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

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

