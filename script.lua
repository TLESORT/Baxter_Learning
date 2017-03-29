require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'xlua'
require 'math'
require 'string'
require 'cunn'
require 'nngraph'

require 'MSDC'
require 'functions'
require 'printing'
require "Get_Baxter_Files"
require 'priors'

REP_criterion=get_Rep_criterion()
PROP_criterion=get_Prop_criterion()
CAUS_criterion=get_Caus_criterion()
TEMP_criterion=nn.MSDCriterion()

image_width=200
image_height=200

function Rico_Training(models, Mode,batch, coef, LR)
   local LR=LR or 0.001
   local optimizer = optim.adam
   
   -- create closure to evaluate f(X) and df/dX
   local feval = function(x)
      -- just in case:
      collectgarbage()
      --get new parameters
      if x ~= parameters then
         parameters:copy(x)
      end
      --reset gradients
      gradParameters:zero()
      if Mode=='Temp' then loss,grad=doStuff_temp(models,TEMP_criterion, batch,coef)
      elseif Mode=='Prop' then loss,grad=doStuff_Prop(models,PROP_criterion,batch,coef)
      elseif Mode=='Caus' then loss,grad=doStuff_Caus(models,CAUS_criterion,batch,coef)
      elseif Mode=='Rep' then loss,grad=doStuff_Rep(models,REP_criterion,batch,coef)
      else print("Wrong Mode")
      end
      return loss,gradParameters
   end
   optimState={learningRate=LR}
   parameters, loss=optimizer(feval, parameters, optimState)
        
   return loss[1]
end


function train_Epoch(list_folders_images,list_txt,Log_Folder,use_simulate_images,LR)

   local BatchSize=16
   local nbEpoch=10
   local totalBatch=20
   local name_save=Log_Folder..'reprLearner1d.t7'
   local coef_Temp=1
   local coef_Prop=1
   local coef_Rep=1
   local coef_Caus=2
   local coef_list={coef_Temp,coef_Prop,coef_Rep,coef_Caus}
   local list_corr={}

   local plot = true
   local loading = true

   nbList= #list_folders_images

   for crossValStep=1,nbList do

      models = createModels()

      currentLogFolder=Log_Folder..'CrossVal'..crossValStep..'/' --*

      if file_exists('imgsCv'..crossValStep..'.t7') and loading then
         print("Data Already Exists, Loading")
         imgs = torch.load('imgsCv'..crossValStep..'.t7')
         imgs_test = imgs[#imgs]
      else
         local imgs, imgs_test = loadTrainTest(list_folders_images,crossValStep)
         torch.save('imgsCv'..crossValStep..'.t7', imgs)
      end

      -- we use last list as test
      list_txt[crossValStep],list_txt[#list_txt] = list_txt[#list_txt], list_txt[crossValStep]
      local txt_test=list_txt[#list_txt]
      local truth=getTruth(txt_test,use_simulate_images)

      assert(#imgs_test==#truth,"Different number of images and corresponding ground truth, something is wrong \nNumber of Images : "..#imgs_test.." and Number of truth value : "..#truth)

      if plot then
         show_figure(truth,currentLogFolder..'GroundTruth.log')
      end
      corr=Print_performance(models, imgs_test,txt_test,"First_Test",currentLogFolder,truth,false)
      print("Correlation before training : ", corr)
      table.insert(list_corr,corr)

      print("Training")

      for epoch=1, nbEpoch do

         print('--------------Epoch : '..epoch..' ---------------')
         local lossTemp=0
         local lossRep=0
         local lossProp=0
         local lossCaus=0

         local causAdded = 0
         
         for numBatch=1,totalBatch do

            indice1=torch.random(1,nbList-1)
            repeat indice2=torch.random(1,nbList-1) until (indice1 ~= indice2)

            txt1=list_txt[indice1]
            txt2=list_txt[indice2]

            imgs1=imgs[indice1]
            imgs2=imgs[indice2]

            -- print("indice1",indice1)
            -- print("indice2",indice2)

            -- print("imgs",#imgs1)
            -- print("imgs",#imgs2)

            -- print("txt1",txt1)
            -- print("txt2",txt2)

            -- if numBatch%3==0 then
            --    causAdded = causAdded + 1
            --    batch=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Caus")
            --    lossCaus = lossCaus + Rico_Training(models, 'Caus',batch, 1,LR)
            -- end

            batch=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Temp")
            lossTemp = lossTemp + Rico_Training(models,'Temp',batch, coef_Temp,LR)

            batch=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Caus")
            lossCaus = lossCaus + Rico_Training(models, 'Caus',batch, 1,LR)

            batch=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Prop")
            lossProp = lossProp + Rico_Training(models, 'Prop',batch, coef_Prop,LR)

            batch=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Rep")
            lossRep = lossRep + Rico_Training(models,'Rep',batch, coef_Rep,LR)


            
            xlua.progress(numBatch, totalBatch)

         end
         corr=Print_performance(models, imgs_test,txt_test,"Test",currentLogFolder,truth,false)
         print("Correlation : ", corr)
         print("lossTemp",lossTemp/totalBatch)
         print("lossProp",lossProp/totalBatch)
         print("lossRep",lossRep/totalBatch)
         print("lossCaus",lossCaus/(totalBatch+causAdded))
         table.insert(list_corr,corr)
         
      end
      corr=Print_performance(models, imgs_test,txt_test,"Test",currentLogFolder,truth,plot)
      --show_figure(list_corr,currentLogFolder..'correlation.log','-')

      --for reiforcement, we need mean and std to normalize representation
      print("SAVING MODEL AND REPR")
      saveMeanAndStdRepr(imgs)
      models.model1:float()
      save_model(models.model1,name_save)

      list_txt[crossValStep],list_txt[#list_txt] = list_txt[#list_txt], list_txt[crossValStep]


   end --*
end

local LR=0.001
local dataAugmentation=true
local Log_Folder='./Log/'
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt()

require('./models/convolutionnal')

torch.manualSeed(42) -- This one is very tough, more than one epoch, and dies
--torch.manualSeed(1337)
train_Epoch(list_folders_images,list_txt,Log_Folder,use_simulate_images,LR)


imgs={} --memory is free!!!!!
