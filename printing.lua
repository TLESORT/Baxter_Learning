---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_performance(models,imgs,txt, name, Log_Folder,truth, displayPlot)

   local REP_criterion=get_Rep_criterion()
   local PROP_criterion=get_Prop_criterion()
   local CAUS_criterion=get_Caus_criterion()
   local TEMP_criterion=nn.MSDCriterion()

   local model=models.model1

   local list_out1={}

   for i=1, #imgs do
      image1=imgs[i]
      Data1=torch.Tensor(1,3,200,200)
      Data1[1]=image1
      
      model:forward(Data1:cuda())
      local State1=model.output[1]	

      table.insert(list_out1,State1)
   end

   corr=ComputeCorrelation(truth,list_out1)
   if displayPlot then
      show_figure(list_out1, Log_Folder..'state'..name..'.log')
      show_figure_normalized(list_out1,truth, Log_Folder..'stateNorm'..name..'.log',corr)
   end
   
   return corr
end

function ComputeCorrelation(truth,output)
   Truth=torch.Tensor(#truth)
   Output=torch.Tensor(#output)
   for i=1, #truth do
      Truth[i]=truth[i]
      Output[i]=output[i]
   end
   corr=torch.cmul((Truth-Truth:mean()),(Output-Output:mean())):mean()
   corr=corr/(Truth:std()*Output:std())
   return corr
end

function show_figure_normalized(output,truth, Name, corr)

   local Truth=torch.Tensor(#truth)
   local Output=torch.Tensor(#output)	
   local corr=corr or 1
   if corr<0 then 
      Variable_Truth='Normalized Truth (*-1)'
      corr=-1
   else Variable_Truth='Normalized Truth ' end
   local Variable_Output='Normalized State'

   for i=1, #truth do
      Truth[i]=truth[i]
      Output[i]=output[i]
   end
   Truth=corr*(Truth-Truth:mean())/Truth:std()
   Output=(Output-Output:mean())/Output:std()


   -- log results to files
   accLogger = optim.Logger(Name)

   for i=1, #output do
      -- update logger
      accLogger:add{[Variable_Output] = Output[i],[Variable_Truth] = Truth[i]}
   end
   -- plot logger
   accLogger:style{[Variable_Output] = '+',[Variable_Truth] = '+'}
   accLogger.showPlot = false
   accLogger:plot()
end

function show_figure(output, Name,point)
   local point=point or '+'
   local Variable_Output='State'
   local accLogger = optim.Logger(Name)
   for i=1, #output do accLogger:add{[Variable_Output] = output[i]}end
   accLogger:style{[Variable_Output] = '+'}
   accLogger.showPlot = true
   accLogger:plot()
end


