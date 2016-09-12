---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_performance(Models,imgs,txt, name, Log_Folder, use_simulate_images)

	local REP_criterion=get_Rep_criterion()
	local PROP_criterion=get_Prop_criterion()
	local CAUS_criterion=get_Caus_criterion()
	local TEMP_criterion=nn.MSDCriterion()

	local Temp=0
	local Rep=0
	local Prop=0
	local Caus=0
	local Model=Models.Model1

	local list_out1={}

	for i=1, #imgs do
		image1=imgs[i]
		Data1=torch.Tensor(1,3,200,200)
		Data1[1]=image1
		
		Model:forward(Data1:cuda())
		local State1=Model.output[1]	

		table.insert(list_out1,State1)
	end

	-- biased estimation of test loss
	local nb_sample=100

	for i=1, nb_sample do
		Prop_batch=getRandomBatch(imgs, txt, 1, 200, 200, 'Prop', use_simulate_images)
		Temp_batch=getRandomBatch(imgs, txt, 1, 200, 200, 'Temp', use_simulate_images)
		Caus_batch=getRandomBatch(imgs, txt, 1, 200, 200, 'Caus', use_simulate_images)
		
		Temp=Temp+doStuff_temp(Models,TEMP_criterion, Temp_batch)
		Prop=Prop+doStuff_Prop(Models,PROP_criterion,Prop_batch)	
		Caus=Caus+doStuff_Caus(Models,CAUS_criterion,Caus_batch)
		Rep=Rep+doStuff_Rep(Models,REP_criterion,Prop_batch)
	end


	show_figure(list_out1, Log_Folder..'state'..name..'.log', 1000)

	return Temp/nb_sample,Prop/nb_sample, Rep/nb_sample, Caus/nb_sample, list_out1
end


---------------------------------------------------------------------------------------
-- Function : Print_Grad(Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_Grad(Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list,Log_Folder)

	local scale= 1000
	local Name = Log_Folder..'Grad.log'
	local accLogger = optim.Logger(Name)

	for i=1, #Temp_grad_list do
	-- update logger
		accLogger:add{['Temp_Grad*'..scale] = Temp_grad_list[i]*scale,
				['Prop_Grad*'..scale] = Prop_grad_list[i]*scale,
				['Rep_Grad*'..scale] = Rep_grad_list[i]*scale,
				['Caus_Grad*'..scale] = Caus_grad_list[i]*scale}
	end
	-- plot logger
	accLogger:style{['Temp_Grad*'..scale] = '-',
			['Prop_Grad*'..scale] = '-',
			['Rep_Grad*'..scale] = '-',
			['Caus_Grad*'..scale] = '-'}
	accLogger.showPlot = false
	accLogger:plot()
end

---------------------------------------------------------------------------------------
-- Function : Print_Loss(Temp_Train,Prop_Train,Rep_Train,Temp_Test,Prop_Test,Rep_Test,Log_Folder)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_Loss(Temp_Train,Prop_Train,Rep_Train,Caus_Train,Temp_Test,Prop_Test,Rep_Test,Caus_Test,Log_Folder)
	show_loss(Temp_Train,Temp_Test, Log_Folder..'Temp_loss.log', 1000)
	show_loss(Prop_Train,Prop_Test, Log_Folder..'Prop_loss.log', 1000)
	show_loss(Rep_Train,Rep_Test, Log_Folder..'Rep_loss.log', 1000)
	show_loss(Caus_Train,Caus_Test, Log_Folder..'Caus_loss.log', 1000)
end


---------------------------------------------------------------------------------------
-- Function : show_loss(list_train, list_test, Name , scale)
-- Input (list_train): list of the train loss
-- Input (list_test): list of the test loss
-- Input (Name): Name of the file
-- Input (scale): multiplicator factor needed because for optim.logger 1.1=1 but 11~=10
---------------------------------------------------------------------------------------
function show_loss(list_train, list_test, Name , scale)

	local scale=scale or 1000
	-- log results to files
	local accLogger = optim.Logger(Name)

	for i=1, #list_train do
	-- update logger
		accLogger:add{['train*'..scale] = list_train[i]*scale,['test*'..scale] = list_test[i]*scale}
	end
	-- plot logger
	accLogger:style{['train*'..scale] = '-',['test*'..scale] = '-'}
	accLogger.showPlot = false
	accLogger:plot()
end

---------------------------------------------------------------------------------------
-- Function : show_figure(list_out1, Name , scale)
-- Input (list_out1): list of the estimate state
-- Input (Name) : Name of the file
-- Input (scale) : multiplicator factor needed because for optim.logger 1.1=1 but 11~=10
---------------------------------------------------------------------------------------
function show_figure(list_out1, Name , scale, Variable_Name)

	Variable_Name=Variable_Name or 'out1'

	local scale=scale or 1000
	-- log results to files
	accLogger = optim.Logger(Name)

	for i=1, #list_out1 do
	-- update logger
		accLogger:add{[Variable_Name] = list_out1[i]*scale}
	end
	-- plot logger
	accLogger:style{[Variable_Name] = '+'}
	accLogger.showPlot = false
	accLogger:plot()
end
