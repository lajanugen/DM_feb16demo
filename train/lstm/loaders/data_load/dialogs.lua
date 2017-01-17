
local dialogs = torch.class('dialogs')
function dialogs:__init(mode)

	self.user_acts = {'ack','affirm','bye','hello','help','negate','null','repeat','reqalts','reqmore','restart','silence','thankyou','confirm','deny','inform','request'}

	self.system_acts = {'affirm','bye','canthear','confirm-domain','negate','repeat','reqmore','welcomemsg','canthelp','canthelp.missing_slot_value','expl-conf',
											'impl-conf','inform','offer','request','select'}

	self.allacts = {'affirm','bye','canthear','confirm-domain','negate','repeat','reqmore','welcomemsg','canthelp','canthelp.missing_slot_value','expl-conf',
										'impl-conf','inform','offer','request','select','ack','hello','null','reqalts','restart','silence','thankyou','confirm','deny','canthelp.exception'}

	self.slots = {'area','food','name','pricerange','addr','phone','postcode','signature','count'}

	self.user_act_inds		= {}
	self.system_act_inds	= {}
	self.allact_inds		= {}
	self.slot_inds			= {}

	local i
	for i = 1,#self.user_acts	do self.user_act_inds[self.user_acts[i]]		= i end
	for i = 1,#self.system_acts	do self.system_act_inds[self.system_acts[i]]	= i end
	for i = 1,#self.allacts		do self.allact_inds[self.allacts[i]]			= i end
	for i = 1,#self.slots		do self.slot_inds[self.slots[i]]				= i end

	--self.data = require('data')
	local data_path = '/home/llajan/dstc2'
	self.train = read_file(data_path .. '/config/dstc2_train.flist')
	self.valid = read_file(data_path .. '/config/dstc2_dev.flist')
	self.test  = read_file(data_path .. '/config/dstc2_test.flist')
	self.alldata = {self.train,self.valid,self.test}

	self.dir_prefix = { data_path .. '/train/',
						data_path .. '/train/',
						data_path .. '/test/'}

	self.NUacts = #self.user_acts
	self.NSacts = #self.system_acts
	self.Nacts	= #self.allacts --self.NSacts + self.NUacts
	self.Nslots = #self.slots

	self.dialog_ptr = {1,1,1}

	if mode == 'debug-rand' then
		math.randomseed(os.time())
		self.debug_dialog = math.floor(math.random()*#self.train)
	elseif mode == 'debug-142' then
		self.debug_dialog = 29
	elseif mode == 'debug-first' then
		self.debug_dialog = 1
	else
		assert(mode=='all', 'unknown mode!')
	end

	self.mode = mode
	if self.mode ~= 'all' then
		self.batch_debug,_ = self:get_next_batch('train')
		self:reset_pointer('train')
	end

	self.debug_print = false
end

function dialogs:get_repr_size()
	return 1 + self.Nacts + self.Nslots
end

function dialogs:reset_pointer(train_test_valid)
	self.dialog_ptr[self:get_id(train_test_valid)] = 1
end

--function dialogs:get_next_batch()
--	return self.get_next_dialog()
--end

function get_act_slot(dact)
	local dialog_act = dact['act']
	--print(dialog_act)
	local slots = dact['slots']

	local slots_present = {}
	local values_present = {}

	if #slots == 1 then -- 1 arg
		if slots[1][1] == 'slot' then
			table.insert(slots_present,slots[1][2] )
			table.insert(values_present,nil)
		else
			table.insert(slots_present,slots[1][1])
			table.insert(values_present,slots[1][2])
		end
	elseif #slots >= 2 then --This happens only for dact 'canthelp'
		for i = 1,#slots do
			table.insert(slots_present,slots[i][1])
			table.insert(values_present,slots[i][2])
		end
	--else -- 0 args
	--	slot = ''
	end
	--if #slots == 2 then
	--	print('2 slots')
	--	print(dact)
	--end
	return dialog_act, slots_present, values_present
end

function dialogs:get_id(train_test_valid)
	if train_test_valid == 'train' then
		return 1
	elseif train_test_valid == 'val' then
		return 2
	else
		return 3
	end
end

function dialogs:get_num_data(train_test_valid)
	return(#self.alldata[self:get_id(train_test_valid)])
end

function dialogs:get_next_batch(train_test_valid)
--function dialogs:get_next_dialog()
    local tvt = self:get_id(train_test_valid)

	local user, system
	if self.mode == 'all' then
		user = read_json	(self.dir_prefix[tvt] .. self.alldata[tvt][self.dialog_ptr[tvt]] .. '/label.json')
		system = read_json	(self.dir_prefix[tvt] .. self.alldata[tvt][self.dialog_ptr[tvt]] .. '/log.json')
	else
		user = read_json	(self.dir_prefix[tvt] .. self.alldata[tvt][self.debug_dialog] .. '/label.json')
		system = read_json	(self.dir_prefix[tvt] .. self.alldata[tvt][self.debug_dialog] .. '/log.json')
	end


	local scan_complete = false
	self.dialog_ptr[tvt] = self.dialog_ptr[tvt] + 1
	if self.dialog_ptr[tvt] > #self.alldata[tvt] then
		self.dialog_ptr[tvt] = 1
		scan_complete = true
	end

	local turns = #user['turns']
	local dialog_tensor = nil
	local mask = {}
	local act_tensor
	--print(conv[1])
	local utt_list = {}
	for i = 1,turns-1 do
			local user_acts	= user['turns'][i]['semantics']['json']
			local system_acts = system['turns'][i]['output']['dialog-acts']
		--if #acts > 1 then
			local system_transcript = system['turns'][i+1]['output']['transcript']
			local user_transcript	= user['turns'][i]['transcription']
			local turn_utt = {usr=user_transcript,sys=system_transcript}
			table.insert(utt_list,JSON:encode(turn_utt))

			act_tensor = torch.Tensor(1 + self.Nacts + self.Nslots):fill(0)
			if dialog_tensor == nil then dialog_tensor = act_tensor
			else						 dialog_tensor = torch.cat(dialog_tensor,act_tensor,2)
			end
			table.insert(mask,1)
			for system_act = 1,#system_acts do
				act,slots,values = get_act_slot(system_acts[system_act])
				--print('system',act,slots)
				--local t
				--print(act)
				--for _,t in ipairs(slots) do
				--	print(t)
				--end

				-- Batch Tensor
				act_tensor = torch.Tensor(1 + self.Nacts + self.Nslots):fill(0)
				act_tensor[1 + self.allact_inds[act]] = 1
				for _,slot in ipairs(slots) do
					if slot and self.slot_inds[slot] then
						act_tensor[1 + self.Nacts + self.slot_inds[slot]] = 1
					end
				end
				if dialog_tensor == nil then dialog_tensor = act_tensor
				else						 dialog_tensor = torch.cat(dialog_tensor,act_tensor,2)
				end
				table.insert(mask,1)

			end
			-- EOD --
			act_tensor = torch.Tensor(1 + self.Nacts + self.Nslots):fill(0)
			act_tensor[1] = 1
			dialog_tensor = torch.cat(dialog_tensor,act_tensor,2)
			table.insert(mask,2)

			if i ~= turns then -- ignore last user turn

				if self.debug_print then
					io.write('#####################################','\n')
					io.write(i,user['turns'][i]['transcription'],'\n')
					io.write('####','\n')
					io.write(user_acts,'\n')
				end
				--print(#user_acts)
				if #user_acts == 0 then
					user_acts = {}
					user_acts[1] = {act = 'null',slots = {}}
				end
				for user_act = 1,#user_acts do
					act,slots,values = get_act_slot(user_acts[user_act])
					--print('user',act,slots)
					if self.debug_print then
						io.write('* Act ' .. tostring(user_act),'\n')
						io.write(act,'\n')
						io.write(slots,'\n')
					end

					-- Batch Tensor
					act_tensor = torch.Tensor(1 + self.Nacts + self.Nslots):fill(0)
					act_tensor[1 + self.allact_inds[act]] = 1
					for _,slot in ipairs(slots) do
						if slot and self.slot_inds[slot] then
							act_tensor[1 + self.Nacts + self.slot_inds[slot]] = 1
						end
					end

					if self.debug_print then
						io.write('-------------------------------------','\n')
						io.write('Tensor : ','\n')
						for i = 1,1+self.Nacts+self.Nslots do
							io.write(act_tensor[i])
							io.write(' ')
						end
						io.write('\n')
						--print(act_tensor:reshape(1,act_tensor:size(1)))
						io.write('Acts : ')
						for i = 1,self.Nacts do
							io.write(self.allacts[i])
							io.write(' ')
							io.write(act_tensor[1+i])
							io.write('   ')
						end
						io.write('\n')
						io.write('Slots : ')
						for i = 1,self.Nslots do
							io.write(self.slots[i])
							io.write(' ')
							io.write(act_tensor[1+self.Nacts+i])
							io.write('  ')
						end
						io.write('\n')
						io.write('-------------------------------------','\n')
					end

					if dialog_tensor == nil then dialog_tensor = act_tensor
					else						 dialog_tensor = torch.cat(dialog_tensor,act_tensor,2)
					end
					table.insert(mask,0)
				end
				if self.debug_print then
					io.write('#####################################','\n')
				end
			end

		--end
	end
	dialog_tensor = dialog_tensor:reshape(1,dialog_tensor:size(1),dialog_tensor:size(2))

	return JSON:encode(utt_list)
	--if self.mode ~= 'all' and self.batch_debug then
	--	return self.batch_debug,scan_complete
	--else
	--	return {dialog_tensor,torch.Tensor(mask)},scan_complete
	--end
end

function read_file(filename)
	local file = io.open(filename, "r");
	local arr = {}
	for line in file:lines() do
		table.insert (arr, line);
	end
	return arr
end

function read_json(fname)
  local f = io.open(fname, "r")
  local content = f:read("*all")
  f:close()
  local luatable = JSON:decode(content)
  return luatable
end

