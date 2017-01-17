require('../../../train/lstm/loaders/data_load/dialogs_b')
local dm_data = torch.class('dm_data')

function dm_data:__init()
  -- set state and whatever is necessary
  self.map = {}
  self.imap = {}
  self.vocab_size = nil

  self.data = dialogs('all',1)

  --self:read_vocab()

end


function dm_data:read_vocab()
  -- read vocab and do whatever
end


-- input will be json string corresponding to {'turns': [<turn object>]}, ie, a list of turn objects wrapped into a dictionary with key 'turns'
-- format of <turn object> will be as defined in the LTM spec
-- encoded_input will be same as the matrices passed during training time,
-- but with batch_size=1, and a mask_mat with all ones. That is for example:
-- return {user_mat, system_mat, mask_mat}
function dm_data:encode_input(input)
  -- input_json = JSON:decode(input)
  local input_json = input
  local num_turns = #input_json['turns']
  local user_mat   = torch.Tensor(1,self.data.NUSacts,num_turns)
  local system_mat = torch.Tensor(1,self.data.NSSacts,num_turns)
  local mask_mat   = torch.ones(1,num_turns)

  -- mobdebug.start()
  -- local tmpa=1
  for i = 1,num_turns do
    local usr = self.data:get_tensor_user_demo  (input_json['turns'][i]['nlu-out']['dasv'])
    local sys = torch.zeros(self.data.NSSacts)
    if i ~= num_turns then
    sys = self.data:get_tensor_sys_demo(input_json['turns'][i]['dm-out']['dasv'])
    end
    user_mat[{{1},{},{i}}]:copy(usr)
    system_mat[{{1},{},{i}}]:copy(sys)
  end

  return {user_mat,system_mat,mask_mat}
end

-- output will be the dasv representation predicted by the dm (note that this will be system side dasv; note: values need not be filled here)
-- encoded_output will be json string containing dialog acts and slots as provided in LTM spec, that is for example:
-- return {
--           "dasv": [{
--                         "dialog-act": "str rep",
--                         "slot": "str rep",
--                         "value": "str rep"
--                     }]
--         }
function dm_data:encode_output(output)
	return self.data:mat_to_sentence(output)
end

