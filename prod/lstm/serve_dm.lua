-- this is a sample file of how to start the service, and then send queries

-- our service is starting.. (service is starting for the first time, please give it some time)
require 'torch'
require 'src.dm_svc'
require 'src.util'
--mobdebug = require 'mobdebug'
JSON = assert(loadfile 'lib/JSON.lua')() -- note: this is global!

local service = dm_svc()

-- we are serving now..
-- print('starting timed query now..')
-- local timer = torch.Timer()

------------ helper functions --------------
function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end

function get_recent_val(dbobj, slot)
  local turns = dbobj['turns']
  for k=#turns,1,-1 do
    local nlu_k = turns[k]['nlu-out']
    local dasv_k = nlu_k['dasv']
    for l=1,#dasv_k do
      if dasv_k[l]['slot'] == slot then
        local val_str = dasv_k[l]['value']
        if val_str ~= nil then
          if #val_str>0 then
            return val_str
          end
        end
      end
    end
  end
  return ''
end
---------------------------------------------

-- read input from stdin
local size = 2^13      -- good buffer size (8K)
local input_str = io.read(size)
local input = JSON:decode(input_str)

 -- call LTM and get dbobj such that dbobj['turns'] has the necessary information
local dbfname = 'db.json'
local dbobj
if file_exists(dbfname) then
  dbobj = util:read_json(dbfname)
else
  dbobj = {}
  dbobj['turns'] = {}
end

-- append latest NLU output to dbobj
local next_turn = #dbobj['turns']+1
dbobj['turns'][next_turn] = {}
dbobj['turns'][next_turn]['nlu-out'] = input

-- call DM to get predicted system dialog acts and slots, ie, dm-out
output = service:serve(dbobj)

-- heuristic h1: update dm-out by filling in necessary values
local sel_da = {'canthelp', 'canthelp.missing_slot_value', 'expl-conf','impl-conf','inform','offer','request','select','canthelp.exception'}
local sel_da_set = {}
for i=1,#sel_da do
  sel_da_set[sel_da[i]] = true
end

for i=1,#output['dasv'] do
  -- for each dialog act
  local this_das = output['dasv'][i]
  local da = this_das['dialog-act']
  if sel_da_set[da] then
    local slot = this_das['slot']
    local value = get_recent_val(dbobj, slot)
    this_das['value'] = value
  end
end

-- heuristic h2: perform database lookup
local activate_h2 = false
if activate_h2 then
  -- perform database lookup if sys dialog-act is inform or offer
  -- for each dialog-act and slot pair
  --   if dialog-act corresponds to certain cases where system does DB lookup (event X) then
  --     if corresponding value has not been filled already by h1 then
  --       dbval = dblookup( all filled slots so far)
  --       if dbval is unique then
  --         set dbval for this dialog-act and slot pair
  --       end
  --     end
  --   end
  -- end
end

-- add dm-out to database and update database
dbobj['turns'][next_turn]['dm-out'] = output
util:write_json(dbfname, dbobj)

-- get string representation of DM's output and give it to NLG
local output_str = JSON:encode(output)
print(output_str)

-- timer:stop()
-- print(string.format('Query took %.8f seconds', timer:time().real))


