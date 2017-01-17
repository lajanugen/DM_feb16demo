
local dstc1 = torch.class('dstc1')

function dstc1:__init(mode)
  self.mode = mode
end

local numl, numt, tot, tot_turns, tot_miss_turns = 0,0,0,0,0
function dstc1:walk(folder)
  for filename in lfs.dir(folder) do
    if filename:match("%.json$") then -- "%." is an escaped ".", "$" is end of string
      if filename:match("%.labels%.json$") then
        local fname = string.format('%s/%s', folder, filename)
        -- print(fname)
        local dialog = read_json(fname)
        -- tabprint(dialog,0)

        local is_labelled, num_turns_miss, num_turns = self:is_labelled(dialog)
        numl = numl + (is_labelled and 1 or 0)
        numt = numt + (self:is_transcribed(dialog) and 1 or 0)
        tot = tot + 1
        tot_miss_turns = tot_miss_turns + num_turns_miss
        tot_turns = tot_turns + num_turns
      end
    elseif filename~=nil and filename~='..' and filename~='.' and not filename:match("%.swp$") and not filename:match("~$") then
      self:walk(string.format('%s/%s', folder, filename))
    end
  end
end

function dstc1:read()
  local folders = {'train1a', 'train1b', 'train1c', 'train2', 'dstc_data_train3_v00/train3', 'test1', 'test2', 'test4', 'dstc_data_test3/test3'} --
  print('#labelled', '#transcribed', '#tot-dialogs', '#labelled-%', '#transcribed-%', 'avg-lab-miss', 'avg-turns')
  for i=1,#folders do
    local folder = string.format('data/gitlfs/dstc1/%s', folders[i])
    print(string.format('walking %s..',folder))
    self:walk(folder)
    print(numl, numt, tot, numl/tot*100, numt/tot*100, tot_miss_turns/tot, tot_turns/tot)
    numl, numt, tot, tot_turns, tot_miss_turns = 0,0,0,0,0
  end

  -- mobdebug.start()
  -- local tmpa=1
end

function dstc1:is_labelled(dialog)
  -- is every turn labelled?
  local turns = dialog['turns']
  local num_turns_miss = 0
  local is_labelled = true
  for i=1,#turns do
    local turn = turns[i]
    if not turn['slu-labels'] then
      -- print(i)
      -- return false
      is_labelled = false
      num_turns_miss = num_turns_miss + 1
    else
      local turn_labelled = false
      for j=1,#turn['slu-labels'] do
        local slulab = turn['slu-labels'][j]
        if slulab['label'] == true then
          turn_labelled = true
          break
        end
      end
      if not turn_labelled then
        -- print(i)
        -- return false
        is_labelled = false
        num_turns_miss = num_turns_miss + 1
      end
    end
  end
  return is_labelled, num_turns_miss, #turns
end

function dstc1:is_transcribed(dialog)
  -- is every turn transcribed?
  local turns = dialog['turns']
  for i=1,#turns do
    local turn = turns[i]
    if turn['transcription-status']~='transcribed' then
      return false
    end
  end
  return true
end

function tabprint(tab, n)
  for key,val in pairs(tab) do
    local str=''
    for i=1,n do
      str = str .. ' '
    end
    print(string.format('%s%s',str,key))
    if type(val) == 'table' then
      tabprint(val,n+2)
    end
  end
end

