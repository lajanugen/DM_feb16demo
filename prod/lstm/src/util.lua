
-- misc utilities
local util = torch.class('util')

function util:clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function util:shuffle_tab(tab)
  local n = #tab
  for i = 1,n do
    local j = math.random(i, n)
    tab[i], tab[j] = tab[j], tab[i]
  end
  return tab
end

function util:read_json(fname)
  local f = assert(io.open(fname, "r"))
  local content = f:read("*all")
  f:close()
  local luatable = JSON:decode(content)
  return luatable
end

function util:write_json(fname, luatable)
  local pretty_json_text = JSON:encode_pretty(luatable)

  local f = assert(io.open(fname, 'w'))
  f:write(pretty_json_text)
  f:close()
end
