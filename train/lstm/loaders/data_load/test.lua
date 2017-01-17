require 'dialogs_b'
require 'JSON'
JSON = (loadfile "JSON.lua")()

local a = dialogs('all',1)
--	print(a:get_next_batch('valid'))
for i = 1,1000 do
	a:get_next_batch('train')
end

