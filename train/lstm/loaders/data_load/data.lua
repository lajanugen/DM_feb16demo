JSON = (loadfile "JSON.lua")() 

function scandir(directory)
	local i, t, popen = 0, {}, io.popen
	--print(directory)
	--print(popen('ls -a "'..directory..'"'))
	dir = popen('ls -a "'..directory..'"')
    for filename in dir:lines() do
		if filename ~= '.' and filename ~= '..' then
			table.insert(t,directory .. '/' .. filename)
		end
    end
	dir:close()
    return t
end

function get_files(num_dialogs,all)
	data_dir = '../data'
	labels, logs, conv = {}, {}, {}
	count = 0
	for _,dir1 in ipairs(scandir(data_dir)) do
		for _,dir2 in ipairs(scandir(dir1)) do
			for _,file in ipairs(scandir(dir2)) do
				if string.sub(file,-8,-6) == 'log' then
					table.insert(logs,file)
				else
					table.insert(labels,file)
					table.insert(conv,file)
				end
				count = count + 1
			end
		end
	end
	return labels, logs
end

function read_json(fname)
  local f = io.open(fname, "r")
  local content = f:read("*all")
  f:close()
  local luatable = JSON:decode(content)
  return luatable
end

--dirs = scandir('./data')
--print(dirs)

return {get_files=get_files, read_json=read_json}
