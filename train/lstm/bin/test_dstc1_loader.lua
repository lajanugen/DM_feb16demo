


require 'torch'
require 'loaders.dstc1'
require 'src.misc'
require 'lfs'
JSON = assert(loadfile "lib/JSON.lua")()
mobdebug = require 'mobdebug'

local loader = dstc1()
loader:read()
