
-- adapted from https://github.com/wojciechz/learning_to_execute
-- utilities for combining/flattening parameters in a model
-- the code in this script is more general than it needs to be, which is
-- why it is kind of a large

require 'torch'
local model_utils = {}
-- this function takes the parameters that the network refers to through net:parameters() and resets
-- those tensor memories to point to a new single storage location; in other words, the networks you
-- send to this function does get changed wrt the location of the storage it refers to.
function model_utils.combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters() -- the example lstm has net_params = {1:toch.Tensor(), 2:toch.Tensor(), ...}, where
                                                               -- the torch.Tensor() instances are neural networks weight matrices W in various layers
        -- mobdebug.start()
        -- local tmp=1
        if net_params then
            for _, p in pairs(net_params) do
                -- print(_,p:size(),torch.type(p),torch.typename(p))
                parameters[#parameters + 1] = p
            end
            -- mobdebug.start()
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new -- couldn't find this function definition in whole of torch source code. could be hidden away in generic/Tensor.c
                                         -- but it looks like this returns a function that can be used to create a new tensor object

        -- following computes the unique number of parameters in the network. this may indeed be necessary, not sure
        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage() -- this returns the storage of the parameters;
                                                    -- see https://github.com/torch/torch7/blob/master/doc/tensor.md#torch.storage
            -- if storageInSet(storages, storage)==nil then
            --     local tmpa=1
            -- elseif storageInSet(storages, storage)==0 then
            --     print('0!!')
            --     mobdebug.start()
            --     local tmpa=1
            -- end
            if not storageInSet(storages, storage) then
                -- print(torch.pointer(storage),nParameters)
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            -- if not storageOffset then
            --     print('ting')
            --     mobdebug.start()
            --     local tmpa=1
            -- end
            storageOffset = storageOffset or 0
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        -- i think the following only changes the results until now if there are duplicate parameters which was handled above through
        -- finding unique parameters through storageInSet
        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end



--[[ Creates clones of the given network.
The clones share all weights and gradWeights with the original network.
Accumulating of the gradients sums the gradients properly.
The clone also allows parameters for which gradients are never computed
to be shared. Such parameters must be returns by the parametersNoGrad
method, which can be null.
--]]
function model_utils.clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            -- here we're setting the cloned parameters to point to the same memory locations as the original parameters are stored
            -- so why clone at all? I think the reason is the following: even though the parameters are shared, the input, outputs etc are not
            -- and since the inputs and outputs are abstracted away by nn within self.output, self.gradInput, we cannot do away with another way than this
            -- but thanks to lua's collectgarbage(), we won't allocate extra memory for the parameters due to the clones; extra memory only for the several
            -- different inputs and outputs which is anyway needed
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

return model_utils
