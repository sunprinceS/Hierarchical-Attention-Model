--[[

  Memory Network class

--]]

local MemN2N, parent = torch.class('HierAttnModel.MemN2N','nn.Module')

function MemN2N:__init(config)
	parent.__init(self)
	self.mem_dim = config.mem_dim
	self.internal_dim = config.internal_dim
	self.mem_size = config.mem_size
	self.hops = config.hops
	self.sim = config.sim
	self.att_norm = config.att_norm
	self.train = false
    self.use_cuda = config.cuda

	-- memnn module
	self.memnn_module = self:new_memnn_module(915)
end

function MemN2N:new_memnn_module(mem_size)
    local mem_in = nn.Identity()()
    local query_in = nn.Identity()()
	local query = nn.CAddTable()(query_in)
	local u = nn.Linear(self.mem_dim,self.internal_dim)(query)
	local c = nn.Linear(self.mem_dim,self.internal_dim)(mem_in)
	local m
	local attention
    local inputs = {}
    inputs[1] = u
    for i = 1, self.hops do
		local in_1
		if self.sim == 'dot' then
			in_1 = nn.Reshape(self.internal_dim, 1)(inputs[i])
			m = nn.Linear(self.mem_dim,self.internal_dim)(mem_in)
		else
			in_1 = nn.Reshape(self.internal_dim,1)(nn.Normalize(2)(inputs[i]))
			m = nn.Normalize(2)(nn.Linear(self.mem_dim,self.internal_dim)(mem_in))
		end
        local p_tmp = nn.MM(false, false){m, in_1}
        local p = nn.Sum(2)(p_tmp) -- dummy sum: change p into 1 dimensional tensor for softmax

		--local attention
		if self.att_norm == 'sharp' then
			local soft_p = nn.SoftMax()(p) -- perform softmax
			attention = nn.Reshape(mem_size,1)(soft_p)
		else
			local sig_p = nn.Sigmoid()(p)
			local sum = nn.Replicate(mem_size)(nn.Sum()(sig_p))
			local sig_norm_p = nn.CDivTable(){sig_p,sum}
			attention = nn.Reshape(mem_size,1)(sig_norm_p)
		end
		attention_2d = nn.Replicate(self.internal_dim,2)(attention)
        local o = nn.Sum(1)(
            nn.CMulTable(){c, attention_2d}
        )
        inputs[i+1] = nn.CAddTable(){inputs[i], o}
    end
	--trun back to the mem_dim , for cosine distance calculation
	local mem_out = inputs[self.hops+1]
	--local mem_out = nn.Linear(self.internal_dim,self.mem_dim)(inputs[self.hops+1])
    local memnn_module = nn.gModule({mem_in, query_in},{mem_out})
    if self.use_cuda then
        memnn_module = memnn_module:cuda()
    end
    if self.memnn_module ~= nil then
        share_params(memnn_module, self.memnn_module)
    end
    return memnn_module
end

function MemN2N:forward(container,memory,query)
	self:allocate_module(container,'memnn_module',memory:size(1))
	container.state = container.memnn_module:forward{memory,query}
	container.memory = memory
	container.query = query
	return container.state
end

function MemN2N:backward(container,grad)
	local memnn_grad = container.memnn_module:backward(
		{container.memory,container.query},grad)
	self:clean(container)

	return memnn_grad[1],memnn_grad[2]
end
function MemN2N:training()
	self.train = true
end

function MemN2N:evaluate()
	self.train = false
end
function MemN2N:clean(container)
	self:free_module(container,'memnn_module')
	container.memory = nil
	container.query = nil
	container.state = nil
	collectgarbage()
end

function MemN2N:allocate_module(container, module,mem_size)
    container[module] = self['new_' .. module](self,mem_size)

    -- necessary for dropout to behave properly
    if self.train then container[module]:training() else container[module]:evaluate() end
end

function MemN2N:free_module(container, module)
    if container[module] == nil then return end
    container[module] = nil
end
