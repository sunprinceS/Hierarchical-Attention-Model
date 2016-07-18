--[[

  A Child-Sum Tree-LSTM with input at each node.

--]]

local ChildSumTreeLSTM, parent = torch.class('HierAttnModel.ChildSumTreeLSTM', 'HierAttnModel.TreeLSTM')

function ChildSumTreeLSTM:__init(config)
    parent.__init(self, config)
    self.cuda = config.cuda
    if self.cuda then
        self.mem_zeros = torch.zeros(self.mem_dim):cuda()
    else
        self.mem_zeros = torch.zeros(self.mem_dim)
    end

    -- composition module
    self.composer = self:new_composer()
    self.composers = {}
end

function ChildSumTreeLSTM:new_composer()
    local input = nn.Identity()()
    local child_c = nn.Identity()()
    local child_h = nn.Identity()()
    local child_h_sum = nn.Sum(1)(child_h)

    local i = nn.Sigmoid()(
        nn.CAddTable(){
            nn.Dropout(self.dropout)(nn.Linear(self.in_dim, self.mem_dim)(input)),
            nn.Dropout(self.dropout)(nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum))
    })
    local f = nn.Sigmoid()(
        HierAttnModel.CRowAddTable(){
            nn.Dropout(self.dropout)(nn.TemporalConvolution(self.mem_dim, self.mem_dim, 1)(child_h)),
            nn.Dropout(self.dropout)(nn.Linear(self.in_dim, self.mem_dim)(input)),
    })
    local update = nn.Tanh()(
        nn.CAddTable(){
            nn.Dropout(self.dropout)(nn.Linear(self.in_dim, self.mem_dim)(input)),
            nn.Dropout(self.dropout)(nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum))
    })
    local c = nn.CAddTable(){
        nn.CMulTable(){i, update},
        nn.Sum(1)(nn.CMulTable(){f, child_c})
    }

    local h
	local o = nn.Sigmoid()(
		nn.CAddTable(){
			nn.Dropout(self.dropout)(nn.Linear(self.in_dim, self.mem_dim)(input)),
			nn.Dropout(self.dropout)(nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum))
	})
	h = nn.CMulTable(){o, nn.Tanh()(c)}

    local composer = nn.gModule({input, child_c, child_h}, {c, h})
    if self.cuda then
        composer = composer:cuda()
    end
    if self.composer ~= nil then
        share_params(composer, self.composer)
    end
    return composer
end

function ChildSumTreeLSTM:new_output_module()
    if self.output_module_fn == nil then return nil end
    local output_module = self.output_module_fn()
    if self.cuda then
        output_module = output_module:cuda()
    end
    if self.output_module ~= nil then
        share_params(output_module, self.output_module)
    end
    return output_module
end


function ChildSumTreeLSTM:forward(node, inputs)
    for i = 1,node.num_children do
        self:forward(node.children[i], inputs)
    end

    local child_c, child_h = self:get_child_states(node)
    self:allocate_module(node, 'composer')
    node.state = node.composer:forward{inputs[node.idx], child_c, child_h}
    return node.state
end

function ChildSumTreeLSTM:backward(tree, inputs, grad,tree_type)
	local zeros = self.mem_zeros
	if tree_type == 'sent' then
		self:_backwardmm(tree.root,inputs,{zeros,zeros},grad)
	else
		self:_backward(tree.root,inputs,{zeros,grad})
	end
	return grad_inputs
end

function ChildSumTreeLSTM:_backward(node, inputs, grad)
	local child_c, child_h = self:get_child_states(node)
	local composer_grad = node.composer:backward(
		{inputs[node.idx], child_c, child_h},
		{grad[1], grad[2]}
	)
	self:free_module(node, 'composer')
	node.state = nil
	local child_c_grads, child_h_grads = composer_grad[2], composer_grad[3]
	for i = 1, node.num_children do
		self:_backward(node.children[i], inputs, {child_c_grads[i], child_h_grads[i]})
	end
end

function ChildSumTreeLSTM:_backwardmm(node,inputs,grad,mem_grad)
	local child_c, child_h = self:get_child_states(node)
	local composer_grad = node.composer:backward(
		{inputs[node.idx], child_c, child_h},
		{grad[1], grad[2] + mem_grad[node.memidx]}
	)
	self:free_module(node, 'composer')
	node.state = nil
	local child_c_grads, child_h_grads = composer_grad[2], composer_grad[3]
	for i = 1, node.num_children do
		self:_backwardmm(node.children[i], inputs, {child_c_grads[i], child_h_grads[i]},mem_grad)
	end
	
end

function ChildSumTreeLSTM:clean(node)
    self:free_module(node, 'composer')
    node.state = nil
    node.output = nil
    for i = 1, node.num_children do
        self:clean(node.children[i])
    end
end

function ChildSumTreeLSTM:parameters()
    local params, grad_params = {}, {}
    local cp, cg = self.composer:parameters()
    tablex.insertvalues(params, cp)
    tablex.insertvalues(grad_params, cg)
    return params, grad_params
end

function ChildSumTreeLSTM:unpack_state(state,i,tree)
    local c, h
    if state == nil then
        c, h = self.mem_zeros, self.mem_zeros
    else
        c, h = unpack(state)
    end
    return c, h
end

function ChildSumTreeLSTM:get_child_states(node)
    local child_c, child_h
    if node.num_children == 0 then
        if self.cuda then
            child_c = torch.zeros(1, self.mem_dim):cuda()
            child_h = torch.zeros(1, self.mem_dim):cuda()
        else
            child_c = torch.zeros(1, self.mem_dim)
            child_h = torch.zeros(1, self.mem_dim)
        end
    else
        if self.cuda then
            child_c = torch.CudaTensor(node.num_children, self.mem_dim)
            child_h = torch.CudaTensor(node.num_children, self.mem_dim)
        else
            child_c = torch.Tensor(node.num_children, self.mem_dim)
            child_h = torch.Tensor(node.num_children, self.mem_dim)
        end
        for i = 1, node.num_children do
            child_c[i], child_h[i] = unpack(node.children[i].state)
        end
    end
    return child_c, child_h
end

