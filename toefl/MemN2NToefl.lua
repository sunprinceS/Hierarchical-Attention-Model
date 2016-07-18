--[[

  Question Answering on TOEFL test (4 choices)

--]]

local MemN2NToefl = torch.class('HierAttnModel.MemN2NToefl')

function MemN2NToefl:__init(config)
	self.num_choices    = 4
	self.internal_dim   = config.internal_dim      or 75
	self.mem_size       = config.memory_size       or 915 -- 915 has been tested
	self.lr             = config.lr                or 1e-2
	self.batch_size     = config.batch_size        or 40
	self.reg            = config.reg               or 1e-4
	self.hops           = config.hops              or 1
	self.emb_lr         = 0
    self.cuda           = config.cuda

    -- attention mechanism
    self.sim = config.sim or 'dot'
    self.att_norm = config.att_norm or 'sharp'

    -- word embedding
    self.emb_dim        = config.emb_vecs:size(2)
	self.mem_dim        = self.emb_dim
    if self.cuda then
        self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim):cuda()
        self.emb.weight:copy(config.emb_vecs:cuda())
        self.in_zeros = torch.zeros(self.emb_dim):cuda()
    else
        self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
        self.emb.weight:copy(config.emb_vecs)
        self.in_zeros = torch.zeros(self.emb_dim)
    end

    -- optimizer configuration
    self.optimizer = config.optimizer or 'adagrad'
    self.optim_state = { learningRate = self.lr }

    -- negative log likelihood optimization objective
    if self.cuda then
	    self.criterion = nn.DistKLDivCriterion():cuda()
    else
	    self.criterion = nn.DistKLDivCriterion()
    end
	self.level = config.level or 'phrase'

	local memnn_config = {
		mem_dim = self.mem_dim,
		internal_dim = self.internal_dim,
		mem_size = self.mem_size,
		hops = self.hops,
		sim = self.sim,
		att_norm = self.att_norm,
        cuda = self.cuda
	}

	self.memnn = HierAttnModel.MemN2N(memnn_config)
	self.sim_output_module = self:new_sim_module()

    self.choice_modules = {}
    for c=1,4 do
        self.choice_modules[c] = self:new_choice_module()
    end

	local modules = nn.Parallel() -- to get the param in different module
		:add(self.memnn)
        :add(self.choice_modules[1])
		:add(sim_output_module)
	self.params, self.grad_params = modules:getParameters()

    for c=2,4 do
        share_params(self.choice_modules[c], self.choice_modules[1])
    end
end

function MemN2NToefl:new_choice_module()
    local input = nn.Identity()()
    local out = nn.Linear(self.mem_dim, self.internal_dim)(input)
    local choice_module = nn.gModule({input},{out})
    return choice_module
end

function MemN2NToefl:new_sim_module()
	local choices = {}
	local similarities = {}

	local mem_out = nn.Identity()()
	for i=1,self.num_choices do
		choices[i] = nn.Identity()()
		similarities[i] = nn.CosineDistance(){mem_out,choices[i]}
	end
	local sim_all = nn.JoinTable(1)({similarities[1],similarities[2],similarities[3],similarities[4]})
	local sim_module = nn.gModule({mem_out,choices[1],choices[2],choices[3],choices[4]},{sim_all})

	local sim_output_module = nn.Sequential()
		:add(sim_module)
		:add(nn.LogSoftMax())
    if self.cuda then
        sim_output_module = sim_output_module:cuda()
    end
	
	return sim_output_module
end

function MemN2NToefl:train(dataset)

	--revoke sub module
	self.memnn:training()

	local indices = torch.randperm(dataset.size) --shuffle
    local memnn_zeros = torch.zeros(self.internal_dim)

    for i = 1, dataset.size, self.batch_size do
        xlua.progress(i, dataset.size)
        local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

        local feval = function(x)
            self.grad_params:zero()
            self.emb:zeroGradParameters()

            local loss = 0
            for j = 1, batch_size do
                local idx = indices[i + j - 1]

				--query
				local query = dataset.queries[idx][1].sent
				for s=2,#dataset.queries[idx] do
					query = torch.cat(query,dataset.queries[idx][s].sent)
				end
				self.emb:forward(query)
				local emb_query
                if self.cuda then
                    emb_query = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
                else
                    emb_query = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
                end
				local query_vec = torch.mean(emb_query,1)

				--choices
				local choice_vecs = {}
				for c=1,self.num_choices do
					self.emb:forward(dataset.choices[idx][c].sent)
					local emb_choice
                    if self.cuda then
                        emb_choice = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
                    else
                        emb_choice = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
                    end
					local choice_vec = torch.mean(emb_choice,1)
                    choice_vec = torch.reshape(choice_vec, self.mem_dim)
                    choice_vec = self.choice_modules[c]:forward(choice_vec)
					table.insert(choice_vecs,choice_vec)
				end

				--memory network
				local story = dataset.sents[idx][1].sent
				for s=2,dataset.num_sent[idx] do
					story = torch.cat(story,dataset.sents[idx][s].sent)
				end
				self.emb:forward(story)
				local memory
                if self.cuda then
                    memory = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
                else
                    memory = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
                end
				
                -- reshape to 1-dim
				query_vec = torch.reshape(query_vec,self.mem_dim)
                local mem_out = self.memnn:forward(dataset.container[idx], memory,{query_vec})

				--calculate the possible choice
				pred = self.sim_output_module:forward{mem_out,choice_vecs[1],choice_vecs[2],choice_vecs[3],choice_vecs[4]}
				
				loss = loss + self.criterion:forward(pred,dataset.golden[idx])
				local sim_grad = self.criterion:backward(pred,dataset.golden[idx])
				local mem_grad = self.sim_output_module:backward({mem_out,choice_vecs[1],choice_vecs[2],choice_vecs[3],choice_vecs[4]},sim_grad)
				
				mem_grad_input,query_grad = self.memnn:backward(dataset.container[idx],mem_grad[1])

                for c=1,4 do
                    self.choice_modules[c]:backward(choice_vecs[c],mem_grad[c+1])
                end
			end

            loss = loss / batch_size
            self.grad_params:div(batch_size)

            -- regularization
            loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
            self.grad_params:add(self.reg, self.params)
            return loss, self.grad_params
        end

        if self.optimizer == 'adagrad' then
            optim.adagrad(feval, self.params, self.optim_state)
        else
            optim.adam(feval, self.params, self.optim_state)
        end
    end
    xlua.progress(dataset.size, dataset.size)
end

function MemN2NToefl:predict(container,sents,choices,queries,num_sent,num_answers,verbose)
	--revoke sub module
	self.memnn:evaluate()
    local prediction = {}
    --query
    local query = queries[1].sent
    for s=2,#queries do
        query = torch.cat(query,queries[s].sent)
    end
    self.emb:forward(query)
    local emb_query
    if self.cuda then
        emb_query = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
    else
        emb_query = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
    end
    local query_vec = torch.mean(emb_query,1)

    --choices
    local choice_vecs = {}
    for c=1,self.num_choices do
        self.emb:forward(choices[c].sent)
        local emb_choice
        if self.cuda then
            emb_choice = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
        else
            emb_choice = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
        end
        local choice_vec = torch.mean(emb_choice,1)
        choice_vec = torch.reshape(choice_vec, self.mem_dim)
        choice_vec = self.choice_modules[c]:forward(choice_vec)
        table.insert(choice_vecs,choice_vec)
    end

    --memory network
    local story = sents[1].sent
    for s=2,num_sent do
        story = torch.cat(story,sents[s].sent)
    end
    self.emb:forward(story)
    local memory
    if self.cuda then
        memory = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
    else
        memory = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
    end
    
    -- reshape to 1-dim
    query_vec = torch.reshape(query_vec,self.mem_dim)
    local mem_out = self.memnn:forward(container, memory,{query_vec})

    --calculate the possible choice
    pred = self.sim_output_module:forward{mem_out,choice_vecs[1],choice_vecs[2],choice_vecs[3],choice_vecs[4]}

	prediction = argmax_n(pred,num_answers)

	--recycle! free the module
	self.memnn:clean(container)

    return prediction
end

function MemN2NToefl:predict_dataset(dataset)
    local predictions = {}
    for i = 1, dataset.size do
        xlua.progress(i, dataset.size)
        table.insert(predictions,self:predict(dataset.container[i],dataset.sents[i],dataset.choices[i],dataset.queries[i],dataset.num_sent[i],#dataset.answers[i],i==1))
    end
    return predictions
end

function argmax_n(v,n)
	sorted,indices = torch.sort(v,true)
	ret , _ = torch.sort(indices[{{1,n}}])
	return ret
end

function argmax(v)
    local idx = 1
    local max = v[1]
    for i = 2, v:size(1) do
        if v[i] > max then
            max = v[i]
            idx = i
        end
    end
    return idx
end

function MemN2NToefl:print_config()
    local num_params = self.params:size(1)
    printf('%-25s = %d\n',   'num params', num_params)
	printf('%-25s = %d\n',   'Number of choices', self.num_choices)
    printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
    printf('%-25s = %d\n',   'MemN2N dim', self.internal_dim)
	printf('%-25s = %d\n',   'memory size', self.mem_size)
    printf('%-25s = %.2e\n', 'learning rate', self.lr)
    printf('%-25s = %d\n',   'minibatch size', self.batch_size)
    printf('%-25s = %.2e\n', 'regularization strength', self.reg)
    printf('%-25s = %d\n',   'hops', self.hops)
    printf('%-25s = %s\n',   'attention similarity', self.sim)
    printf('%-25s = %s\n',   'attention normalization', self.att_norm)
    printf('%-25s = %s\n',   'optimizer', self.optimizer)
end

function MemN2NToefl:save(path)
    local config = {
		num_choices             = self.num_choices,
		internal_dim            = self.internal_dim,
        mem_dim                 = self.mem_dim,
        mem_size                = self.mem_size,
		lr                      = self.lr,
		batch_size              = self.batch_size,
		reg                     = self.reg,
		hops                    = self.hops,
		sim                     = self.sim,
		att_norm                = self.att_norm,
        emb_vecs                = self.emb.weight:float(),
        optimizer               = self.optimizer,
        cuda                    = self.cuda
    }

    torch.save(path, {
        params = self.params,
        config = config,
    })
end

function MemN2NToefl.load(path)
    local state = torch.load(path)
	local model = HierAttnModel.MemN2NToefl.new(state.config)
    model.params:copy(state.params)
    return model
end
