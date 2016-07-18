--[[

  Question Answering on TOEFL test (4 choices)

--]]

local TreeLSTMToefl = torch.class('HierAttnModel.TreeLSTMToefl')

function TreeLSTMToefl:__init(config)
	self.num_choices    = 4
	self.mem_dim        = config.mem_dim           or 75 --TreeLSTM memory dimension
	self.lr             = config.lr                or 0.05
	self.emb_lr         = config.emb_lr            or 0.1
	self.batch_size     = config.batch_size        or 25
	self.reg            = config.reg               or 1e-4
    self.structure      = config.structure         or 'treelstm'
    self.cuda           = config.cuda


    -- word embedding
    self.emb_dim = config.emb_vecs:size(2)
    if self.cuda then
        self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim):cuda()
        self.in_zeros = torch.zeros(self.emb_dim):cuda()
        self.emb.weight:copy(config.emb_vecs:cuda())
    else
        self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
        self.in_zeros = torch.zeros(self.emb_dim)
        self.emb.weight:copy(config.emb_vecs)
    end


    -- optimizer configuration
    self.optim_state = { learningRate = self.lr }

    -- negative log likelihood optimization objective
    if self.cuda then
	    self.criterion = nn.DistKLDivCriterion():cuda()
    else
	    self.criterion = nn.DistKLDivCriterion()
    end
	local treelstm_config = {
		in_dim = self.emb_dim,
		mem_dim = self.mem_dim,
        gate_output = true,
        cuda = self.cuda
	}

    if self.structure == 'treelstm' then
        self.treelstm = HierAttnModel.ChildSumTreeLSTM(treelstm_config)
    else
        error('invalid parse tree type: ' .. self.structure)
    end

	self.sim_output_module = self:new_sim_module()

	local modules = nn.Parallel() -- to get the param in different module
		:add(self.treelstm)
		:add(sim_output_module)
	self.params, self.grad_params = modules:getParameters()
end

function TreeLSTMToefl:new_sim_module()
	local choices = {}
	local similarities = {}

	local mem_in = nn.Identity()()
    local mem_out = nn.CAddTable()(mem_in)
	for i=1,self.num_choices do
		choices[i] = nn.Identity()()
		similarities[i] = nn.CosineDistance(){mem_out,choices[i]}
	end
	local sim_all = nn.JoinTable(1)({similarities[1],similarities[2],similarities[3],similarities[4]})
	local sim_module = nn.gModule({mem_in,choices[1],choices[2],choices[3],choices[4]},{sim_all})

	local sim_output_module = nn.Sequential()
		:add(sim_module)
		:add(nn.LogSoftMax())

    if self.cuda then
        sim_output_module = sim_output_module:cuda()
    end
	
	return sim_output_module
end

function TreeLSTMToefl:train(dataset)

	--revoke sub module
	self.treelstm:training()

	local indices = torch.randperm(dataset.size) --shuffle
	local tree_mem_zeros = torch.zeros(self.mem_dim)

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
				local emb_queries = {}
                local rep_queries = {}
				for q=1,#dataset.queries[idx] do
					self.emb:forward(dataset.queries[idx][q].sent)
					local emb_query
                    if self.cuda then
                        emb_query = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
                    else
                        emb_query = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
                    end
					local _, query_vec = self.treelstm:unpack_state(self.treelstm:forward(dataset.queries[idx][q].tree.root, emb_query))
					table.insert(emb_queries,emb_query)
                    table.insert(rep_queries, query_vec)
				end

				--choices
				local choice_vecs = {}
				local emb_choices = {}
				for c=1,self.num_choices do
					self.emb:forward(dataset.choices[idx][c].sent)
					local emb_choice
                    if self.cuda then
                        emb_choice = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
                    else
                        emb_choice = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
                    end
					table.insert(emb_choices,emb_choice)
					local choice_root = dataset.choices[idx][c].tree.root
					local _, choice_vec = self.treelstm:unpack_state(self.treelstm:forward(choice_root, emb_choices[c]))
					table.insert(choice_vecs,choice_vec)
				end

				local emb_sents  = {}
                local rep_sents = {}
				for s=1,dataset.num_sent[idx] do
					local sent = dataset.sents[idx][s].sent
					local tree = dataset.sents[idx][s].tree
					self.emb:forward(sent)

					-- forward thru one tree
					local emb_sent
                    if self.cuda then
                        emb_sent = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
                    else
                        emb_sent = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
                    end
					table.insert(emb_sents,emb_sent)

					self.treelstm:forward(dataset.sents[idx][s].tree.root,emb_sent)
                    _, rep_sents[s] = self.treelstm:unpack_state(dataset.sents[idx][s].tree.root.state)
                end

                local story = {}
				for s=1,dataset.num_sent[idx] do
                    table.insert(story, rep_sents[s])
                end

				for q=1,#dataset.queries[idx] do
                    table.insert(story, rep_queries[q])
                end

				local pred = self.sim_output_module:forward{story,choice_vecs[1],choice_vecs[2],choice_vecs[3],choice_vecs[4]}
				loss = loss + self.criterion:forward(pred,dataset.golden[idx])
				local output_grad = self.criterion:backward(pred,dataset.golden[idx])
                local sim_grad = self.sim_output_module:backward({story,choice_vecs[1],choice_vecs[2],choice_vecs[3],choice_vecs[4]}, output_grad)

				for c=1,self.num_choices do
					self.treelstm:backward(dataset.choices[idx][c].tree,emb_choices[c],sim_grad[c+1],'choice')
                end
                
				--backward thru backround knowledge sent
                local treelstm_grad = sim_grad[1]
				for s=1,dataset.num_sent[idx] do
					self.treelstm:backward(dataset.sents[idx][s].tree,emb_sents[s],treelstm_grad[s],'_sent') -- for sentence level
				end
				--backward thru query
				for q=1,#dataset.queries[idx] do
					self.treelstm:backward(dataset.queries[idx][q].tree,emb_queries[q],treelstm_grad[dataset.num_sent[idx]+q],'query')
				end
            end

            loss = loss / batch_size
            self.grad_params:div(batch_size)
            self.emb.gradWeight:div(batch_size)
			self.emb:updateParameters(self.emb_lr) -- norm to num input

            -- regularization
            loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
            self.grad_params:add(self.reg, self.params)
            return loss, self.grad_params
        end

        optim.adagrad(feval, self.params, self.optim_state)
    end
    xlua.progress(dataset.size, dataset.size)
end

function TreeLSTMToefl:predict(sents, choices, queries, num_sent, num_answers)
	self.treelstm:training()
    local prediction = {}

    --query
    local emb_queries = {}
    local rep_queries = {}
    for q=1,#queries do
        self.emb:forward(queries[q].sent)
        local emb_query
        if self.cuda then
            emb_query = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
        else
            emb_query = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
        end
        local _, query_vec = self.treelstm:unpack_state(self.treelstm:forward(queries[q].tree.root, emb_query))
        table.insert(emb_queries,emb_query)
        table.insert(rep_queries, query_vec)
    end

    --choices
    local choice_vecs = {}
    local emb_choices = {}
    for c=1,self.num_choices do
        self.emb:forward(choices[c].sent)
        local emb_choice
        if self.cuda then
            emb_choice = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
        else
            emb_choice = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
        end
        table.insert(emb_choices,emb_choice)
        local choice_root = choices[c].tree.root
        local _, choice_vec = self.treelstm:unpack_state(self.treelstm:forward(choice_root, emb_choices[c]))
        table.insert(choice_vecs,choice_vec)
    end

    local emb_sents  = {}
    local rep_sents = {}
    for s=1,num_sent do
        local sent = sents[s].sent
        local tree = sents[s].tree
        self.emb:forward(sent)

        -- forward thru one tree
        local emb_sent
        if self.cuda then
            emb_sent = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
        else
            emb_sent = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
        end
        table.insert(emb_sents,emb_sent)

        self.treelstm:forward(sents[s].tree.root,emb_sent)
        _, rep_sents[s] = self.treelstm:unpack_state(sents[s].tree.root.state)
    end

    local story = {}
    for s=1,num_sent do
        table.insert(story, rep_sents[s])
    end

    for s=1,#queries do
        table.insert(story, rep_queries[q])
    end

    local pred = self.sim_output_module:forward{story,choice_vecs[1],choice_vecs[2],choice_vecs[3],choice_vecs[4]}
	
	prediction = argmax_n(pred,num_answers)

	for s=1,num_sent do
		self.treelstm:clean(sents[s].tree.root)
	end
	
	for q=1,#queries do
		self.treelstm:clean(queries[q].tree.root)
	end

	for c=1,self.num_choices do
		self.treelstm:clean(choices[c].tree.root)
	end

    return prediction
end

function TreeLSTMToefl:predict_dataset(dataset)
    local predictions = {}
    for i = 1, dataset.size do
        xlua.progress(i, dataset.size)
        table.insert(predictions, self:predict(dataset.sents[i], dataset.choices[i], dataset.queries[i], dataset.num_sent[i], #dataset.answers[i]))
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

function TreeLSTMToefl:print_config()
    local num_params = self.params:size(1)
    printf('%-25s = %d\n',   'num params', num_params)
	printf('%-25s = %d\n',   'Number of choices', self.num_choices)
    printf('%-25s = %s\n',   'TreeLSTM structure', self.structure)
    printf('%-25s = %d\n',   'TreeLSTM memory dim', self.mem_dim)
    printf('%-25s = %.2e\n', 'learning rate', self.lr)
    printf('%-25s = %d\n',   'minibatch size', self.batch_size)
    printf('%-25s = %.2e\n', 'regularization strength', self.reg)
    printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
end

function TreeLSTMToefl:save(path)
    local config = {
		num_choices             = self.num_choices,
        mem_dim                 = self.mem_dim,
		lr                      = self.lr,
		emb_lr                  = self.emb_lr,
		batch_size              = self.batch_size,
		reg                     = self.reg,
		structure               = self.structure,
        emb_vecs                = self.emb.weight:float(),
        cuda                    = self.cuda
    }

    torch.save(path, {
        params = self.params,
        config = config,
    })
end

function TreeLSTMToefl.load(path)
    local state = torch.load(path)
    local model = HierAttnModel.TreeLSTMToefl.new(state.config)
    model.params:copy(state.params)
    return model
end
