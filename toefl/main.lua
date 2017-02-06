--[[

  Training script for QA on the TOEFL listening comprehension test dataset

--]]

require('..')

function accuracy(pred, gold)
	local correct = 0
	for i=1,#gold do
		for j=1,#gold[i] do
			if gold[i][j] == pred[i][j] then
				correct = correct + 1
				break
			end
		end
	end
  return correct / #gold
end

-- read command line arguments
local args = lapp [[
Training script for QA on the TOEFL listening comprehension
    -m,--model  (default ham)           Model architecture: [ham, lstm, bilstm, treelstm, memn2n]
    -t,--task   (default manual)        [manual, ASR]
    -d,--dim    (default 75)            Sentence representation model memory dimension
    -i,--internal (default 75)          MemN2N internal dimension
    -e,--epochs (default 10)            Number of training epochs
    -h,--hops (default 1)               Number of hops in MemN2N
    -l,--lr (default 0.002)             Learning rate
    -s,--similarity (default cosine)    Similarity for calculating attention
    -n,--attnorm (default sharp)        Normalization method for attention
    -v,--level (default phrase)         [phrase,sentence]
    -p,--prune (default 1)              Pruning rate
    -o,--dropout (default 0.0)          Dropout rate
    -y,--layers (default 1)             LSTM/BiLSTM layers
    -z,--optimizer (default adagrad)    Optimizer: [adagrad, adam]
    -c,--cuda                           Cuda
]]

if args.cuda then
    require('cutorch')
    require('cunn')
end

printf('%-25s = %s\n', 'task',args.task)
printf('%-25s = %s\n', 'level',args.level)
printf('%-25s = %.2f\n', 'pruning rate',args.prune)

local model_name, model_class, model_structure, mem_size
if args.model == 'ham' then
    model_name = 'Hierarchical Attention Model'
    model_class = HierAttnModel.HierAttnModelToefl
elseif args.model == 'lstm' then
    model_name = 'Unidirectional LSTM'
    model_class = HierAttnModel.LSTMToefl
elseif args.model == 'bilstm' then
    model_name = 'Bidirectional LSTM'
    model_class = HierAttnModel.LSTMToefl
elseif args.model == 'treelstm' then
    model_name = 'Tree-structured LSTM'
    model_class = HierAttnModel.TreeLSTMToefl
elseif args.model == 'memn2n' then
	model_name = 'End-to-end Memory Network'
	model_class = HierAttnModel.MemN2NToefl
end

if args.level == 'phrase' then
	mem_size = 995
else
	mem_size = 78
end
model_structure = args.model
header(model_name .. ' for TOEFL Question Answering')

-- directory containing dataset files
local data_dir = string.format('data/toefl/%s_trans/',args.task)

-- load vocab
local vocab = HierAttnModel.Vocab('data/toefl/manual_trans/vocab-cased.txt') -- whole vocab

-- load embeddings
print('loading word embeddings')
local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = HierAttnModel.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
	local w = string.gsub(vocab:token(i), '\\', '')  --remove escape characters
	if emb_vocab:contains(w) then
		vecs[i] = emb_vecs[emb_vocab:index(w)]
	else
		num_unk = num_unk + 1
		vecs[i]:uniform(-0.05, 0.05)
	end
end

print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()

-- load datasets
local num_choices = 4
print('loading datasets')
local train_dir = 'data/toefl/manual_trans/train/'
local dev_dir = 'data/toefl/manual_trans/dev/'
local test_manual_dir = 'data/toefl/manual_trans/test/'
local test_dir = data_dir .. 'test/'
local train_dataset = HierAttnModel.read_toefl_dataset(train_dir, vocab, num_choices, args.prune, args.cuda)
local dev_dataset = HierAttnModel.read_toefl_dataset(dev_dir, vocab, num_choices, args.prune, args.cuda)
local test_dataset = HierAttnModel.read_toefl_dataset(test_dir, vocab, num_choices, args.prune, args.cuda)
local test_manual_dataset = HierAttnModel.read_toefl_dataset(test_manual_dir, vocab, num_choices, args.prune, args.cuda)

printf('num train = %d\n', train_dataset.size)
printf('num dev   = %d\n', dev_dataset.size)
printf('num test  = %d\n', test_dataset.size)

-- initialize model
local model = model_class{
	level           = args.level,
	lr              = args.lr,
	mem_dim         = args.dim,
	structure       = model_structure,
    optimizer       = args.optimizer,
	-- memory mechanism
    sim             = args.similarity,
    att_norm        = args.attnorm,
	internal_dim    = args.internal,
	mem_size     = mem_size,
	hops            = args.hops,
	--word embedding
    emb_vecs        = vecs,
    -- dropout rate
    dropout         = args.dropout,
    --for lstm/bilstm
    num_layers      = args.layers,
    -- use cuda
    cuda            = args.cuda
}

-- number of epochs to train
local num_epochs = args.epochs

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model
local test_scores ={}
header('Training model')
for i = 1, num_epochs do
    local start = sys.clock()
    printf('-- epoch %d\n', i)
    model:train(train_dataset)
    printf('-- finished epoch in %.2fs\n', sys.clock() - start)

    -- uncomment to print training accuracy
    --[[
	local train_predictions = model:predict_dataset(train_dataset)
	local train_score = accuracy(train_predictions, train_dataset.answers)
	printf('-- train score: %.4f\n', train_score)
    --]]

	--
	local dev_predictions = model:predict_dataset(dev_dataset)
	local dev_score = accuracy(dev_predictions, dev_dataset.answers)
	printf('-- dev score: %.4f\n', dev_score)

    if dev_score > best_dev_score then
        best_dev_score = dev_score
        best_dev_model = model_class{
            level           = args.level,
            lr              = args.lr,
            mem_dim         = args.dim,
            structure       = model_structure,
            optimizer       = args.optimizer,
            -- memory mechanism
            sim             = args.similarity,
            att_norm        = args.attnorm,
            internal_dim    = args.internal,
            mem_size     = mem_size,
            hops            = args.hops,
            --word embedding
            emb_vecs        = vecs,
            -- dropout rate
            dropout         = args.dropout,
            --for lstm/bilstm
            num_layers      = args.layers,
            -- use cuda
            cuda            = args.cuda
        }
        best_dev_model.params:copy(model.params)
        best_dev_model.emb.weight:copy(model.emb.weight)
    end
end

printf('finished training in %.2fs\n', sys.clock() - train_start)
header('Evaluating on test set')
printf('-- using model with dev score = %.4f\n', best_dev_score)
local test_predictions = best_dev_model:predict_dataset(test_dataset)
printf('-- test score: %.4f\n', accuracy(test_predictions, test_dataset.answers))

-- create models directories if necessary
local models_dir
if args.model == 'ham' and args.level == 'phrase' then
    models_dir = HierAttnModel.models_dir .. '/phrase_level'
elseif args.model == 'ham' then
    models_dir = HierAttnModel.models_dir .. '/sentence_level'
else
    models_dir = HierAttnModel.models_dir .. '/' .. args.model
end
if lfs.attributes(models_dir) == nil then
    lfs.mkdir(models_dir)
end

-- get paths
local file_idx = 1
local model_save_path
while true do
    if args.model == 'ham' then
        model_save_path = string.format(
            models_dir .. '/%s.prune%.1f.dim%d.int%d.hops%d.dropout%.1f.%s.%d.th', args.task, args.prune, args.dim, args.internal, args.hops, args.dropout, args.attnorm, file_idx)
    elseif args.model == 'lstm' or args.model == 'bilstm' then
        model_save_path = string.format(
            models_dir .. '/%s.prune%.1f.dim%d.layers%d.%d.th', args.task, args.prune, args.dim, args.layers, file_idx)
    elseif args.model == 'treelstm' then
        model_save_path = string.format(
            models_dir .. '/%s.prune%.1f.dim%d.%d.th', args.task, args.prune, args.dim, file_idx)
    else
        model_save_path = string.format(
            models_dir .. '/%s.prune%.1f.dim%d.hops%d.%d.th', args.task, args.prune, args.dim, args.hops, file_idx)
    end
    if lfs.attributes(model_save_path) == nil then
        break
    end
    file_idx = file_idx + 1
end
-- write model to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)
