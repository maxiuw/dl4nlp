import torch
from a1_2.A2_skeleton import A2ModelConfig, A2Transformer, generate, load_olmo2_pretrained
from a1_1.A1_skeleton import build_tokenizer, A1Trainer,A1RNNModel, A1RNNModelConfig
train = False
gen = True
use_olmo = False           
checkpoint = 'epoch_transformer'   
olmo_model_name = 'allenai/OLMo-2-0425-1B'
olmo_local_dir = 'olomo/'  # local save dir; downloaded from HF on first run

if use_olmo:
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if os.path.isdir(olmo_local_dir):
        print(f'Loading OLMo-2 from local dir {olmo_local_dir}...')
        tokenizer = AutoTokenizer.from_pretrained(olmo_local_dir)
        model = AutoModelForCausalLM.from_pretrained(olmo_local_dir, torch_dtype=torch.bfloat16)
    else:
        print(f'Downloading OLMo-2 from HF and saving to {olmo_local_dir}...')
        tokenizer = AutoTokenizer.from_pretrained(olmo_model_name)
        model = AutoModelForCausalLM.from_pretrained(olmo_model_name, torch_dtype=torch.bfloat16)
        tokenizer.save_pretrained(olmo_local_dir)
        model.save_pretrained(olmo_local_dir)
        print('Saved.')
    model.embedding = model.model.embed_tokens
else:
    # create necessary objects and train 'train.txt' for trainign and 'val.txt' for validation
    print('Building tokenizer...')
    tokenizer = build_tokenizer('a1_1/train.txt', max_voc_size=50000) #, model_max_length=256)
    config = A2ModelConfig(
        vocab_size=len(tokenizer),
        embedding_size=256,
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=4,
        max_position_embeddings=512,
    )
    model =  A2Transformer.from_pretrained(checkpoint) #(config)

    # if checkpoint:
    #     print(f'Loading model from {checkpoint}...')
    #     model = A1RNNModel.from_pretrained(checkpoint)
    # else:
    #     config = A1RNNModelConfig(
    #         vocab_size=len(tokenizer),
    #         embedding_size=256,
    #         hidden_size=256,
    #         num_hidden_layers=2,
    #     )
    #     model = A1RNNModel(config)
train_dataset = open('a1_1/train.txt', 'r').readlines()
eval_dataset = open('a1_1/val.txt', 'r').readlines()
# use a1 trainier for training
if train and not use_olmo:
    class TrainingArguments:
        def __init__(self, learning_rate=1e-3, num_train_epochs=10, per_device_train_batch_size=16, per_device_eval_batch_size=16, output_dir='output', optim='adamw_torch', eval_strategy='epoch', use_cpu=False):
            self.learning_rate = learning_rate
            self.num_train_epochs = num_train_epochs
            self.per_device_train_batch_size = per_device_train_batch_size
            self.per_device_eval_batch_size = per_device_eval_batch_size
            self.output_dir = output_dir
            self.optim = optim
            self.eval_strategy = eval_strategy
            self.use_cpu = use_cpu  
    args = TrainingArguments()
    print('Training...')
    trainer = A1Trainer(model, args, train_dataset, eval_dataset, tokenizer)
    trainer.train()
    
elif gen:
    print('\n--- Generation examples ---')
    prompts = [
        'In natural language processing, a Transformer',
        'Is Stockholm the capital of Sweden? Answer yes or no. The answer is',
        'Write a Python program that reverses a list.',
    ]
    if use_olmo:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        for prompt in prompts:
            print(f'\nPrompt: {prompt}')
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=1.0, top_k=50)
            print(tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True))
    else:
        for prompt in prompts:
            print(f'\nPrompt: {prompt}')
            print('temp=1.0, topk=None :', generate(model, tokenizer, prompt, max_length=50))
            print('temp=0.5, topk=10   :', generate(model, tokenizer, prompt, max_length=50, temperature=0.5, topk=10))
            print('temp=1.5, topk=50   :', generate(model, tokenizer, prompt, max_length=50, temperature=1.5, topk=50))

