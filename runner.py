import torch
from a1_2.A2_skeleton import A2ModelConfig, A2Transformer, generate, load_olmo2_pretrained
from a1_1.A1_skeleton import build_tokenizer, A1Trainer,A1RNNModel, A1RNNModelConfig, predict_next_word
train = False
gen = True
use_olmo = True   
use_transformer = True  # if True, use A2Transformer; else use A1RNNModel        
checkpoint =  'epoch_transformer'    # or 'epoch_10'
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
    if use_transformer:
        model =  A2Transformer.from_pretrained(checkpoint) #(config)

    else:
        if checkpoint:
            print(f'Loading model from {checkpoint}...')
            model = A1RNNModel.from_pretrained(checkpoint)
        else:
            config = A1RNNModelConfig(
                vocab_size=len(tokenizer),
                embedding_size=256,
                hidden_size=256,
                num_hidden_layers=2,
            )
            model = A1RNNModel(config)
        
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
        # Task 3.3: generate with the pre-trained OLMo-2 model for comparison against our own model's output.
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
            # Task 5.1 (A1) / 3.1 (A2): deterministic next-word prediction (argmax).
            print('next word (argmax) :', predict_next_word(model, tokenizer, prompt))
            # Task 3.2 (A2): sampling-based generation under different temperature/top-k settings.
            print('temp=1.0, topk=None :', generate(model, tokenizer, prompt, max_length=50))
            print('temp=0.5, topk=10   :', generate(model, tokenizer, prompt, max_length=50, temperature=0.5, topk=10))
            print('temp=1.5, topk=50   :', generate(model, tokenizer, prompt, max_length=50, temperature=1.5, topk=50))

# generation examples for a1_1
'''
--- Generation examples ---

Prompt: In natural language processing, a Transformer
temp=1.0, topk=None : is not either an indication of speech , in contrast to an audience or function . it is possible to reconstruct all class software throughout history . though , on such rogers as `` j '' , which means `` y '' or `` indelible '' , the bias process
temp=0.5, topk=10   : can be used to create a <UNK> function ( which is a <UNK> ) that is the same as the `` <UNK> '' ( <UNK> ) , which is the same as the `` <UNK> '' . it is not a common sense of the language and the language of
temp=1.5, topk=50   : can handle no <UNK> ( though <UNK> data structures such as a lack of an optical function of `` a module ; '' `` g '' - `` g '' and in addition to that `` f '' ) . however , as well as : <EOS>

Prompt: Is Stockholm the capital of Sweden? Answer yes or no. The answer is
temp=1.0, topk=None : impossible following by germany . nevertheless , each hundred schedules , gauss 's opinion meteorites , came through it from allah , including ludwig von <UNK> , who , in turn he observed , `` testimony from it was a few universities with different parts of the civilization , the
temp=0.5, topk=10   : that the `` <UNK> '' ( `` <UNK> '' ) is the <UNK> . the two countries have a total of <UNK> <UNK> , the <UNK> and the <UNK> ( <UNK> ) . <EOS>
temp=1.5, topk=50   : held to do . the question that an equal number of historical names for the european union or europe , however , be said , and that is not a group of member states have come to fruition . but rather the <UNK> can now become on all matters outside

Prompt: Write a Python program that reverses a list.
temp=1.0, topk=None : scores such as structured functions can be rewritten as out of computer that way out of data for funding , reading , changing clicks or small entries . the hardware is given only in <UNK> support . the first 160 modern referendum , based on a method that expanded permanent
temp=0.5, topk=10   : <EOS>
temp=1.5, topk=50   : as for reasons , <UNK> <UNK> is the only means with a <UNK> and `` <UNK> character '' as , such as some of the same parts of its <UNK> usage in the film `` i do n't always be understood before your audience of their own and the voice

'''

# generation examples a1_2
'''
--- Generation examples ---

Prompt: In natural language processing, a Transformer
temp=1.0, topk=None : language may cell and trouble with a data given such a context-free language . <EOS>
temp=0.5, topk=10   : language was introduced by the <UNK> <UNK> ( <UNK> ) , which was a common language , but the <UNK> languages were not known . <EOS>
temp=1.5, topk=50   : compiler was designed to convert its languages in , and as did after the application was released in 1975 by using the group 's name was introduced : the first group used that computer access was the case with use in what was known to describe the language of their

Prompt: Is Stockholm the capital of Sweden? Answer yes or no. The answer is
temp=1.0, topk=None : forced to marry and of the roman catholic missionaries of alexandria ( brachiopods , stucco ) and fort pasha , victorious catholic density , pentecostals ( up to harassing ) , and the new french , over 200 years of age . it is also known greek as easter assyrians
temp=0.5, topk=10   : called `` <UNK> '' , which is a `` <UNK> '' , and `` <UNK> '' , and `` <UNK> '' . it is a `` <UNK> '' , meaning `` <UNK> '' , and `` <UNK> '' , and `` <UNK> '' is a `` <UNK> '' ( <UNK>
temp=1.5, topk=50   : set towards the <UNK> province , a french and republic in which it is connected between the capital state of the united kingdom by a small force over an area of the city will be given as a new line , from that direction in which it establishes power .

Prompt: Write a Python program that reverses a list.
temp=1.0, topk=None : heap display format uses `` an aerobic mechanical character ( multiple memory code ) '' as to <UNK> after applying the domain of a video vowel , `` monitor '' and `` <UNK> '' in turn is different and `` <UNK> `` g '' , creating the promised derivatives of
temp=0.5, topk=10   : <EOS>
temp=1.5, topk=50   : we can do a way . i is working with your machine interface for it , such that each program will have all <UNK> our customers . for the three decades will take some of them as `` one or another will give forth from the other ’ '' (
'''
# for olomo 

'''
--- Generation examples ---

Prompt: In natural language processing, a Transformer
-based attention mechanism has been utilized for semantic segmentation [1]. This structure has been utilized for the unsupervised segmentation of videos [2]. In the aforementioned works, the attention mechanism is applied for predicting the next action frames, which does not suit for

Prompt: Is Stockholm the capital of Sweden? Answer yes or no. The answer is
 no. It should be noted, however, that capital of Sweden has changed considerably over time. Stockholm is not the oldest capital of the country. The history of Stockholm stretches back over the period 700 to approximately 1150 AD. Thereafter the

Prompt: Write a Python program that reverses a list.
 Read the given list and the reversed list. Your program should print both the lists. The given list should not be reversed.

For instance, if the given list is ['a', 'b', 'c', 'd', 'e'] and the
'''


