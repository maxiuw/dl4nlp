from a1_2.A2_skeleton import A2ModelConfig, A2Transformer
from a1_1.A1_skeleton import build_tokenizer, A1Trainer,A1RNNModel, A1RNNModelConfig


# create necessary objects and train 'train.txt' for trainign and 'val.txt' for validation
print('Building tokenizer...')
tokenizer = build_tokenizer('a1_1/train.txt', max_voc_size=50000) #, model_max_length=256)
# config = A2ModelConfig(
#     vocab_size=len(tokenizer),
#     embedding_size=256,
#     hidden_size=256,
#     num_attention_heads=8,
#     num_hidden_layers=4,
#     max_position_embeddings=512,
# )
# model =  A2Transformer(config)
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
# calculate the perplexity on the validation set and print it out

