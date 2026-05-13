
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from a1_1.A1_skeleton import build_tokenizer, A1Trainer
from transformers.modeling_outputs import CausalLMOutput

class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(
        self,
        vocab_size=None,
        embedding_size=None,
        hidden_size=None,
        intermediate_size=None,
        num_attention_heads=12,
        num_hidden_layers=12,
        rope_theta=10000.0,
        hidden_act='silu',
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size if intermediate_size is not None else hidden_size * 4
        self.num_hidden_layers = num_hidden_layers



class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        # TODO: initalize components here

        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.fn = nn.SiLU()
    def forward(self, hidden_states):
        x1 = self.w1(hidden_states)
        x2 = self.w2(hidden_states)
        x = self.fn(x1) * x2
        x = self.w3(x)
        return x         

# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config):
        super().__init__()
        # TODO: Use config.rms_norm_eps
        # TODO: initalize weights here
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.rms_norm_eps
        
    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states * self.weight


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        # TODO: set up W_q, W_k, W_v, W_o here
        # TODO: set up normalizers here
        self.W_Q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_K = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_V = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_O = nn.Linear(config.hidden_size, config.hidden_size, bias=False) # final proj
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, hidden_states, rope_rotations):
        batch_size, seq_length, hidden_size = hidden_states.size()
        q = self.W_Q(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, rope_rotations)
        # scale dot att from torch
        # att_out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(q.device) * float('-inf'))
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.masked_fill(torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(attn_weights.device), float('-inf'))
        attn_probs = self.softmax(attn_weights)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        output = self.W_O(attn_output)
        return output


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        # TODO: set up attention, MLP, and normalizers here.
        self.attention = A2Attention(config)
        self.mlp = A2MLP(config)
        self.norm1 = A2RMSNorm(config)
        self.norm2 = A2RMSNorm(config)
        
    def forward(self, hidden_states, rope_rotations):
        hidden_states = hidden_states + self.attention(self.norm1(hidden_states), rope_rotations)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        # TODO: Set up the other components here.
        # TODO: put all transformer decoder layers in a ModuleList.
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder_layers = nn.ModuleList([A2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = A2RMSNorm(config)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
        # This line should be called after you have set up all components.
        self.post_init()


    def forward(self, input_ids, labels=None):
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers

        # TODO: Call embedding, transformer decoder layers, last normalizer, and unembedding.
        # TODO: Compute the loss as in Assignment 1 if labels is not None.
        hidden_states = self.embedding(input_ids)
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states, rope_rotations)
        hidden_states = self.norm(hidden_states)
        logits = self.unembedding(hidden_states)    
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)


#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin

if __name__ == "__main__":
    # You can use this space for testing your code.
    # use trainig scheme from ../a1_1/A1_skeleton.pyif __name__ == '__main__':
    # create necessary objects and train 'train.txt' for trainign and 'val.txt' for validation
    print('Building tokenizer...')
    tokenizer = build_tokenizer('../a1_1/train.txt', max_voc_size=50000) #, model_max_length=256)
    config = A2ModelConfig(vocab_size=len(tokenizer), embedding_size=256, hidden_size=768)
    model = A2Transformer(config)
    train_dataset = open('../a1_1/train.txt', 'r').readlines()
    eval_dataset = open('../a1_1/val.txt', 'r').readlines()
    # use a1 trainier for training
    class TrainingArguments:
        def __init__(self, learning_rate=1e-3, num_train_epochs=10, per_device_train_batch_size=64, per_device_eval_batch_size=64, output_dir='output', optim='adamw_torch', eval_strategy='epoch', use_cpu=False):
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
    
