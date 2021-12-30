
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import RobertaForMaskedLM, RobertaModel, PreTrainedModel
from pytorch_transformers.modeling_bert import BertLayerNorm, gelu
from config.cfg import cfg, path, hyper_roberta
from torch.autograd import Variable
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gpu_id'])
device = torch.device(cfg['device'])

class LMHead(nn.Module):
    def __init__(self):
        super(LMHead, self).__init__()
        self.label_emb = nn.Embedding(cfg['word_size'], hyper_roberta['label_dim'])
        self.label_emb.weight = PromptMask().roberta.embeddings.word_embeddings.weight.clone()
        self.dence = nn.Linear(hyper_roberta['label_dim'], hyper_roberta['label_dim'])
        self.layer_norm = BertLayerNorm(hyper_roberta['label_dim'], eps=1e-5)
        self.classifer = nn.Linear(hyper_roberta['label_dim'], 2)

    def forward(self, input_x):

        gumbel_softmax = F.gumbel_softmax(input_x, hard=True)
        # x = torch.softmax(x, dim=1)
        roberta_emb = self.label_emb.weight

        input_x = torch.matmul(gumbel_softmax, roberta_emb)

        x = self.dence(input_x)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.classifer(x)
        return x


class PromptMask(nn.Module):
    def __init__(self):
        super(PromptMask, self).__init__()
        self.roberta = RobertaForMaskedLM.from_pretrained(path['roberta_path'])

    def forward(self, input_x):
        mask0 = (input_x == 50264)
        mask1 = (input_x != 1).type(torch.long)

        input_x = self.roberta(input_x, attention_mask=mask1)
        x = input_x[0]
        # x = self.lm_head(x)
        x = x[mask0]

        return x






