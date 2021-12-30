
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
        self.dence = nn.Linear(hyper_roberta['label_dim'], hyper_roberta['label_dim'])

        self.layer_norm = BertLayerNorm(hyper_roberta['label_dim'], eps=1e-5)
        self.classifer = nn.Linear(hyper_roberta['label_dim'], cfg['word_size'], bias=False)
        # self.classifer.weight = PromptMask().roberta.embeddings.word_embeddings.weight
        self.bias = nn.Parameter(torch.zeros(cfg['word_size']))

    def forward(self, input_x, mask0):
        # x = input_x
        # x = self.dropout(x)
        # x = self.dence_word(x)
        # x = gelu(x)
        #
        # x = F.gumbel_softmax(x, hard=True)
        # # # x = torch.softmax(input_x, dim=1)
        # roberta_emb = self.label_emb.weight
        #
        # input_x = torch.matmul(x, roberta_emb)

        x = self.dence(input_x)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.classifer(x) + self.bias
        return x[mask0]


class PromptMask(nn.Module):
    def __init__(self):
        super(PromptMask, self).__init__()
        self.roberta = RobertaForMaskedLM.from_pretrained(path['roberta_path'])

        self.classifer = nn.Linear(hyper_roberta['label_dim'], 2)
        # self.emb = self.roberta.roberta.embeddings.word_embeddings
        # self.classifer.weight = nn.Parameter(torch.cat([self.emb.weight[6587, :].unsqueeze(dim=0), self.emb.weight[372, :].unsqueeze(dim=0)], dim=0).clone())
        self.roberta._tie_or_clone_weights(self.classifer, self.roberta.roberta.embeddings.word_embeddings)


    def forward(self, input_x):
        mask0 = (input_x == 50264)
        mask1 = (input_x != 1).type(torch.long)

        input_x = self.roberta.roberta(input_x, attention_mask=mask1)
        x = input_x[0]

        x = self.roberta.lm_head.dense(x)
        x = gelu(x)
        x = self.roberta.lm_head.layer_norm(x)

        x = x[mask0]
        x = self.classifer(x)


        # x = self.lm_head(x, mask0)
        return x






