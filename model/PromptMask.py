
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
        # gumbel_softmax = F.gumbel_softmax(x, hard=True)
        # x = torch.softmax(x, dim=1)
        # roberta_emb = self.roberta_mask.roberta.embeddings.word_embeddings.weight

        # x = torch.matmul(gumbel_softmax, roberta_emb)
        # x = self.lm_head(x)

        return x






