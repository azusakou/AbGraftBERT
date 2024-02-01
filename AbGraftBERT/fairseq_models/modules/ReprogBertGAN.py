from transformers import EsmConfig, EsmForMaskedLM
from transformers.models.esm.modeling_esm import EsmPreTrainedModel, EsmEmbeddings, EsmModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertOnlyMLMHead

from torch import nn
import torch
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import CrossEntropyLoss


class EsmConfigProtein(EsmConfig):
    def __init__(self, vocab_size_protein=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size_protein = vocab_size_protein

class AttentionModule(torch.nn.Module):
    def __init__(self, i_feature_size, o_feature_size, output_feature_size):
        super(AttentionModule, self).__init__()
        self.query_transform = torch.nn.Linear(i_feature_size, o_feature_size, bias=False)
        self.key_transform = torch.nn.Linear(i_feature_size, o_feature_size, bias=False)
        self.value_transform = torch.nn.Linear(i_feature_size, o_feature_size, bias=False)

        # Additional linear layer to transform the output feature size
        self.output_transform = torch.nn.Linear(o_feature_size, output_feature_size)

    def forward(self, query, key, value, mask=None):
        query = self.query_transform(query)
        key = self.key_transform(key)
        value = self.value_transform(value)

        scores = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(weights, value)

        # Transforming the output to the desired feature size
        transformed_output = self.output_transform(output)
        return transformed_output

class ModifiedBertLMPredictionHead(BertLMPredictionHead):
    def __init__(self, config):
        super().__init__(config)
        self.gamma = nn.Linear(config.vocab_size, config.vocab_size_protein, bias=False)
        self.gammabias = nn.Parameter(torch.zeros(config.vocab_size_protein))
        self.gamma.bias = self.gammabias

        # add layer to generate the representation of angles and distance

        #self.struc = AttentionModule(config.vocab_size, config.vocab_size_protein, 3)
        #self.fc = nn.Linear(config.vocab_size_protein+3, config.vocab_size_protein, bias=False)

    def forward(self, hidden_states):
        hidden_states = super().forward(hidden_states)
        hidden_states = self.gamma(hidden_states)
        #hidden_struc = torch.sigmoid(self.struc(hidden_states, hidden_states, hidden_states))

        #hidden_states = torch.cat((hidden_states, hidden_struc), dim=-1)
        #hidden_states = self.fc(hidden_states)
        return hidden_states, hidden_states #, hidden_struc


class ModifiedBertOnlyMLMHead(BertOnlyMLMHead):
    def __init__(self, config):
        super().__init__(config)
        self.predictions = ModifiedBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores, hidden_struc = self.predictions(sequence_output)
        return prediction_scores, hidden_struc


class ModifiedBertPreTrainedModel(EsmPreTrainedModel):
    config_class = EsmConfigProtein


class ModifiedBertEmbeddings(EsmEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super().__init__(config)
        self.conf = config

        # CHANGED: Added theta projection matrix of input protein domain to english domain
        self.theta = nn.Embedding(config.vocab_size_protein+1, config.vocab_size, padding_idx=config.pad_token_id_prot)


        #print(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
    def forward(
        self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
	
        if inputs_embeds is None:
            inputs_embeds = self.theta(input_ids)#;print(inputs_embeds.size(),self.word_embeddings.weight.size())
            inputs_embeds = torch.matmul(inputs_embeds, self.word_embeddings.weight)


        # Note that if we want to support ESM-1 (not 1b!) in future then we need to support an
        # embedding_scale factor here.
        embeddings = inputs_embeds

        # Matt: ESM has the option to handle masking in MLM in a slightly unusual way. If the token_dropout
        # flag is False then it is handled in the same was as BERT/RoBERTa. If it is set to True, however,
        # masked tokens are treated as if they were selected for input dropout and zeroed out.
        # This "mask-dropout" is compensated for when masked tokens are not present, by scaling embeddings by
        # a factor of (fraction of unmasked tokens during training) / (fraction of unmasked tokens in sample).
        # This is analogous to the way that dropout layers scale down outputs during evaluation when not
        # actually dropping out values (or, equivalently, scale up their un-dropped outputs in training).
        if self.token_dropout:
            embeddings = embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8  # Hardcoded as the ratio used in all ESM model training runs
            src_lengths = attention_mask.sum(-1)
            mask_ratio_observed = (input_ids == self.mask_token_id).sum(-1).float() / src_lengths
            embeddings = (embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]).to(
                embeddings.dtype
            )

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)
        # Matt: I think this line was copied incorrectly from BERT, disabling it for now.
        # embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
    def create_position_ids_from_input_ids(self, input_ids, padding_idx, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx


class ModifiedBertModel(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.embeddings = ModifiedBertEmbeddings(config)


class EsmForMaskedLMProt(ModifiedBertPreTrainedModel, EsmForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        #self.gamma = nn.Linear(768, 1280)#, bias=False)
        #self.gammabias = nn.Parameter(torch.zeros(config.vocab_size_protein))
        #self.gamma.bias = self.gammabias
        # CHANGED: BertModel -> ModifiedBertModel
        self.esm = ModifiedBertModel(config, add_pooling_layer=False)
        #self.theta = nn.Linear(1280, 768)

        # CHANGED: BertOnlyMLMHead -> ModifiedBertOnlyMLMHead
        #self.cls = ModifiedBertOnlyMLMHead(config)#;print (config)
	
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            #token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            past_key_value=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        return sequence_output, outputs
'''        prediction_scores, hidden_struc = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # CHANGED: self.config.vocab_size -> self.config.vocab_size_protein
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size_protein), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), hidden_struc'''
