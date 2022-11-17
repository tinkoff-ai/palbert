# coding=utf-8
# Copyright 2022 Tinkoff, Google AI, Google Brain and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import AlbertConfig, AlbertPreTrainedModel
from transformers.models.albert.modeling_albert import (AlbertEmbeddings,
                                                        AlbertLayerGroup)


class AlbertTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(
            config.embedding_size, config.hidden_size
        )
        self.albert_layer_groups = nn.ModuleList(
            [AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)]
        )

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(
                self.config.num_hidden_layers / self.config.num_hidden_groups
            )

            # Index of the hidden group
            group_idx = int(
                i / (self.config.num_hidden_layers / self.config.num_hidden_groups)
            )

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[
                    group_idx * layers_per_group : (group_idx + 1) * layers_per_group
                ],
            )
            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

    def adaptive_forward(
        self,
        hidden_states,
        current_layer,
        attention_mask=None,
        head_mask=None,
        use_prev_hiddens=False,
        prev_state_influence=None,
    ):
        if current_layer == 0:
            hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        else:
            hidden_states = hidden_states[0]

        layers_per_group = int(
            self.config.num_hidden_layers / self.config.num_hidden_groups
        )

        # Index of the hidden group
        group_idx = int(
            current_layer
            / (self.config.num_hidden_layers / self.config.num_hidden_groups)
        )

        # Index of the layer inside the group
        layer_idx = int(current_layer - group_idx * layers_per_group)

        layer_group_output = self.albert_layer_groups[group_idx](
            hidden_states,
            attention_mask,
            head_mask[
                group_idx * layers_per_group : (group_idx + 1) * layers_per_group
            ],
        )
        hidden_states = layer_group_output[0]

        return (hidden_states,)


class AlbertPABEEModel(AlbertPreTrainedModel):
    config_class = AlbertConfig
    base_model_prefix = "albert"

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.init_weights()
        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0

        self.regression_threshold = 0.2

    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold

    def set_patience(self, patience):
        self.patience = patience

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f"*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up =  {self.config.num_hidden_layers / avg_inf_layers:.2f} ***"
        print(message)
        return avg_inf_layers

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        ALBERT has a different architecture in that its layers are shared across groups, which then has inner groups.
        If an ALBERT model has 12 hidden layers and 2 hidden groups, with two inner groups, there
        is a total of 4 different layers.
        These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
        while [2,3] correspond to the two inner groups of the second hidden layer.
        Any layer with in index other than [0,1,2,3] will result in an error.
        See base class PreTrainedModel for more information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[
                inner_group_idx
            ].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_dropout=None,
        output_layers=None,
        regression=False,
    ):
        r"""
        Return:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction (classification)
                objective during pre-training.
                This output is usually *not* a good summary
                of the semantic content of the input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        Example::
            from transformers import AlbertModel, AlbertTokenizer
            import torch
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            model = AlbertModel.from_pretrained('albert-base-v2')
            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids)
            last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = embedding_output

        if self.training:
            res = []
            for i in range(self.config.num_hidden_layers):
                encoder_outputs = self.encoder.adaptive_forward(
                    encoder_outputs,
                    current_layer=i,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                )

                pooled_output = self.pooler_activation(
                    self.pooler(encoder_outputs[0][:, 0])
                )
                logits = output_layers[i](output_dropout(pooled_output))
                res.append(logits)
        elif self.patience == 0:  # Use all layers for inference
            encoder_outputs = self.encoder(
                encoder_outputs, extended_attention_mask, head_mask=head_mask
            )
            pooled_output = self.pooler_activation(
                self.pooler(encoder_outputs[0][:, 0])
            )
            res = [output_layers[self.config.num_hidden_layers - 1](pooled_output)]
        else:
            patient_counter = 0
            patient_result = None
            calculated_layer_num = 0
            for i in range(self.config.num_hidden_layers):
                calculated_layer_num += 1
                encoder_outputs = self.encoder.adaptive_forward(
                    encoder_outputs,
                    current_layer=i,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                )

                pooled_output = self.pooler_activation(
                    self.pooler(encoder_outputs[0][:, 0])
                )
                logits = output_layers[i](pooled_output)
                if regression:
                    labels = logits.detach()
                    if patient_result is not None:
                        patient_labels = patient_result.detach()
                    if (patient_result is not None) and torch.abs(
                        patient_result - labels
                    ) < self.regression_threshold:
                        patient_counter += 1
                    else:
                        patient_counter = 0
                else:
                    labels = logits.detach().argmax(dim=1)
                    if patient_result is not None:
                        patient_labels = patient_result.detach().argmax(dim=1)
                    if (patient_result is not None) and torch.all(
                        labels.eq(patient_labels)
                    ):
                        patient_counter += 1
                    else:
                        patient_counter = 0

                patient_result = logits
                if patient_counter == self.patience:
                    break
            res = [patient_result]
            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1
        return res


class AlbertPABEEForSequenceClassification(
    AlbertPreTrainedModel
):  # from https://github.com/JetRunner/PABEE/blob/master/pabee/modeling_albert.py
    # coding=utf-8
    # Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertPABEEModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifiers = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, self.config.num_labels)
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.init_weights()

    def set_patience(self, patience):
        self.albert.set_patience(patience)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in ``[0, ..., config.num_labels - 1]``.
                If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
                If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
            loss: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
                Classification (or regression if config.num_labels==1) loss.
            logits ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
            Examples::
                from transformers import AlbertTokenizer, AlbertForSequenceClassification
                import torch
                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')
                input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
                labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
                outputs = model(input_ids, labels=labels)
                loss, logits = outputs[:2]
        """

        logits = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_dropout=self.dropout,
            output_layers=self.classifiers,
            regression=self.num_labels == 1,
        )

        outputs = {"logits": logits[-1]}

        if labels is not None:
            total_loss = None
            total_weights = 0
            for ix, logits_item in enumerate(logits):
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits_item.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        logits_item.view(-1, self.num_labels), labels.view(-1)
                    )
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss * (ix + 1)
                total_weights += ix + 1
            outputs["loss"] = total_loss / total_weights

        return outputs  # (loss), logits, (hidden_states), (attentions)


class PAlbertModel(AlbertPreTrainedModel):
    config_class = AlbertConfig
    base_model_prefix = "albert"

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.init_weights()
        self.epsilon = 0.05
        self.threshold = 0.5
        self.inference_instances_num = 0
        self.inference_layers_num = 0

        self.layer_positional_embeddings = nn.Embedding(
            config.num_hidden_layers, config.hidden_size
        )

        self.regression_threshold = 0

        self.pooler_for_task = True

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def init_layer_pos_embeddings(self):
        self.layer_positional_embeddings.weight.data.copy_(
            self.get_embedding(self.config.num_hidden_layers, self.config.hidden_size)
            / 100  # mean absolute value
            # ~ 0.02
        )

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f"*** Threshold = {self.threshold} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up = {self.config.num_hidden_layers/avg_inf_layers:.2f}x ***"
        print(message)
        return avg_inf_layers

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def disable_pooler_for_task(self):
        self.pooler_for_task = False

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        ALBERT has a different architecture in that its layers are shared across groups, which then has inner groups.
        If an ALBERT model has 12 hidden layers and 2 hidden groups, with two inner groups, there
        is a total of 4 different layers.
        These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
        while [2,3] correspond to the two inner groups of the second hidden layer.
        Any layer with in index other than [0,1,2,3] will result in an error.
        See base class PreTrainedModel for more information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[
                inner_group_idx
            ].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_dropout=None,
        output_layers=None,
        use_prev_hiddens=True,
        use_layer_pos_embeddings=True,
        prev_state_influence="cat",
        detach_lambda: bool = False,
        exit_criteria: str = "threshold",
    ):
        r"""
        WARNING! Inference works only with batch_size=1
        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to float if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = embedding_output
        res = defaultdict(lambda: [])
        prev_state = None
        acc_logits = 0
        if self.training:
            current_cdf = torch.zeros(input_ids.size(0), device=input_ids.device)
            current_condition_prob = torch.ones_like(
                current_cdf, device=input_ids.device
            )
            for current_layer_idx in range(self.config.num_hidden_layers):
                encoder_outputs = self.encoder.adaptive_forward(
                    encoder_outputs,
                    current_layer=current_layer_idx,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    use_prev_hiddens=use_prev_hiddens,
                    prev_state_influence=prev_state_influence,
                )

                pooled_output = self.pooler_activation(
                    self.pooler(encoder_outputs[0][:, 0])
                )
                pos_pooled_output, prev_state = self.preprocess_poller_output(
                    current_layer_idx,
                    detach_lambda,
                    output_layers,
                    pooled_output,
                    prev_state,
                    prev_state_influence,
                    use_layer_pos_embeddings,
                    use_prev_hiddens,
                )

                if self.pooler_for_task:
                    logits = output_layers.cls_head(output_dropout(pooled_output))
                else:
                    logits = output_layers.cls_head(encoder_outputs[0])

                if output_layers.exit_head.is_rnn:
                    exit_logits, prev_state = output_layers.exit_head(
                        output_dropout(pos_pooled_output), prev_state
                    )
                else:
                    exit_logits = output_layers.exit_head(
                        output_dropout(pos_pooled_output)
                    )
                exit_probs = torch.sigmoid(exit_logits).view(-1)
                current_pdf = (
                    exit_probs * current_condition_prob
                )  # geometric_like distribution
                current_condition_prob = (1 - exit_probs) * current_condition_prob
                current_pdf = torch.masked_fill(
                    current_pdf,
                    current_cdf > 1 - self.epsilon,
                    self.epsilon / self.config.num_hidden_layers,
                )
                current_cdf = current_cdf + current_pdf
                res["logits"].append(logits)
                res["exit_pdf"].append(current_pdf)
                if torch.all(current_cdf > 1 - self.epsilon):
                    for current_layer_idx in range(
                        self.config.num_hidden_layers - current_layer_idx - 1
                    ):
                        current_pdf = (
                            torch.zeros_like(current_pdf)
                            + self.epsilon / self.config.num_hidden_layers
                        )
                        res["exit_pdf"].append(current_pdf)
                    break
        elif self.threshold == 0:  # Use all layers for inference
            encoder_outputs = self.encoder(
                encoder_outputs, extended_attention_mask, head_mask=head_mask
            )
            pooled_output = self.pooler_activation(
                self.pooler(encoder_outputs[0][:, 0])
            )
            res = [output_layers[self.config.num_hidden_layers - 1](pooled_output)]
        else:
            current_cdf = 0
            conditioned_prob = 1
            calculated_layer_num = 0
            for current_layer_idx in range(self.config.num_hidden_layers):
                calculated_layer_num += 1
                encoder_outputs = self.encoder.adaptive_forward(
                    encoder_outputs,
                    current_layer=current_layer_idx,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                )

                pooled_output = self.pooler_activation(
                    self.pooler(encoder_outputs[0][:, 0])
                )
                pos_pooled_output, prev_state = self.preprocess_poller_output(
                    current_layer_idx,
                    detach_lambda,
                    output_layers,
                    pooled_output,
                    prev_state,
                    prev_state_influence,
                    use_layer_pos_embeddings,
                    use_prev_hiddens,
                )
                if output_layers.exit_head.is_rnn:
                    exit_logits, prev_state = output_layers.exit_head(
                        output_dropout(pos_pooled_output), prev_state
                    )
                else:
                    exit_logits = output_layers.exit_head(
                        output_dropout(pos_pooled_output)
                    )
                lambdas = torch.sigmoid(exit_logits)
                current_pdf = lambdas.item() * conditioned_prob
                conditioned_prob *= 1 - lambdas.item()
                current_cdf += current_pdf
                if current_cdf > self.threshold and exit_criteria == "threshold":
                    logits = output_layers.cls_head(pooled_output)
                    res["logits"] = [logits]
                    break
                elif exit_criteria == "sample":
                    exit_flag = torch.bernoulli(lambdas).item()
                    if exit_flag > 0:
                        logits = output_layers.cls_head(pooled_output)
                        res["logits"] = [logits]
                        break

                if exit_criteria == "expectation":
                    pdf = (
                        (1 - current_pdf)
                        if current_layer_idx + 1 == self.config.num_hidden_layers
                        else current_pdf
                    )
                    if self.config.num_labels == 1:
                        # is_regression
                        acc_logits += pdf * output_layers.cls_head(pooled_output)
                    else:
                        acc_logits += pdf * F.softmax(
                            output_layers.cls_head(pooled_output), -1
                        )
                calculated_layer_num = current_layer_idx + 1
                if current_layer_idx + 1 == self.config.num_hidden_layers:
                    logits = output_layers.cls_head(pooled_output)
                    if exit_criteria == "expectation":
                        if not self.config.num_labels == 1:
                            # classification
                            acc_logits = torch.log(acc_logits)
                        res["logits"] = [acc_logits]
                    else:
                        res["logits"] = [logits]
            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1
        return res

    def preprocess_poller_output(
        self,
        current_layer_idx,
        detach_lambda,
        output_layers,
        pooled_output,
        prev_state,
        prev_state_influence,
        use_layer_pos_embeddings,
        use_prev_hiddens,
    ):
        if use_layer_pos_embeddings:
            pos_pooled_output = self.layer_positional_embeddings_forward(
                current_layer_idx=current_layer_idx,
                pooled_output=pooled_output,
                detach_lambda=detach_lambda,
            )
        else:
            pos_pooled_output = pooled_output
        if prev_state is None and not output_layers.exit_head.is_rnn:
            prev_state = torch.zeros_like(pos_pooled_output)
        if use_prev_hiddens:
            if not output_layers.exit_head.is_rnn:
                _prev_state = pos_pooled_output
                pos_pooled_output = self.prev_state_merging(
                    pos_pooled_output=pos_pooled_output,
                    prev_state=prev_state,
                    influence_type=prev_state_influence,
                )
                prev_state = _prev_state
        return pos_pooled_output, prev_state

    def layer_positional_embeddings_forward(
        self, current_layer_idx, pooled_output, detach_lambda
    ):
        inp_layer = torch.tensor(
            current_layer_idx, device=pooled_output.device, dtype=torch.long
        )
        positional_layer_encoding = self.layer_positional_embeddings(
            inp_layer.unsqueeze(0)
        )
        if detach_lambda:
            pos_pooled_output = (
                pooled_output.detach()
                + positional_layer_encoding.expand_as(pooled_output)
            )
        else:
            pos_pooled_output = pooled_output + positional_layer_encoding.expand_as(
                pooled_output
            )
        return pos_pooled_output

    @staticmethod
    def prev_state_merging(pos_pooled_output, prev_state, influence_type: str):
        if influence_type == "cat":
            return torch.cat((pos_pooled_output, prev_state), dim=1)
        elif influence_type == "diff":
            return pos_pooled_output - prev_state
        elif influence_type == "diff_cat":
            return torch.cat((pos_pooled_output, prev_state), dim=1)
        else:
            raise ValueError(f"Influence type {influence_type} is not understood")


class LambdaLayer(nn.Module):
    def __init__(self, num_layers: int = 1, rnn: bool = False, hidden_dim: int = 768):
        super(LambdaLayer, self).__init__()
        self.is_rnn = rnn
        self.num_layers = num_layers
        if self.is_rnn:
            self.feature_extractor = nn.RNN(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            if num_layers == 1:
                self.feature_extractor = nn.Identity()
            else:
                self.feature_extractor = nn.Sequential(
                    *[
                        nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
                        for _ in range(num_layers)
                    ]
                )
        self.cls = nn.Linear(hidden_dim, 1)

    def forward(self, x, hiddens=None):
        if self.is_rnn:
            features, hiddens = self.feature_extractor(x.unsqueeze(1), hiddens)
            features = features[:, -1]  # last layer output
            # hiddens = hiddens[-1].unsqueeze(0)  # last hidden state
            return self.cls(features), hiddens
        features = self.feature_extractor(x)
        return self.cls(features)


class PonderClassifier(nn.Module):
    def __init__(self, hidden_state_size: int = 768, num_labels: int = 2):
        super().__init__()
        self.cls_head = nn.Linear(hidden_state_size, num_labels)
        self.exit_head = nn.Linear(hidden_state_size, 1)

    def set_lambda_layer_arch(
        self, num_layers: int = 1, rnn: bool = False, hidden_dim: int = 768
    ):
        self.exit_head = LambdaLayer(
            num_layers=num_layers,
            rnn=rnn,
            hidden_dim=hidden_dim,
        )


class PAlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = PAlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifiers = PonderClassifier(config.hidden_size, self.config.num_labels)

        self.init_weights()
        self.config = config
        self.is_rnn = False

    def set_epsilon(self, epsilon: float):
        self.albert.set_epsilon(epsilon)

    def init_pe(self):
        position = torch.arange(1).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.config.hidden_size, 2)
            * (-math.log(10000.0) / self.config.hidden_size)
        )
        pe = torch.zeros(self.config.num_hidden_layers, 1, self.config.hidden_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.squeeze(0)
        self.layer_pos_embeddings.weight.data = pe

    def set_threshold(self, threshold: float):
        self.albert.set_threshold(threshold)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        detach_lambda=False,
        use_prev_hiddens=False,
        use_layer_pos_embeddings=True,
        prev_state_influence="cat",
        exit_criteria="threshold",
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in ``[0, ..., config.num_labels - 1]``.
                If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
                If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
            loss: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
                Classification (or regression if config.num_labels==1) loss.
            logits ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
            Examples::
                from transformers import AlbertTokenizer, AlbertForSequenceClassification
                import torch
                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')
                input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
                labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
                outputs = model(input_ids, labels=labels)
                loss, logits = outputs[:2]
        """

        model_outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_dropout=self.dropout,
            output_layers=self.classifiers,
            detach_lambda=detach_lambda,
            use_layer_pos_embeddings=use_layer_pos_embeddings,
            use_prev_hiddens=use_prev_hiddens,
            prev_state_influence=prev_state_influence,
            exit_criteria=exit_criteria,
        )
        outputs = {"logits": model_outputs["logits"][-1]}
        if "exit_pdf" in model_outputs.keys():
            outputs["exit_pdf"] = model_outputs["exit_pdf"]

        if labels is not None and "exit_pdf" in model_outputs.keys():
            total_loss = 0.0

            for c_probs, c_logits in zip(
                model_outputs["exit_pdf"], model_outputs["logits"]
            ):
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss(reduction="none")
                    loss = loss_fct(c_logits.view(-1), labels.view(-1))  # batch_size
                    reduced_loss = c_probs * loss
                    total_loss += reduced_loss.mean()  # batchmean reduction
                else:
                    loss_fct = CrossEntropyLoss(reduction="none")
                    loss = loss_fct(c_logits.view(-1, self.num_labels), labels.view(-1))
                    reduced_loss = c_probs * loss
                    total_loss += reduced_loss.mean()  # batchmean reduction
            outputs["loss"] = total_loss

        return outputs  # (loss), logits, (hidden_states), (attentions)


class PAlbertForQuestionAnswering(AlbertPreTrainedModel):
    def __init__(self, config):
        self.albert = PAlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifiers = PonderClassifier(config.hidden_size, self.config.num_labels)
        self.albert.disabel_pooler_for_task()

        self.init_weights()
        self.config = config
        self.is_rnn = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        detach_lambda=False,
        use_prev_hiddens=False,
        use_layer_pos_embeddings=True,
        prev_state_influence="cat",
        exit_criteria="threshold",
    ):

        model_outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_dropout=self.dropout,
            output_layers=self.classifiers,
            detach_lambda=detach_lambda,
            use_layer_pos_embeddings=use_layer_pos_embeddings,
            use_prev_hiddens=use_prev_hiddens,
            prev_state_influence=prev_state_influence,
            exit_criteria=exit_criteria,
        )
        last_logits = model_outputs["logits"][-1]
        start_logits, end_logits = last_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        outputs = {
            "logits": last_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
        if "exit_pdf" in model_outputs.keys():
            outputs["exit_pdf"] = model_outputs["exit_pdf"]

        if (
            start_positions is not None
            and end_positions is not None
            and "exit_pdf" in model_outputs.keys()
        ):
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            total_loss = 0.0

            for c_probs, c_logits in zip(
                model_outputs["exit_pdf"], model_outputs["logits"]
            ):
                c_start_logits, c_end_logits = c_logits.split(1, dim=-1)
                c_start_logits = c_start_logits.squeeze(-1).contiguous()
                c_end_logits = c_end_logits.squeeze(-1).contiguous()
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = c_start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)

                loss_fct = CrossEntropyLoss(
                    reduction="none", ignore_index=ignored_index
                )
                start_loss = loss_fct(c_start_logits, start_positions)
                end_loss = loss_fct(c_end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
                reduced_loss = c_probs * loss
                total_loss += reduced_loss.mean()  # batchmean reduction
            outputs["loss"] = total_loss

        return outputs  # (loss), logits, (hidden_states), (attentions)
