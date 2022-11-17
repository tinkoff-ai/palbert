# coding=utf-8
# Copyright 2022 Tinkoff.
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

from collections import defaultdict
from copy import deepcopy
from typing import Dict, Tuple

import torch
import wandb
from torch import Tensor, nn
from tqdm.autonotebook import tqdm

from loss import RegularizationLoss, kl_with_temperature


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device: str,
        metric,
        scaler=None,
        is_regression: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric
        self.scaler = scaler
        self.is_regression = is_regression
        self.device = device
        if self.is_regression:
            self.base_loss_fn = nn.MSELoss(reduction="none")
        else:
            self.base_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.accumulator = defaultdict(float)
        self.best_metric_dict = {}
        self.best_sd = None
        self.mnli_buffer = {}

    def _step(
        self,
        batch,
        beta,
        prior_lambda,
        pbar,
        is_training_phase,
        detach_lambda,
        prior_type,
        use_prev_hiddens,
        use_layer_pos_embeddings,
        prev_state_influence,
        exit_criteria="threshold",
    ):
        if beta > 0:
            outputs = self.model(
                **batch,
                detach_lambda=detach_lambda,
                use_prev_hiddens=use_prev_hiddens,
                use_layer_pos_embeddings=use_layer_pos_embeddings,
                prev_state_influence=prev_state_influence,
                exit_criteria=exit_criteria,
            )
        else:
            outputs = self.model(**batch)
        if isinstance(outputs, dict):
            logits = outputs["logits"]
            base_loss = outputs.get("loss")
            exit_pdf = outputs.get("exit_pdf")
        else:
            logits = outputs.logits
            base_loss = outputs.loss

        if is_training_phase:
            log_str = f"Task Loss: {base_loss.item():.3f}"
            self.accumulator["task_loss"] += base_loss.item()
        # if alpha > 0.:
        # kd_loss = self.kd_loss(
        # logits=logits[:-1],
        # reference_logits=logits[-1],
        # kl_temperature=kl_temperature,
        # two_sided=two_sided
        # )
        # kd_loss = self.loss_reduce(loss=kd_loss, probs=exit_pdf, reduction=kd_loss_reduction)
        # base_loss = (1 - alpha) * base_loss + alpha * kd_loss
        # self.accumulator["kd_loss"] += kd_loss.item()
        # log_str += f", KD Loss: {kd_loss.item():.3f}"
        loss = None

        if beta > 0 and is_training_phase:
            regularization_loss = self.count_reg_loss(
                probs=exit_pdf, lambda_p=prior_lambda, prior_type=prior_type
            )
            loss = base_loss + beta * regularization_loss
            log_str += f", Reg Loss: {regularization_loss.item():.3f}"
            pdf_sum = torch.stack(exit_pdf).sum(0).mean(-1).item()
            log_str += f", PDF Sum: {pdf_sum:.3f}"
            steps = torch.arange(1, len(exit_pdf) + 1, device=loss.device)
            expected_steps = (torch.stack(exit_pdf) * steps[:, None]).sum(dim=0).mean()
            log_str += f", Expected steps: {expected_steps.item():.2f}"
            self.accumulator["expected_steps"] += expected_steps.item()
            self.accumulator["reg_loss"] += regularization_loss.item()
            self.accumulator["layer_pdf"] += (
                torch.stack(exit_pdf).detach().cpu().mean(-1).view(-1).numpy()
            )
        elif is_training_phase:
            loss = base_loss
            log_str = ""
        if is_training_phase:
            pbar.set_description(f"Loss: {loss.item():.3f}, " + log_str)

        return loss, logits

    def train(
        self,
        dataloaders,
        num_epochs: int = 10,
        beta: float = 0.2,
        prior_lambda: float = 0.5,
        debug: bool = False,
        patience: int = -1,
        val_metric: str = "accuracy",
        pabee: bool = False,
        pondering: bool = False,
        detach_lambda: bool = False,
        prior_type: str = "geometric",
        use_layer_pos_encoding=True,
        use_prev_hiddens=True,
        prev_state_influence="cat",
        exit_criteria="threshold",
    ):
        try:
            p_model = self.model.albert
        except:
            p_model = self.model.roberta
        self.model.to(self.device)
        self.best_metric_dict = {val_metric: -1}
        if patience > 0:
            steps_without_improvement = 0
        for i in range(num_epochs):
            metrics = {}
            print(f"Epoch {i+1}")

            for loader_key, loader in dataloaders.items():
                is_training_phase = "train" in loader_key
                if is_training_phase:
                    self.model.train()
                    self.optimizer.zero_grad()
                    torch.set_grad_enabled(True)
                else:
                    self.model.eval()
                    torch.set_grad_enabled(False)
                pbar = tqdm(loader, leave=False)
                if (pabee or pondering) and not is_training_phase:
                    p_model.reset_stats()

                for b_idx, batch in enumerate(pbar):
                    if is_training_phase:
                        self.optimizer.zero_grad()
                    batch = self.batch_to_device(batch, self.device)

                    with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                        loss, logits = self._step(
                            batch=batch,
                            beta=beta,
                            prior_lambda=prior_lambda,
                            pbar=pbar,
                            is_training_phase=is_training_phase,
                            detach_lambda=detach_lambda,
                            prior_type=prior_type,
                            use_prev_hiddens=use_prev_hiddens,
                            use_layer_pos_embeddings=use_layer_pos_encoding,
                            prev_state_influence=prev_state_influence,
                            exit_criteria=exit_criteria,
                        )
                    if is_training_phase:
                        self.optimize_loss(loss)
                    predictions = (
                        logits.argmax(-1) if not self.is_regression else logits.view(-1)
                    )
                    self.metric.add_batch(
                        predictions=predictions, references=batch["labels"]
                    )
                    if b_idx > 3 and debug:
                        break

                metrics = self.metric.compute()
                metrics = self.post_process_metrics(metrics, loader_key)
                for k, v in self.accumulator.items():
                    metrics[k] = v / (b_idx + 1)  # todo reduction
                if "layer_pdf" in self.accumulator.keys():
                    print(metrics["layer_pdf"])
                self.accumulator = defaultdict(float)
                if is_training_phase:
                    metrics_ = {}
                    for k, v in metrics.items():
                        metrics_["train_" + k] = v
                    metrics = metrics_

            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
            if val_metric in metrics.keys():
                if (
                    patience > 0
                    and metrics[val_metric] <= self.best_metric_dict[val_metric]
                    and not is_training_phase
                ):
                    steps_without_improvement += 1
                elif metrics[val_metric] > self.best_metric_dict[val_metric]:
                    steps_without_improvement = 0
                    self.best_sd = deepcopy(self.model.state_dict())
                    print("Updating best state dict")
            if (pondering or pabee) and not is_training_phase:
                metrics["mean_layers"] = p_model.log_stats()
            if val_metric in metrics.keys():
                self.update_best_metrics(c_metrics=metrics)
                wandb.log(metrics, commit=False)
                c_best = {"best_" + k: v for k, v in self.best_metric_dict.items()}
                wandb.log(c_best)
                print(f"Steps without improvement: {steps_without_improvement}")
            if steps_without_improvement == patience:
                # early exiting
                break

    def test(
        self,
        loaders,
        pabee: bool = False,
        pondering: bool = False,
        beta: int = 0.5,
        use_layer_pos_encoding=True,
        use_prev_hiddens=True,
        prev_state_influence="cat",
        exit_criteria="threshold",
        output_logits=False,
    ):
        try:
            p_model = self.model.albert
        except:
            p_model = self.model.roberta

        predictions = defaultdict(lambda: [])
        torch.set_grad_enabled(False)
        logits = defaultdict(lambda: [])
        self.model.eval()
        self.model.to(self.device)
        for loader_key, loader in loaders.items():
            if pabee or pondering:
                p_model.reset_stats()
            for batch in loader:
                batch = self.batch_to_device(batch, self.device)
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    batch["labels"] = torch.ones_like(batch["labels"])
                    loss, logits = self._step(
                        batch=batch,
                        beta=beta if pondering else 0,
                        pbar=None,
                        is_training_phase=False,
                        detach_lambda=None,
                        prior_lambda=None,
                        prior_type=None,
                        use_prev_hiddens=use_prev_hiddens,
                        prev_state_influence=prev_state_influence,
                        use_layer_pos_embeddings=use_layer_pos_encoding,
                        exit_criteria=exit_criteria,
                    )
                    preds = (
                        logits.argmax(-1) if not self.is_regression else logits.view(-1)
                    )
                    preds = preds.detach().cpu().item()  # as batch_size = 1
                predictions[loader_key] += [preds]
                if output_logits:
                    logits[loader_key] += [logits.detach().cpu().numpy()]
            if pabee or pondering:
                predictions[f"speedup_{loader_key}"] = p_model.log_stats()
        if output_logits:
            return predictions, logits
        return predictions

    def base_loss(self, logits: Tuple[Tensor], batch) -> Tuple[Tensor]:
        losses = tuple()
        for c_logits in logits:
            c_loss = self.base_loss_fn(c_logits, batch["labels"])
            losses += (c_loss,)
        return losses

    def kd_loss(
        self,
        logits: Tuple[Tensor],
        kl_temperature: float,
        two_sided: bool,
        reference_logits: Tensor,
    ) -> Tuple[Tensor]:
        kd_losses = tuple()
        for c_logits in logits:
            kd_losses += (
                kl_with_temperature(
                    c_logits,
                    reference_logits,
                    temperature=kl_temperature,
                    two_sided=two_sided,
                ),
            )
        return kd_losses

    @staticmethod
    def batch_to_device(batch, device):
        keys = batch.keys()
        for k in keys:
            batch[k] = batch[k].to(device)
        return batch

    def count_reg_loss(self, probs, lambda_p, prior_type):
        probs = torch.stack(probs)  # (layer_num, batch_size)
        probs = probs.transpose(0, 1)  # (batch_size, layer_num)
        loss_fn = RegularizationLoss(
            lambda_p=lambda_p,
            max_steps=self.model.config.num_hidden_layers,
            prior_type=prior_type,
        ).to(probs.device)
        return loss_fn(p=probs)

    def optimize_loss(self, loss):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def update_best_metrics(self, c_metrics):
        for metric_key, metric_value in c_metrics.items():
            if "loss" in metric_key:
                reference_value = self.best_metric_dict.get(metric_key, 10)
                self.best_metric_dict[metric_key] = min(reference_value, metric_value)
            else:
                reference_value = self.best_metric_dict.get(metric_key, -1)
                self.best_metric_dict[metric_key] = max(reference_value, metric_value)

    def save_model(self, filename):
        self.model.load_state_dict(self.best_sd)
        self.model.save_pretrained(filename)
        torch.save(self.best_sd, f"{filename}/pytorch_model.bin")

    def post_process_metrics(
        self, metrics: Dict[str, float], loader_key: str
    ) -> Dict[str, float]:
        if "train" in loader_key:
            self.mnli_buffer = {}
        if "f1" in metrics.keys():
            metrics["accuracy_f1"] = (metrics["accuracy"] + metrics["f1"]) / 2
        if "pearson" in metrics.keys():
            metrics["pearson_spearman"] = (
                metrics["pearson"] + metrics["spearmanr"]
            ) / 2
        if "validation_matched" == loader_key:
            self.mnli_buffer[loader_key] = metrics
            metrics_ = {}
            for m_k, m_v in metrics.items():
                metrics_[f"{m_k}_matched"] = m_v
            metrics = metrics_
        if "validation_mismatched" == loader_key:
            self.mnli_buffer[loader_key] = metrics
            metrics_ = {}
            for m_k, m_v in metrics.items():
                metrics_[f"{m_k}_mismatched"] = m_v
            metrics = metrics_

        if len(self.mnli_buffer) == 2:
            for (matched_metric_key, matched_metric_value), (
                mismatched_metric_key,
                mismatched_metric_value,
            ) in zip(
                self.mnli_buffer["validation_matched"].items(),
                self.mnli_buffer["validation_mismatched"].items(),
            ):
                assert matched_metric_key == mismatched_metric_key
                metrics[f"{matched_metric_key}_matched"] = matched_metric_value
                metrics[matched_metric_key] = (
                    mismatched_metric_value + matched_metric_value
                ) / 2
        return metrics
