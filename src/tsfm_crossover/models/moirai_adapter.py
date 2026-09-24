"""Joint-target MOIRAI 1.1 with pinned official wrapper and PackedNLLLoss."""

from .adapters import ExperimentAdapter, verified_source


class MoiraiAdapter(ExperimentAdapter):
    def _load(self):
        self.source = verified_source("uni2ts", "2.0.0", self.spec["code_commit"], "uni2ts")
        from uni2ts.model.moirai import MoiraiModule

        return MoiraiModule.from_pretrained(self.spec["repository"], **self.pretrained_kwargs())

    def _forecast(self):
        from uni2ts.model.moirai import MoiraiForecast

        module = self.model.module if self.optimizer_created else self.model
        return MoiraiForecast(
            prediction_length=self.config.horizon,
            target_dim=len(self.config.channel_names),
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=self.config.context,
            module=module,
            patch_size=64,
            num_samples=self.config.num_samples,
        )

    def _predict(self, past):
        import torch

        return self._forecast().eval()(
            past,
            torch.ones_like(past, dtype=torch.bool),
            torch.zeros(past.shape[:2], dtype=torch.bool, device=past.device),
        )

    def point_forecast(self, prediction):
        if prediction.ndim != 4 or prediction.shape[1] != self.config.num_samples:
            raise ValueError("MOIRAI samples must be (batch, samples, horizon, channels)")
        return prediction.median(dim=1).values

    def _configure(self):
        from uni2ts.loss.packed import PackedNLLLoss
        from uni2ts.model.moirai import MoiraiFinetune

        self.model = MoiraiFinetune(
            min_patches=2,
            min_mask_ratio=0.15,
            max_mask_ratio=0.5,
            max_dim=max(128, len(self.config.channel_names)),
            num_training_steps=self.config.max_optimizer_steps,
            num_warmup_steps=0,
            module=self.model,
            num_samples=self.config.num_samples,
            loss_func=PackedNLLLoss(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            context_length=self.config.context,
            prediction_length=self.config.horizon,
            patch_size=64,
            finetune_pattern="full",
        ).to(self.device)
        return self.model.configure_optimizers()["optimizer"]

    def _loss(self, batch, *, training):
        import torch

        if not self.optimizer_created:
            raise RuntimeError("configure official training wrapper before objective evaluation")
        forecast = self._forecast()
        past, future = batch.past, batch.future
        values = forecast._convert(
            64,
            past,
            torch.ones_like(past, dtype=torch.bool),
            torch.zeros(past.shape[:2], dtype=torch.bool, device=past.device),
            future_target=future,
            future_observed_target=torch.ones_like(future, dtype=torch.bool),
            future_is_pad=torch.zeros(future.shape[:2], dtype=torch.bool, device=future.device),
        )
        packed = dict(
            zip(
                (
                    "target",
                    "observed_mask",
                    "sample_id",
                    "time_id",
                    "variate_id",
                    "prediction_mask",
                ),
                values,
                strict=True,
            )
        )
        packed["patch_size"] = torch.full_like(packed["time_id"], 64, dtype=torch.long)
        return (
            self.model.training_step(packed, self.global_step)
            if training
            else self.model.validation_step(packed, self.global_step)
        )
