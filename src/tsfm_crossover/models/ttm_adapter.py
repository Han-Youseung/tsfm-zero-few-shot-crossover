"""Pinned TTM R3 official forward/loss, crop and caller-side zero padding."""

from .adapters import ExperimentAdapter, verified_source


class TTMAdapter(ExperimentAdapter):
    def _load(self):
        self.source = verified_source(
            "granite-tsfm", "0.3.9", self.spec["code_commit"], "tsfm_public"
        )
        from tsfm_public.models.tinytimemixer import (
            TinyTimeMixerForDecomposedPrediction,
            TinyTimeMixerForPrediction,
        )

        cls = (
            TinyTimeMixerForPrediction
            if self.config.horizon == 720
            else TinyTimeMixerForDecomposedPrediction
        )
        return cls.from_pretrained(
            self.spec["repository"],
            **self.pretrained_kwargs(),
            prediction_filter_length=self.config.horizon,
        )

    def _past(self, past):
        if self.config.horizon == 720:
            import torch

            return torch.nn.functional.pad(past, (0, 0, 512, 0))
        return past

    def _predict(self, past):
        return self.model(past_values=self._past(past)).prediction_outputs

    def point_forecast(self, prediction):
        return prediction

    def _loss(self, batch, *, training):
        return self.model(past_values=self._past(batch.past), future_values=batch.future).loss

    def _configure(self):
        import torch

        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
