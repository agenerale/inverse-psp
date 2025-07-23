import logging
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from timm.utils.model import get_state_dict, unwrap_model
from timm.utils.model_ema import ModelEmaV2

import logging
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from timm.utils.model import get_state_dict, unwrap_model
from timm.utils.model_ema import ModelEmaV2


class EMACallback(Callback):
    """
    Model Exponential Moving Average. Empirically it has been found that using the moving average
    of the trained parameters of a deep network is better than using its trained parameters directly.

    If `use_ema_weights`, then the EMA parameters of the network are set after training ends.
    """

    def __init__(self, decay=0.9999, use_ema_weights: bool = True):
        self.decay = decay
        self.ema = None
        self.use_ema_weights = use_ema_weights

    def on_fit_start(self, trainer, pl_module):
        "Initialize `ModelEmaV2` from timm to keep a copy of the moving average of the weights"
        self.ema = ModelEmaV2(pl_module, decay=self.decay, device=None)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        "Update the stored parameters using a moving average"
        self.ema.update(pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        "Do validation using the stored parameters"
        self.store(pl_module.parameters())
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())

    def on_validation_end(self, trainer, pl_module):
        "Restore original parameters to resume training later"
        self.restore(pl_module.parameters())

    def on_train_end(self, trainer, pl_module):
        if self.use_ema_weights:
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())
            msg = "Model weights replaced with the EMA version."
            print(msg)

    def state_dict(self):
        "Save EMA state"
        if self.ema is not None:
            return {"state_dict_ema": get_state_dict(self.ema, unwrap_model)}
        return {}

    def load_state_dict(self, state_dict):
        "Load EMA state"
        if self.ema is not None and "state_dict_ema" in state_dict:
            self.ema.module.load_state_dict(state_dict["state_dict_ema"])

    def store(self, parameters):
        "Save the current parameters for restoring later."
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        "Restore the parameters stored with the `store` method."
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        "Copy current parameters into the given collection of parameters."
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)


# class EMACallback(Callback):
#     """
#     Model Exponential Moving Average. Empirically it has been found that using the moving average
#     of the trained parameters of a deep network is better than using its trained parameters directly.

#     If `use_ema_weights`, then the ema parameters of the network is set after training end.
#     """

#     def __init__(self, decay=0.9999, use_ema_weights: bool = True):
#         self.decay = decay
#         self.ema = None
#         self.use_ema_weights = use_ema_weights

#     def on_fit_start(self, trainer, pl_module):
#         "Initialize `ModelEmaV2` from timm to keep a copy of the moving average of the weights"
#         self.ema = ModelEmaV2(pl_module, decay=self.decay, device=None)

#     def on_train_batch_end(
#         self, trainer, pl_module, outputs, batch, batch_idx
#     ):
#         "Update the stored parameters using a moving average"
#         # Update currently maintained parameters.
#         self.ema.update(pl_module)

#     def on_validation_epoch_start(self, trainer, pl_module):
#         "do validation using the stored parameters"
#         # save original parameters before replacing with EMA version
#         self.store(pl_module.parameters())

#         # update the LightningModule with the EMA weights
#         # ~ Copy EMA parameters to LightningModule
#         self.copy_to(self.ema.module.parameters(), pl_module.parameters())

#     def on_validation_end(self, trainer, pl_module):
#         "Restore original parameters to resume training later"
#         self.restore(pl_module.parameters())

#     def on_train_end(self, trainer, pl_module):
#         # update the LightningModule with the EMA weights
#         if self.use_ema_weights:
#             self.copy_to(self.ema.module.parameters(), pl_module.parameters())
#             msg = "Model weights replaced with the EMA version."
#             print(msg)

#     def on_save_checkpoint(self, trainer, pl_module, checkpoint):
#         if self.ema is not None:
#             return {"state_dict_ema": get_state_dict(self.ema, unwrap_model)}

#     def on_load_checkpoint(self, trainer, pl_module, checkpoint, **kwargs):
#         if self.ema is not None:
#             self.ema.module.load_state_dict(checkpoint["state_dict_ema"])

#     def store(self, parameters):
#         "Save the current parameters for restoring later."
#         self.collected_params = [param.clone() for param in parameters]

#     def restore(self, parameters):
#         """
#         Restore the parameters stored with the `store` method.
#         Useful to validate the model with EMA parameters without affecting the
#         original optimization process.
#         """
#         for c_param, param in zip(self.collected_params, parameters):
#             param.data.copy_(c_param.data)

#     def copy_to(self, shadow_parameters, parameters):
#         "Copy current parameters into given collection of parameters."
#         for s_param, param in zip(shadow_parameters, parameters):
#             if param.requires_grad:
#                 param.data.copy_(s_param.data)