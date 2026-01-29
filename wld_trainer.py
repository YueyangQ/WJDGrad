from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import time
import torch
import collections
from packaging import version
from torch.distributions import Categorical
import torch.nn as nn
from loss_func.repnoise_loss import rep_noise_loss
from transformers import Trainer
from transformers import logging
import torch.nn.functional as F

import transformers
from transformers.utils import (
    is_sagemaker_mp_enabled
)
from transformers.trainer_pt_utils import (
    get_parameter_names,
)

from transformers.models.llama.modeling_llama import LlamaAttention,LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.gemma.modeling_gemma import GemmaAttention
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)
IGNORE_INDEX = -100

class SafeGrad_WJD_Alpha_Trainer(Trainer):
    """
    SafeGrad + W-JDGrad (alpha-clamped) Trainer - EXACTLY matches Algorithm pseudocode
    
    This implementation PRECISELY follows the pseudocode in the paper:
    
    Algorithm: WJDGrad for Secure LLM Fine-tuning
    ============================================
    Initialize: λ (reliability relaxation), T (total steps), 
                D_u (user dataset), D_a (alignment dataset),
                w_0 (initial params), η (learning rate)
    Assume: Alignment gradient is reliable; user gradient may be noisy/poisoned
    
    For t = 0, 1, 2, ..., T-1:
        1. Sample user batch (x_u, y_u) from D_u
        2. g_u ← ∇ℓ_u(w; x_u, y_u)
        3. Sample alignment batch (x_a, y_a) from D_a
        4. g_a ← ∇ℓ_a(w; x_a, y_a)  [via KL(P_ref || P_theta)]
        5. c ← ⟨g_a, g_u⟩
        
        6. if c ≥ 0:
               // No conflict: standard joint update
               g ← g_a + g_u
           else:
               // Conflict: WJDGrad closed-form update
               d ← g_a - g_u
               α_uncon ← ⟨g_a, d⟩ / ||d||²
               α_low ← max(0, (-λ - ⟨g_a, g_u⟩) / (||g_u||² - ⟨g_a, g_u⟩))
               α_up ← min(1, ||g_a||² / (||g_a||² - ⟨g_a, g_u⟩))
               α ← min(α_up, max(α_low, α_uncon))
               g ← g_a + α(g_u - g_a)
        
        7. w_{t+1} ← w_t - η·g
    
    return w_T
    
    Key Implementation Details:
    - Uses KL(P_ref || P_theta) for alignment loss (same as SafeGrad_WJD_Trainer)
    - Computes alpha bounds exactly as shown in pseudocode
    - Handles intersection params only; others use single-source grads
    """

    def __init__(
        self,
        *args,
        projection: bool = True,
        ref_model=None,
        ref_model_name_or_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize WJDGrad Trainer - matches Algorithm pseudocode initialization.
        
        Args:
          projection: kept for compatibility (unused in WJDGrad)
          ref_model / ref_model_name_or_path: frozen safety-aligned reference model (w*)
          
        Training args (in self.args) - corresponds to Algorithm parameters:
          - wjd_lambda: float = 0.5      # λ: Reliability relaxation parameter
          - kl_temperature: float = 1.0   # τ: Temperature for KL divergence
          - learning_rate: float          # η: Learning rate
          - guide_data_num: int > 0       # Size of alignment dataset D_a
          
        The reference model w* is used to compute alignment gradient via:
          g_a = ∇ KL(P_{w*} || P_w)
        """
        super().__init__(*args, **kwargs)

        if ref_model is None and ref_model_name_or_path is None:
            raise ValueError(
                "SafeGrad_WJD_Alpha_Trainer requires a reference model (w*) for alignment gradient."
            )

        # Reference model: w* (frozen safety-aligned model)
        if ref_model is not None:
            print("Using provided reference model w* for alignment gradient computation.")
            self.ref_model = ref_model
        else:
            print(f"Loading reference model w* from: {ref_model_name_or_path}")
            self.ref_model = transformers.AutoModelForCausalLM.from_pretrained(
                ref_model_name_or_path,
                load_in_8bit=False,
                torch_dtype=torch.float16 if self.args.fp16 else (torch.bfloat16 if self.args.bf16 else torch.float32),
                device_map="auto",
            )

        # Ensure reference model is on correct device and frozen
        self.ref_model.to(self.accelerator.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Training state
        self.projection = projection  # kept for compatibility
        self.t = 0  # Current step (matches pseudocode: t = 0, 1, 2, ..., T-1)
        self.alignment_dataloader = None
        self.data_iter = None

        # Extract hyperparameters from args
        lambda_val = getattr(self.args, "wjd_lambda", 0.5)
        tau = getattr(self.args, "kl_temperature", 1.0)
        
        print("=" * 70)
        print("SafeGrad_WJD_Alpha_Trainer initialized (matches Algorithm pseudocode)")
        print("=" * 70)
        print(f"  λ (wjd_lambda):        {lambda_val}  # Reliability relaxation")
        print(f"  τ (kl_temperature):    {tau}         # KL temperature")
        print(f"  η (learning_rate):     {self.args.learning_rate}  # Learning rate")
        print(f"  Reference model (w*):  {'Provided' if ref_model else ref_model_name_or_path}")
        print("=" * 70)

    # ---------------- Alignment dataloader (D_a in pseudocode) ----------------
    def get_alignment_dataloader(self, alignment_dataset) -> DataLoader:
        from transformers.trainer_utils import seed_worker

        data_collator = self.data_collator
        sampler = RandomSampler(alignment_dataset)

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(alignment_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(alignment_dataset, **dataloader_params))

    def init(self, alignment_dataset):
        """
        Initialize training state before starting training loop.
        Call this once before training to set up alignment dataset D_a.
        
        Corresponds to Algorithm initialization:
          - D_a: alignment dataset (clean safety-aligned data)
          - t = 0: step counter
        """
        self.t = 0  # Training step counter (matches pseudocode: t = 0, 1, 2, ..., T-1)
        
        if getattr(self.args, "guide_data_num", 0) > 0:
            print(f"Initializing alignment dataloader with guide_data_num={self.args.guide_data_num}")
            self.alignment_dataloader = self.get_alignment_dataloader(alignment_dataset)
            self.data_iter = iter(self.alignment_dataloader)
        else:
            raise ValueError(
                "SafeGrad_WJD_Alpha_Trainer requires guide_data_num > 0 to enable alignment gradient. "
                "Set args.guide_data_num to the number of alignment samples per epoch."
            )

    def sample_from_alignment(self):
        """
        Sample one batch x_a ~ D_a from alignment dataset.
        
        Corresponds to Algorithm step 2:
          "Sample batch x_a ~ D_a, compute L_a(w; x_a) = KL(P_{w*} || P_w)"
        
        Returns:
          batch: dictionary with input_ids, attention_mask, labels
        """
        if self.data_iter is None:
            raise ValueError(
                "Alignment dataloader not initialized. Call trainer.init(alignment_dataset) "
                "and ensure args.guide_data_num > 0."
            )
        try:
            batch = next(self.data_iter)
        except StopIteration:
            # Restart iterator when epoch ends
            self.data_iter = iter(self.alignment_dataloader)
            batch = next(self.data_iter)
        return batch

    # ---------------- flatten / unflatten ----------------
    @staticmethod
    def _flatten_grad_dict(
        grad_dict: Dict[str, torch.Tensor],
        param_order: Tuple[Tuple[str, torch.nn.Parameter], ...]
    ) -> Optional[torch.Tensor]:
        flats = []
        for name, p in param_order:
            g = grad_dict.get(name, None)
            if g is not None:
                flats.append(g.reshape(-1).to(torch.float32))
        if not flats:
            return None
        return torch.cat(flats, dim=0)

    @staticmethod
    def _unflatten_to_params(
        flat_vec: torch.Tensor,
        param_order: Tuple[Tuple[str, torch.nn.Parameter], ...],
        g_task: Dict[str, torch.Tensor],
        g_align: Dict[str, torch.Tensor],
    ):
        """
        Write merged grads for intersection params; keep task-only & align-only as fallbacks.
        """
        offset = 0
        in_intersection_names = set()

        # Write intersection merged grads
        for name, p in param_order:
            gt = g_task.get(name, None)
            ga = g_align.get(name, None)
            if (gt is not None) and (ga is not None):
                numel = p.numel()
                seg = flat_vec[offset: offset + numel].to(p.dtype).view_as(p)
                p.grad = seg
                offset += numel
                in_intersection_names.add(name)

        # Fill task-only & align-only
        param_name_to_param = dict(param_order)
        # NOTE: param_order is intersection-only in our use; so we need full name->param map externally.
        # We'll override in training_step with a full map pass.

    # ---------------- W-JDGrad alpha-clamped merge (MATCHES Algorithm pseudocode steps 3-6) ----------------
    @staticmethod
    def _wjdgrad_alpha_merge(
        g_align_rel: torch.Tensor,
        g_task_noisy: torch.Tensor,
        lam: float,
        eps: float = 1e-12,
    ):
        """
        Implements WJDGrad merge with alpha bounds (Algorithm pseudocode steps 3-6).
        
        Args:
          g_align_rel (g_a): reliable alignment gradient
          g_task_noisy (g_u): possibly noisy task gradient
          lam (λ): reliability relaxation parameter (default 0.5)
          eps: numerical stability constant
          
        Pseudocode mapping:
          Step 3: Check compatibility c = <g_a, g_u>
                  if c >= 0: g = g_a + g_u (compatible case)
                  
          Step 4: Compute unconstrained alpha:
                  α_uncon = <g_a, g_a - g_u> / ||g_a - g_u||^2
                  
          Step 5: Compute lower bound (reliability constraint):
                  α_low = max(0, (-λ - <g_a, g_u>) / (||g_u||^2 - <g_a, g_u>))
                  
          Step 6: Compute upper bound (alignment preservation):
                  α_up = min(1, ||g_a||^2 / (||g_a||^2 - <g_a, g_u>))
                  
                  Final alpha: α = min(α_up, max(α_low, α_uncon))
                  Merged gradient: g = g_a + α(g_u - g_a)
        
        Returns:
          g: merged gradient
          dot12: <g_a, g_u> (compatibility metric)
          denom_d: ||g_a - g_u||^2 (conflict magnitude)
          alpha_uncon: unconstrained optimal alpha
          alpha_low: lower bound from reliability constraint
          alpha_up: upper bound from alignment preservation
          alpha: final clipped alpha value
          compatible: True if no conflict (<g_a, g_u> >= 0)
        """
        ga = g_align_rel
        gu = g_task_noisy
        
        # Step 3: Compatibility check
        dot12 = torch.dot(ga, gu)  # c = <g_a, g_u>

        # Compatible case: no gradient conflict
        if dot12 >= 0:
            g = ga + gu  # Standard joint descent direction
            denom_d = torch.tensor(0.0, device=ga.device, dtype=ga.dtype)
            alpha_uncon = torch.tensor(1.0, device=ga.device, dtype=ga.dtype)
            alpha_low = torch.tensor(0.0, device=ga.device, dtype=ga.dtype)
            alpha_up = torch.tensor(1.0, device=ga.device, dtype=ga.dtype)
            alpha = torch.tensor(1.0, device=ga.device, dtype=ga.dtype)
            return g, dot12, denom_d, alpha_uncon, alpha_low, alpha_up, alpha, True

        # Conflict case: WJDGrad closed-form solution with alpha bounds
        d = ga - gu  # Difference vector
        denom_d = torch.dot(d, d).clamp_min(eps)  # ||g_a - g_u||^2

        # Step 4: Unconstrained optimal alpha (minimizes weighted combination)
        # α_uncon = <g_a, d> / ||d||^2 = <g_a, g_a - g_u> / ||g_a - g_u||^2
        alpha_uncon = torch.dot(ga, d) / denom_d

        # Step 5: Lower bound α_low (ensures reliability: <g_merged, g_u> >= -λ)
        # α_low = max(0, (-λ - <g_a, g_u>) / (||g_u||^2 - <g_a, g_u>))
        gu_norm2 = torch.dot(gu, gu)  # ||g_u||^2
        denom_low = (gu_norm2 - dot12).clamp_min(eps)  # ||g_u||^2 - <g_a, g_u>
        alpha_low = (-lam - dot12) / denom_low
        alpha_low = torch.clamp(alpha_low, min=0.0)  # Ensure non-negative

        # Step 6: Upper bound α_up (preserves alignment: <g_merged, g_a> >= 0)
        # α_up = min(1, ||g_a||^2 / (||g_a||^2 - <g_a, g_u>))
        ga_norm2 = torch.dot(ga, ga)  # ||g_a||^2
        denom_up = (ga_norm2 - dot12).clamp_min(eps)  # ||g_a||^2 - <g_a, g_u>
        alpha_up = ga_norm2 / denom_up
        alpha_up = torch.clamp(alpha_up, max=1.0)  # Ensure <= 1

        # Final alpha: clip unconstrained solution to [α_low, α_up]
        # α = min(α_up, max(α_low, α_uncon))
        alpha = torch.max(alpha_low, alpha_uncon)
        alpha = torch.min(alpha_up, alpha)

        # Merged gradient: g = g_a + α(g_u - g_a)
        g = ga + alpha * (gu - ga)

        return g, dot12, denom_d, alpha_uncon, alpha_low, alpha_up, alpha, False

    # ---------------- training step - MATCHES Algorithm pseudocode steps 1-7 ----------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Implements Algorithm training step (pseudocode steps 1-7):
        
        1. Sample batch x_u ~ D_u, compute L_u(w; x_u), get g_u = ∇L_u(w; x_u)
        2. Sample batch x_a ~ D_a, compute L_a(w; x_a), get g_a = ∇L_a(w; x_a)
        3. Check compatibility:
             if <g_a, g_u> >= 0: g = g_a + g_u (no conflict)
             else: compute WJDGrad solution with alpha bounds
        4-6. Alpha bounds and clipping (see _wjdgrad_alpha_merge)
        7. Update w ← w - η g (done by optimizer outside this method)
        """
        model.train()

        # ===== Step 1: Sample x_u ~ D_u, compute g_u (task gradient) =====
        # L_u is standard CE loss for user task batch
        finetune_inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss_task = self.compute_loss(model, finetune_inputs, return_outputs=False)

        self.accelerator.backward(loss_task)
        g_task = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)

        # ===== Step 2: Sample x_a ~ D_a, compute g_a (alignment gradient) =====
        # L_a = KL(P_{w*} || P_w) where w* is frozen reference model
        alignment_inputs = self.sample_from_alignment()
        alignment_inputs = self._prepare_inputs(alignment_inputs)

        # Forward pass on reference model w* (frozen, no grad)
        with torch.no_grad():
            ref_outputs = self.ref_model(**alignment_inputs)
            logits_ref = ref_outputs.logits

        # Forward pass on trainable model w
        finetune_outputs = model(**alignment_inputs)
        logits_theta = finetune_outputs.logits

        # KL divergence with temperature τ: KL(P_ref || P_theta)
        tau = float(getattr(self.args, "kl_temperature", 1.0))
        log_p_ref = F.log_softmax(logits_ref / tau, dim=-1)
        log_p_theta = F.log_softmax(logits_theta / tau, dim=-1)

        # Mask for valid tokens (ignore padding and -100 labels)
        labels = alignment_inputs.get("labels", None)
        attn = alignment_inputs.get("attention_mask", None)

        if labels is not None:
            valid_mask = (labels != -100)
        else:
            valid_mask = attn.bool() if attn is not None else torch.ones_like(
                logits_theta[..., 0], dtype=torch.bool
            )
        if attn is not None:
            valid_mask = valid_mask & (attn.bool())

        valid_mask_b = valid_mask.unsqueeze(-1)

        # KL divergence: sum over vocab, average over valid tokens
        kl_per_token = F.kl_div(
            input=log_p_theta,      # log P_theta
            target=log_p_ref,       # log P_ref (log-space target)
            reduction="none",
            log_target=True
        )

        kl_per_token = kl_per_token * valid_mask_b
        num_valid_tokens = valid_mask.sum()

        if num_valid_tokens > 0:
            # Scale by τ² for temperature-scaled KL
            loss_align = kl_per_token.sum() * (tau * tau) / num_valid_tokens
        else:
            loss_align = torch.tensor(0.0, device=logits_theta.device, dtype=logits_theta.dtype)

        # Compute g_a = ∇L_a(w; x_a)
        self.accelerator.backward(loss_align)
        g_align = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)

        # ===== Step 3: Compatibility check + WJDGrad merge =====
        # Only merge on intersection parameters (where both g_u and g_a exist)
        intersection = [(name, p) for name, p in model.named_parameters() if (name in g_task and name in g_align)]
        param_order = tuple(intersection)

        # Flatten to vectors for dot product computation
        g_rel = self._flatten_grad_dict(g_align, param_order)   # g_a (reliable alignment gradient)
        g_noisy = self._flatten_grad_dict(g_task, param_order)  # g_u (possibly noisy task gradient)

        lam = float(getattr(self.args, "wjd_lambda", 0.5))

        # Full parameter map for fallback (non-intersection params)
        name_to_param = {name: p for name, p in model.named_parameters()}

        # Handle edge case: no intersection
        if (g_rel is None) or (g_noisy is None):
            # No overlapping parameters: prefer alignment gradient if available, else task
            for name, p in name_to_param.items():
                ga = g_align.get(name, None)
                gt = g_task.get(name, None)
                if ga is not None:
                    p.grad = ga
                elif gt is not None:
                    p.grad = gt

            logs = {
                "loss_task": float(loss_task.detach().cpu()),
                "loss_align_kld": float(loss_align.detach().cpu()),
                "wjd_has_intersection": False,
                "wjd_lambda": lam,
                "kl_temperature": float(tau),
            }
            self.log(logs)
            return loss_task.detach() / self.args.gradient_accumulation_steps

        # ===== Steps 4-6: Alpha bounds and WJDGrad merge =====
        # _wjdgrad_alpha_merge implements:
        #   if <g_a, g_u> >= 0: g = g_a + g_u (compatible case)
        #   else: compute alpha with bounds [alpha_low, alpha_up], clamp alpha_uncon
        #         g = g_a + alpha (g_u - g_a)
        g_merged, dot12, denom_d, a_uncon, a_low, a_up, alpha, compatible = self._wjdgrad_alpha_merge(
            g_rel, g_noisy, lam
        )

        # Write merged gradients back to intersection parameters
        offset = 0
        merged_names = set()
        for name, p in param_order:
            numel = p.numel()
            seg = g_merged[offset: offset + numel].to(p.dtype).view_as(p)
            p.grad = seg
            offset += numel
            merged_names.add(name)

        # Fill non-intersection parameters with single-source gradients
        # (task-only or alignment-only parameters not in intersection)
        for name, p in name_to_param.items():
            if name in merged_names:
                continue
            gt = g_task.get(name, None)
            ga = g_align.get(name, None)
            if gt is not None and ga is None:
                p.grad = gt
            elif ga is not None and gt is None:
                p.grad = ga

        # ===== Step 7: Update w ← w - η g =====
        # (Optimizer step is called automatically by Trainer after this method returns)

        # Logging for analysis
        logs = {
            "loss_task": float(loss_task.detach().cpu()),
            "loss_align_kld": float(loss_align.detach().cpu()),
            "wjd_has_intersection": True,
            "wjd_compatible": bool(compatible),
            "wjd_dot12": float(dot12.detach().cpu()),  # <g_a, g_u>
            "wjd_denom_d": float(denom_d.detach().cpu()),  # ||g_a - g_u||^2
            "wjd_alpha_uncon": float(a_uncon.detach().cpu()),  # unconstrained alpha
            "wjd_alpha_low": float(a_low.detach().cpu()),  # lower bound
            "wjd_alpha_up": float(a_up.detach().cpu()),  # upper bound
            "wjd_alpha": float(alpha.detach().cpu()),  # final alpha used
            "wjd_lambda": lam,
            "kl_temperature": float(tau),
        }
        self.log(logs)

        return loss_task.detach() / self.args.gradient_accumulation_steps
