"""
Unlearning methods for credit scoring transformers.

Methods:
  1. Full Retrain       -- gold standard (slow)
  2. SISA               -- shard-based retraining
  3. Gradient Ascent     -- maximize loss on forget set
  4. Gradient Difference -- ascend forget + descend retain
  5. Influence Functions -- Newton-step approximation
  6. Fine-tune Retain    -- continue training on retain only
  7. SCRUB              -- teacher-student KL divergence

LoRA-based methods (novel contribution):
  - Forget Adapter      -- Phase 2: LoRA GradDiff on forget set
  - Retain Adapter      -- Phase 3: KL distillation to recover utility
"""

# Individual method files (new modular structure)
from unlearning.full_retrain import unlearn as full_retrain_unlearn
from unlearning.full_retrain import full_retrain
from unlearning.sisa import unlearn as sisa_unlearn
from unlearning.sisa import sisa_full
from unlearning.gradient_ascent import unlearn as ga_unlearn
from unlearning.gradient_ascent import gradient_ascent_unlearn
from unlearning.gradient_diff import unlearn as graddiff_unlearn
from unlearning.gradient_diff import gradient_diff_unlearn
from unlearning.influence_functions import unlearn as influence_unlearn
from unlearning.influence_functions import influence_fn_unlearn
from unlearning.finetune_retain import unlearn as ft_retain_unlearn
from unlearning.finetune_retain import finetune_retain_unlearn
from unlearning.scrub import unlearn as scrub_unlearn_method
from unlearning.scrub import scrub_unlearn
from unlearning.random_labels import unlearn as random_labels_unlearn_method
from unlearning.random_labels import random_labels_unlearn

# Legacy baselines (kept for backward compatibility with existing pipeline)
from unlearning.baselines import (
    baseline_full_retrain,
    baseline_gradient_ascent,
    baseline_finetune_retain,
    baseline_sisa,
    baseline_influence_functions,
    baseline_random_labels,
)

# LoRA-based methods
from unlearning.forget_adapter import run_forget_adapter
from unlearning.retain_adapter import run_retain_adapter
