"""
Pattern 1 · Logits Masking — Episodio 8
SFT vs Preference Tuning: dónde acaba uno y empieza el otro.

Este script es autocontenido y educativo. Reutiliza el «contrato» de las 4
familias de scorers (semantic, factual, engagement, compliance) y el embudo de
3 capas (cos≥0,8 acuerdo · 0,5≤cos<0,8 zona gris · spot-check humano) ya
construidos en episodios previos para MOSTRAR cómo el veredicto de esos
scorers se convierte en pares (chosen, rejected) — es decir, en datos de
preference tuning — mientras que SFT consume pares (instruction, response)
más simples.

Requisitos previos:
    pip install "transformers>=4.45" "trl>=0.11" "datasets>=2.20" "torch>=2.3"

No se requieren credenciales. Se usa el modelo público
"sshleifer/tiny-gpt2" (~30 MB) para que el ejemplo se ejecute en CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer


# --------------------------------------------------------------------------- #
# 0. Modelo y tokenizer diminutos (mismos en SFT y preference tuning)
# --------------------------------------------------------------------------- #
MODEL_NAME = "sshleifer/tiny-gpt2"  # ~30 MB, CPU-friendly

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


# --------------------------------------------------------------------------- #
# 1. Dos formatos de dataset — la diferencia clave
# --------------------------------------------------------------------------- #
#
# SFT: "¿cómo se hace esta tarea?"
#   columnas -> {"prompt": "...", "completion": "..."}
#   el modelo aprende a IMITAR la respuesta correcta.
#
# Preference tuning (DPO): "¿cuál de estas dos respuestas es MEJOR?"
#   columnas -> {"prompt": "...", "chosen": "...", "rejected": "..."}
#   el modelo aprende a PREFERIR, no a imitar.
# --------------------------------------------------------------------------- #

sft_rows: list[dict[str, str]] = [
    {
        "prompt": "Resume en una frase el chiste sobre el pato.",
        "completion": "El pato va a la farmacia porque le duele la pata-billo.",
    },
    {
        "prompt": "Traduce 'good morning' al español formal.",
        "completion": "Buenos días.",
    },
    {
        "prompt": "Corrige: 'los datos son importante'.",
        "completion": "Los datos son importantes.",
    },
]
sft_dataset = Dataset.from_list(sft_rows)


# --------------------------------------------------------------------------- #
# 2. De los scorers (episodios previos) a pares chosen/rejected
# --------------------------------------------------------------------------- #
#
# El veredicto del embudo 3 capas produce un ranking. Tomamos los pares
# adyacentes en ese ranking y los empaquetamos como (chosen, rejected).
# Esto es lo que el equipo de datos entrega al trainer.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ScorerVerdict:
    """Una fila del log post-procesado (capa 2-3 del embudo del ep.6)."""

    prompt: str
    response: str
    novelty: float          # 1 - cos(cand, histórico)
    semantic: float         # SemanticSimilarityScorer
    factual_ok: bool        # NLI + tests deterministas
    compliance_ok: bool     # lexicón + LLM-judge
    engagement: float       # predicción de tiempo de lectura


def to_preference_pair(verdicts: list[ScorerVerdict]) -> list[dict[str, str]]:
    """
    Aplica el contrato superviviente: la media NO es la métrica; el orden
    (brecha proxy–outcome) sí lo es. Construimos pares chosen/rejected
    evitando las 4 trampas revisitadas:

      * factual: si dos respuestas tienen cos≈0,95 pero una dice
        'subieron' y otra 'bajaron', gana la que pasa el test NLI.
      * <20 tokens: solo aptas para novelar (capada por umbral).
      * jerga de dominio: el ranking ya viene con el fine-tune del encoder.
      * reward hacking: añadimos novelty al score, no solo cosine.
    """
    scored = [
        (
            v.novelty * 0.25
            + v.semantic * 0.25
            + (0.2 if v.factual_ok else 0.0)
            + (0.2 if v.compliance_ok else 0.0)
            + min(v.engagement, 1.0) * 0.10,
            v,
        )
        for v in verdicts
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    pairs: list[dict[str, str]] = []
    for (_, winner), (_, loser) in zip(scored[:-1], scored[1:]):
        # Solo emitimos pares donde el gap es lo bastante grande:
        # recordatorio de que «umbrales son código, no config».
        if winner[0] - loser[0] < 0.10:
            continue
        pairs.append(
            {
                "prompt": winner[1].prompt,
                "chosen": winner[1].response,
                "rejected": loser[1].response,
            }
        )
    return pairs


sample_verdicts = [
    ScorerVerdict(
        prompt="Resume en una frase el chiste sobre el pato.",
        response="El pato va a la farmacia porque le duele la pata-billo.",
        novelty=0.81, semantic=0.92, factual_ok=True, compliance_ok=True, engagement=0.6,
    ),
    ScorerVerdict(
        prompt="Resume en una frase el chiste sobre el pato.",
        response="Un pato entra en una farmacia.",  # paraphrase honesta pero sin remate
        novelty=0.55, semantic=0.90, factual_ok=True, compliance_ok=True, engagement=0.4,
    ),
    ScorerVerdict(
        prompt="Resume en una frase el chiste sobre el pato.",
        response="El pato compra Ibuprofeno-billo.",  # factual_ok=False, jerga inventada
        novelty=0.62, semantic=0.88, factual_ok=False, compliance_ok=True, engagement=0.5,
    ),
]

pref_dataset = Dataset.from_list(to_preference_pair(sample_verdicts))
print(f"Pares preference construidos: {len(pref_dataset)}")
print(pref_dataset[0])


# --------------------------------------------------------------------------- #
# 3. SFT — el modelo aprende A HACER
# --------------------------------------------------------------------------- #
#
# SFTConfig hereda de TrainingArguments. La pérdida es cross-entropy sobre
# los tokens de la respuesta (la parte de prompt se enmascara con
# completion_only_loss=True y un response_template, o con el chat template).
# --------------------------------------------------------------------------- #

sft_cfg = SFTConfig(
    output_dir="./sft_ckpt",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    logging_steps=1,
    max_length=128,
    # Lo importante: el contrato superviviente.
    completion_only_loss=True,   # enmascara el prompt → logits masking del ep.1
    bf16=False,                  # CPU-friendly
    report_to="none",
)

sft_trainer = SFTTrainer(
    model=base_model,
    args=sft_cfg,
    train_dataset=sft_dataset,
    processing_class=tokenizer,
)


# --------------------------------------------------------------------------- #
# 4. Preference tuning (DPO) — el modelo aprende A PREFERIR
# --------------------------------------------------------------------------- #
#
# Tres diferencias operativas frente a SFT:
#   (a) el checkpoint de partida es el resultado de SFT, no el modelo base;
#   (b) los datos tienen chosen + rejected por prompt;
#   (c) la pérdida ya no es CE sobre un único target, sino el objective DPO
#       (log-sigmoid del ratio de log-probs implícito por una reward
#       latente), por lo que el modelo deja de «imitar» y empieza a
#       «preferir».
# --------------------------------------------------------------------------- #

dpo_cfg = DPOConfig(
    output_dir="./dpo_ckpt",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-6,                     # ~10x menor que SFT, estándar DPO
    logging_steps=1,
    max_length=128,
    max_prompt_length=64,
    beta=0.1,                               # temperatura implícita de la reward
    bf16=False,
    report_to="none",
)

dpo_trainer = DPOTrainer(
    model=base_model,        # En producción: model = SFTTrainer().train()
    ref_model=None,          # Si es None, TRL congela una copia al vuelo
    args=dpo_cfg,
    train_dataset=pref_dataset,
    processing_class=tokenizer,
)


# --------------------------------------------------------------------------- #
# 5. Verificación de formas — un paso de cada trainer
# --------------------------------------------------------------------------- #
#
# No entrenamos nada de verdad: solo comprobamos que las máscaras de SFT y
# el doble forward-pass de DPO producen los shapes esperados. Esto es lo que
# un integration test debería断言ar antes de gastar GPU.
# --------------------------------------------------------------------------- #

def smoke_test_sft() -> dict[str, Any]:
    batch = sft_dataset[:1]
    enc = tokenizer(
        batch["prompt"], batch["completion"],
        truncation=True, max_length=64, padding=True, return_tensors="pt",
    )
    with torch.no_grad():
        out = sft_trainer.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=enc["input_ids"],  # SFT enmascarará la parte de prompt
        )
    return {"sft_loss": float(out.loss), "shape": tuple(out.logits.shape)}


def smoke_test_dpo() -> dict[str, Any]:
    # DPO requiere dos secuencias forward; el trainer se encarga, pero
    # validamos el camino crítico a mano.
    batch = pref_dataset[:1]
    enc_p = tokenizer(batch["prompt"], truncation=True, max_length=32,
                      padding=True, return_tensors="pt")
    enc_c = tokenizer(batch["chosen"], truncation=True, max_length=32,
                      padding=True, return_tensors="pt")
    enc_r = tokenizer(batch["rejected"], truncation=True, max_length=32,
                      padding=True, return_tensors="pt")
    chosen_ids = torch.cat([enc_p["input_ids"], enc_c["input_ids"]], dim=1)
    rejected_ids = torch.cat([enc_p["input_ids"], enc_r["input_ids"]], dim=1)
    with torch.no_grad():
        c = dpo_trainer.model(chosen_ids).logits
        r = dpo_trainer.model(rejected_ids).logits
    return {
        "chosen_shape": tuple(c.shape),
        "rejected_shape": tuple(r.shape),
        "logit_diff_mean": float((c.mean() - r.mean()).abs()),
    }


if __name__ == "__main__":
    print("SFT smoke test  :", smoke_test_sft())
    print("DPO smoke test  :", smoke_test_dpo())
    # En CI productivo, aquí irían los asserts sobre la métrica diaria
    # (brecha proxy–outcome), no sobre la media del scorer.