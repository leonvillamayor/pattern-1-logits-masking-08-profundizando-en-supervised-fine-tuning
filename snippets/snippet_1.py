# Ejemplo ilustrativo (pseudo-código, no ejecutar literalmente)
from transformers import TrainingArguments

args = TrainingArguments(
    learning_rate=2e-5,        # ancla cerca del fin del pretraining
    num_train_epochs=2,        # pocas vueltas, ver abajo
    per_device_train_batch_size=4,
    warmup_ratio=0.03,         # subida suave para no romper al inicio
    lr_scheduler_type="cosine",
)