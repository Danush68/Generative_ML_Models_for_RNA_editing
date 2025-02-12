from modelgenerator.tasks import SequenceRegression
model = SequenceRegression.from_config({"model.backbone": "aido_rna_1b600m"}).eval()
transformed_batch = model.transform({"sequences": ["ACGT", "AGCT"]})
logits = model(transformed_batch)
print(logits)
