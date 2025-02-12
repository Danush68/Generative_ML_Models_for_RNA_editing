from genbio_finetune.tasks import Embed
model = Embed.from_config({"model.backbone": "aido_rna_1b600m"}).eval()
transformed_batch = model.transform({"sequences": ["ACGT", "ACGT"]})
embedding = model(transformed_batch)
print(embedding.shape)
print(embedding)
