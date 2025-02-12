from modelgenerator.tasks import Embed
model = Embed.from_config({"model.backbone": "aido_rna_1b600m"}).eval()
transformed_batch = model.transform({"sequences": ["ACGT", "AGCT"]})
embedding = model(transformed_batch)
print(embedding.shape)
print(embedding)
