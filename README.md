# OOD-DFQ
Data-Free Quantization with Out-of-Distribution Data

## Requirements

Python >= 3.7.10

Pytorch == 1.8.1

## Reproduce results

### Stage1: Curate OOD data.

```
cd data_generate
```

Each helper script (for example `run_generate_imagenet.sh`) now invokes the unified informativeness-based curation pipeline. Update the dataset path, optional teacher checkpoint, and output directory before running:

```
bash run_generate_imagenet.sh
```

The command computes augmentation sensitivity and augmentation potential for a large ImageNet subset, ranks images by the unified score, and writes curated pickle shards together with metadata.


### Stage2: Train the quantized network

```
cd ..
```
1. Modify "qw" and "qa" in cifar10_resnet20.hocon to select desired bit-width.

2. Modify "dataPath" in cifar10_resnet20.hocon to the real dataset path (for construct the test dataloader).

3. Modify "generateDataPath" and ""generateLabelPath" in cifar10_resnet20.hocon to the data_path and label_path you just generate from Stage1.

4. Use the commands in run.sh to train the quantized network. Please note that the model that generates the data and the quantized model should be the same.
