# EF25_NIPS  
**Experimental Setup for the Paper**

This repository includes all experiments and implementations required for the EF25_NIPS study.

---

## ✅ Architectures and Datasets

- **Model**: ResNet-18  
- **Dataset**: CIFAR-10  
- **Client Configurations**: Experiments conducted with both 10 and 100 clients  
- **Additional Model Support**: Swin Transformer with the ImageNet and Food101 dataset  

---

## ✅ Implemented Client Selection Strategies

| Method | Command | Description |
|--------|---------|-------------|
| Top-k via loss | `method: 'loss'; k: k` | Selects top-k clients with the highest loss values |
| Top-k via gradient norm | `method: 'gradient_norm'; k: k` | Selects top-k clients with the largest gradient norms |
| Top-k via BANT method | `method: 'bant'; k: k` | Uses BANT trust scores to select k clients with the highest trust |
| Top-k via random sampling | `method: 'random'; k: k` | Randomly selects k clients on each training round |

---

## ✅ Additional Strategies

| Method | Description |
|--------|-------------|
| Hybrid top-k via method1 + top-t via method2 | Selects k clients using method1 and t additional clients using method2 |
| Top-k via gradient angle similarity | Selects k clients whose gradients have the smallest deviation from the mean gradient direction |

---

## ✅ Command to Launch Experiments

```bash
CUDA_VISIBLE_DEVICES=1 taskset -c 1-34 nice -n 2 nohup python train.py \
    base_dir=home \
    models@models_dict.model1=resnet18 \
    federated_method=ef_full \
    observed_data_params@dataset=cifar10 \
    observed_data_params@trust_df=cifar10_trust \
    observed_data_params@filter=filter_dataframe_files \
    observed_data_params@server_test=cifar10 \
    losses@loss=ce \
    federated_params.round_epochs=1 \
    federated_params.amount_of_clients=10 \
    training_params.batch_size=16 \
    training_params.device_ids=[1] \
    federated_method.trust_sample_amount=1500 \
    federated_params.print_client_metrics=False \
    > outputs/efgrad_3of10_patol_3its_0.2theta.txt &
