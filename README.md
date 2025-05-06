# EF25_NIPS
Experiments for article

We have only resnet model with cifar-10 dataset for 10 and 100 clients. We also will add SWIN-transformer with imagenet dataset and LLM model. 

now we have four strategies for compressing:
| method | command | description |
|:----------|:--------|:----------|
|top-k via loss| method: 'loss'; k: k| Using loss from clients we choose top-k clients with the biggest loss value|
|top-k via norm of gradients| method: 'gradient_norm'; k: k| Using gradients from clients we choose top-k clients with the biggest gradients norms|
|top-k via bant method| method: 'bant'; k: k| Using bant method we calculate trust scores and choose k clients with the biggest scores|
|top-k via random| method: 'random'; k: k| Randomly choose k clients on each epoch|

In the nearest time we add some new methods: 
| method |description |
|:----------|:--------|
|top-k via method1 + top-t via method2|We choose k clients on each epoch via method1 and choose t clients via method2|
|top-k via angles of gradients|we get gradients from clients, choosing k clients with the least deviation from the mean |

Method launch command
```CUDA_VISIBLE_DIVECES=1 taskset -c 1-34 nice -n 2  nohup python train.py base_dir=home models@models_dict.model1=resnet18 federated_method=ef_full observed_data_params@dataset=cifar10 observed_data_params@trust_df=cifar10_trust observed_data_params@filter=filter_dataframe_files observed_data_params@server_test=cifar10 losses@loss=ce federated_params.round_epochs=1 federated_params.amount_of_clients=10 training_params.batch_size=16 training_params.device_ids=[1] federated_method.trust_sample_amount=1500 federated_params.print_client_metrics=False > outputs/efgrad_3of10_patol_3its_0.2theta.txt &
```
ef_full.yaml must be supplemented with parameters and methods.
