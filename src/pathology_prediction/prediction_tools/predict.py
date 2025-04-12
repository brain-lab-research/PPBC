import hydra
from omegaconf import DictConfig
import torch

from utils.metrics_utils import print_metrics
from utils.utils import extract_from_checkpoint, get_correct_device
from ecglib.data.datasets import EcgDataset
from utils.model_utils import get_model
from utils.predict_utils import prep_csv_data, predict_on_dataset, save_model_outputs


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def predict(cfg: DictConfig):
    if cfg.federated_params.method in ["FedAvg", "FedProx", "SCAFFOLD", "FLTrust"]:
        state_dict = torch.load(cfg.predict.checkpoint_path)
    else:
        (_, state_dict, _, _) = extract_from_checkpoint(cfg.predict.checkpoint_path)

    data_sources = cfg["observed_data_params"][cfg.dataset]
    leads_num = len(cfg.ecg_record_params.leads)
    # Whether we save results
    model_output_path = (
        cfg.predict.save_model_output if cfg.predict.save_model_output else ""
    )

    for test_source in data_sources:
        print(f"Counting metrics for test data from source {test_source}...")
        # ====================================== Prepare data...
        (
            ecg_info,
            test_target,
            pathology_names,
            num_classes,
        ) = prep_csv_data(cfg, test_source)

        test_dataset = EcgDataset(
            ecg_info,
            test_target,
            data_type="npz",
            frequency=cfg.ecg_record_params.resampled_frequency,
            leads=cfg.ecg_record_params.leads,
            ecg_length=cfg.ecg_record_params.observed_ecg_length,
            classes=num_classes,
            cut_range=cfg.ecg_record_params.ecg_cut_range,
            norm_type=cfg.ecg_record_params.normalization,
            augmentation=None,
        )

        # ====================================== Know your device...
        device = get_correct_device(cfg)
        print(f"**** Device from config: {device}")

        # ====================================== Prepare model...
        model = get_model(cfg=cfg, leads_num=leads_num)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # ====================================== Go...
        thresholds = [0.5]
        results = predict_on_dataset(
            model=model,
            dataset=test_dataset,
            batch_size=cfg.training_params.batch_size,
            device=device,
            num_workers=cfg.training_params.num_workers,
            thresholds=thresholds,
        )

        # ====================================== Prepare metrics report...
        print_metrics(results, thresholds, pathology_names)

        # ====================================== Save model output...
        save_model_outputs(ecg_info, results, model_output_path)


if __name__ == "__main__":
    predict()
