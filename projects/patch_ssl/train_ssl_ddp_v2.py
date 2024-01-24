from dataclasses import dataclass
from medAI.datasets import ExactNCT2013BmodePatches, PatchOptions, CohortSelectionOptions
import torch


@dataclass(frozen=True)
class DataConfig: 
    patch_options_ssl: PatchOptions = PatchOptions()
    fold: int = 0 
    n_folds: int = 5
    benign_to_cancer_ratio_ssl: int | None = None


def get_dataloaders(config: DataConfig): 
    from medAI.datasets import ExactNCT2013BmodePatches, PatchOptions
    patch_options_ssl = config.patch_options_ssl

    class SSLTransform: 
        def __call__(self, item): 
            patch = item['patch']
            patch = torch.from_numpy(patch).float()
            patch = patch.unsqueeze(0)
            patch = (patch - patch.mean()) / patch.std()
            return patch

    class ZipDataset: 
        def __init__(self, dataset1, dataset2): 
            self.dataset1 = dataset1
            self.dataset2 = dataset2
            self.length = len(dataset1)
        def __getitem__(self, idx): 
            return self.dataset1[idx], self.dataset2[idx]
        def __len__(self):
            return self.length
        
    ssl_dataset = ZipDataset(
        ExactNCT2013BmodePatches(
            transform=SSLTransform(), 
            cohort_selection_options=CohortSelectionOptions(
                fold=config.fold, n_folds=config.n_folds,
                min_involvement=None, 
                remove_benign_from_positive_patients=False, 
                benign_to_cancer_ratio=config.benign_to_cancer_ratio_ssl
            ),
            patch_options=patch_options_ssl
        ), 
        ExactNCT2013BmodePatches(
            transform=SSLTransform(), 
            cohort_selection_options=CohortSelectionOptions(
                fold=config.fold, n_folds=config.n_folds,
                min_involvement=None, 
                remove_benign_from_positive_patients=False, 
                benign_to_cancer_ratio=config.benign_to_cancer_ratio_ssl
            ), 
            patch_options=patch_options_ssl
        )
    )

    class SLTransform: 
        def __call__(self, item): 
            patch = item['patch']
            patch = torch.from_numpy(patch).float()
            patch = patch.unsqueeze(0)
            patch = (patch - patch.mean()) / patch.std()

            label = torch.tensor(item['grade'] != "Benign").long() 
            involvement = torch.tensor(item['involvement']).float() / 100.0
            core_id = torch.tensor(item['id']).long()

            return patch, label, involvement, core_id

    patch_options_sl = PatchOptions(
        prostate_mask_threshold=0.8,
    )
    sl_dataset_train = ExactNCT2013BmodePatches(
        transform=SLTransform(), 
        cohort_selection_options=CohortSelectionOptions(
            fold=config.fold, n_folds=config.n_folds,
            min_involvement=40, 
            remove_benign_from_positive_patients=True, 
            benign_to_cancer_ratio=2
        ),
        patch_options=patch_options_sl
    )
    sl_dataset_val = ExactNCT2013BmodePatches(
        transform=SLTransform(), 
        cohort_selection_options=CohortSelectionOptions(
            fold=config.fold, n_folds=config.n_folds,
            min_involvement=None, 
            remove_benign_from_positive_patients=False, 
            benign_to_cancer_ratio=None
        ),
        patch_options=patch_options_sl
    )
    sl_dataset_test = ExactNCT2013BmodePatches(
        transform=SLTransform(), 
        cohort_selection_options=CohortSelectionOptions(
            fold=config.fold, n_folds=config.n_folds,
            min_involvement=None, 
            remove_benign_from_positive_patients=False, 
            benign_to_cancer_ratio=None
        ),
        patch_options=patch_options_sl
    )

    return ssl_dataset, sl_dataset_train, sl_dataset_val, sl_dataset_test


class Config: 
    ...


class Main:
    def __call__(self, config: Config): 
        ...
    

if __name__ == "__main__": 
    output = get_dataloaders(DataConfig())
    breakpoint()