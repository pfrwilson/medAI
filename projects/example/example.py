import medAI
from dataclasses import dataclass
import logging
import wandb


@dataclass
class MyConfig(medAI.utils.setup.BasicExperimentConfig):
    """Configuration for the experiment."""
    project: str = "example"
    epochs: int = 10 


class MyExperiment(medAI.utils.setup.BasicExperiment): 
    config_class = MyConfig
    config: MyConfig

    def setup(self): 
        super().setup()
        from torch.utils.data import DataLoader
        self.train_loader = DataLoader(range(100), batch_size=10, shuffle=True)

    def __call__(self): 
        self.setup()
        for epoch in range(self.config.epochs): 
            logging.info(f"Epoch {epoch}")
            for batch in self.train_loader: 
                wandb.log({"sum": sum(batch)})


if __name__ == '__main__': 
    MyExperiment.submit()