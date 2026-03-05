import pandas as pd
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
    
class Logger:
    """
    A unified logger interface for swanlab and wandb`
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.logger_type = config.logger.type
        self.logger = None
        
        config_dict = OmegaConf.to_container(config, resolve=True)
        
        init_kwargs = {
            "project": config.logger.project_name,
            "config": config_dict,
            "reinit": True,
        }
        
        if hasattr(config.logger, "run_name") and config.logger.run_name is not None:
            init_kwargs["name"] = config.logger.run_name
            
        if self.logger_type == "swanlab":
            import swanlab
            swanlab.init(**init_kwargs)  # ty:ignore[invalid-argument-type]
            self.logger = swanlab
        elif self.logger_type == "wandb":
            import wandb
            wandb.init(**init_kwargs) # ty:ignore[invalid-argument-type]
            self.logger = wandb
        else:
            print(f"Unknown logger type '{self.logger_type}'. Logging will be disabled.")
            self.logger_type = None
            self.logger = None
        
    def log_metrics(self, metrics: dict, step: int):
        if self.logger_type is None or self.logger is None:
            return
        if self.logger_type in ["wandb", "swanlab"]:
            return self.logger.log(metrics, step=step)
    
    def log_text(self, key: str, text: str, step: int):
        if self.logger_type is None or self.logger is None:
            return
        if "wandb" == self.logger_type:
            self.logger.log({key: self.logger.Html(text)}, step=step)  # ty:ignore[possibly-missing-attribute]
        elif "swanlab" == self.logger_type:
            self.logger.log({key: self.logger.Text(text)}, step=step)  # ty:ignore[possibly-missing-attribute]
    
    def log_table(self, key: str, table: dict):
        if "wandb" == self.logger_type and self.logger is not None:
            self.logger.log({key: self.logger.Table(dataframe=pd.DataFrame(table))})  # ty:ignore[possibly-missing-attribute]
        elif "swanlab" == self.logger_type and self.logger is not None:
            etable = self.logger.echarts.Table()  # ty:ignore[possibly-missing-attribute]
            etable.add(
                list(table.keys()),
                [list(row) for row in zip(*table.values())],
            )
            self.logger.log({key: etable})
    
    def close(self):
        if self.logger is not None and self.logger_type in ["wandb", "swanlab"]:
            self.logger.finish()
        
            
        
        