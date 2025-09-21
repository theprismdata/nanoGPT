"""
MLflow utilities for nanoGPT training
"""

import os
import mlflow
import mlflow.pytorch
from typing import Optional, Dict, Any


def setup_mlflow_experiment(experiment_name: str, run_name: Optional[str] = None) -> str:
    """
    Setup MLflow experiment and start a new run
    
    Args:
        experiment_name: Name of the experiment
        run_name: Optional name for the run (auto-generated if None)
    
    Returns:
        run_id: The ID of the started run
    """
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Start run
    if run_name is None:
        import time
        run_name = f"run_{int(time.time())}"
    
    run = mlflow.start_run(run_name=run_name)
    return run.info.run_id


def log_training_config(config: Dict[str, Any]) -> None:
    """
    Log training configuration parameters to MLflow
    
    Args:
        config: Dictionary containing training configuration
    """
    # Log all config parameters
    mlflow.log_params(config)
    
    # Log additional metadata
    mlflow.log_param("framework", "PyTorch")
    mlflow.log_param("model_type", "GPT")


def log_model_checkpoint(model, checkpoint_path: str, model_name: str = "nanoGPT") -> None:
    """
    Log model checkpoint to MLflow
    
    Args:
        model: The trained model
        checkpoint_path: Path to the checkpoint file
        model_name: Name for the registered model
    """
    # Log the model
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=model_name
    )
    
    # Log checkpoint as artifact
    mlflow.log_artifact(checkpoint_path, "checkpoints")


def log_training_metrics(metrics: Dict[str, float], step: int) -> None:
    """
    Log training metrics to MLflow
    
    Args:
        metrics: Dictionary of metrics to log
        step: Training step number
    """
    mlflow.log_metrics(metrics, step=step)


def get_best_model_from_experiment(experiment_name: str, metric_name: str = "val_loss", 
                                 ascending: bool = True) -> Optional[str]:
    """
    Get the best model run from an experiment based on a metric
    
    Args:
        experiment_name: Name of the experiment
        metric_name: Metric to use for comparison
        ascending: Whether lower values are better (True for loss, False for accuracy)
    
    Returns:
        run_id of the best model, or None if no runs found
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return None
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if runs.empty:
            print(f"No runs found in experiment '{experiment_name}'")
            return None
        
        # Filter runs that have the metric
        runs_with_metric = runs[runs[f"metrics.{metric_name}"].notna()]
        if runs_with_metric.empty:
            print(f"No runs with metric '{metric_name}' found")
            return None
        
        # Get best run
        if ascending:
            best_run = runs_with_metric.loc[runs_with_metric[f"metrics.{metric_name}"].idxmin()]
        else:
            best_run = runs_with_metric.loc[runs_with_metric[f"metrics.{metric_name}"].idxmax()]
        
        return best_run['run_id']
    
    except Exception as e:
        print(f"Error getting best model: {e}")
        return None


def load_model_from_run(run_id: str, model_path: str = "model") -> Any:
    """
    Load a model from a specific MLflow run
    
    Args:
        run_id: ID of the MLflow run
        model_path: Path to the model within the run artifacts
    
    Returns:
        Loaded model
    """
    model_uri = f"runs:/{run_id}/{model_path}"
    return mlflow.pytorch.load_model(model_uri)


def list_experiments() -> None:
    """List all MLflow experiments"""
    experiments = mlflow.search_experiments()
    print("Available experiments:")
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")


def list_runs_in_experiment(experiment_name: str) -> None:
    """List all runs in a specific experiment"""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if runs.empty:
            print(f"No runs found in experiment '{experiment_name}'")
            return
        
        print(f"Runs in experiment '{experiment_name}':")
        for _, run in runs.iterrows():
            print(f"  - {run['run_id']}: {run.get('tags.mlflow.runName', 'unnamed')}")
            if 'metrics.val_loss' in run and not pd.isna(run['metrics.val_loss']):
                print(f"    Val Loss: {run['metrics.val_loss']:.4f}")
    
    except Exception as e:
        print(f"Error listing runs: {e}")


if __name__ == "__main__":
    # Example usage
    print("MLflow utilities for nanoGPT")
    print("Available functions:")
    print("  - setup_mlflow_experiment()")
    print("  - log_training_config()")
    print("  - log_model_checkpoint()")
    print("  - log_training_metrics()")
    print("  - get_best_model_from_experiment()")
    print("  - load_model_from_run()")
    print("  - list_experiments()")
    print("  - list_runs_in_experiment()")
