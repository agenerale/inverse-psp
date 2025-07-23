import argparse
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar

from models.cnf import MLP, ForwardModel, FlowModel, sample_model, compute_log_likelihood, base_log_prob
from helpers.function_helpers import load_data, eval_mae
from helpers.plotting_helpers import plot_trajectories, plot_corner_theta, plot_corner_prop

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CNFPipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = self._configure_device()
        self.model = None
        self.foward_model = None
        
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load and validate configuration"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        required_sections = ['train', 'model', 'paths', 'evaluation']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
                
        return config

    def _configure_device(self):
        """Configure hardware device"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_models(self):
        """Initialize model components"""
        # Initialize Forward Models
        
        if 'model_2' in self.config['paths'].keys():
            fnames_model1 = {
                'likelihood': self.config['paths']['likelihood_1'],
                'model': self.config['paths']['model_1']
            }
            fnames_model2 = {
                'likelihood': self.config['paths']['likelihood_2'],
                'model': self.config['paths']['model_2']
            }
            
            self.forward_model = ForwardModel(fnames_model1, fnames_model2).to(self.device)
        else:
            fnames_model1 = {
                'likelihood': self.config['paths']['likelihood_1'],
                'model': self.config['paths']['model_1']
            }
            
            self.forward_model = ForwardModel(fnames_model1).to(self.device)            
        
        # Initialize CNF
        mlp = MLP(
            self.config['model']['ndim'],
            self.config['model']['cdim'],
            self.config['model']['edim'],
            layers=self.config['model']['layers'],
            w=self.config['model']['width'],
            #w_embed=self.config['model']['w_embed'],
            num_heads=self.config['model']['n_heads'],
            dropout=self.config['model']['dropout']
        )
        
        self.model = FlowModel(mlp, self.forward_model, self.config).to(self.device)

    def load_model(self, model_path: str):
        """Load model from checkpoint"""
        logger.info(f"Loading model from {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device)
            logger.info(f"Model loaded to {self.device}")
        except Exception as e:
            logger.error(f"Model load failed: {str(e)}")
            raise

    def save_model(self, save_path: str):
        """Save model to checkpoint"""
        logger.info(f"Saving model to {save_path}")
        
        try:
            torch.save({
                'state_dict': self.model.state_dict(),
                'config': self.config
            }, save_path)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    def train(self):
        """Execute full training pipeline"""
        logger.info("Initializing training")
        
        trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            max_epochs=self.config['train']['n_epoch'],
            default_root_dir="logs/",
            gradient_clip_val=self.config['train']['clip'],
            callbacks=[RichProgressBar()]
        )
        
        try:
            trainer.fit(self.model)
            save_path = Path(self.config['paths']['output_dir'])
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)
            self.save_model(save_path / "model.ckpt")
            logger.info(f"Training completed successfully. Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def evaluate_model(self, evaluate_all=False):
        """Run full evaluation pipeline"""
        logger.info("Starting model evaluation")
        
        # Load data
        data = load_data(self.config['paths']['data'], device=self.device)
            
        # Create output directory
        output_dir = Path(self.config['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        imgs_subdir = output_dir / "imgs"
        imgs_subdir.mkdir(parents=True, exist_ok=True)
        
        if evaluate_all:
            
            train_metrics = self._eval_group(data, output_dir, imgs_subdir, mode='train')
            test_metrics = self._eval_group(data, output_dir, imgs_subdir, mode='test')
            
            results = {
                'train_marginal_log_likelihood': train_metrics[0],
                'train_mae': train_metrics[1:self.config['model']['cdim']+1],
                'train_nmae': train_metrics[self.config['model']['cdim']+1:],
                'test_marginal_log_likelihood': test_metrics[0],
                'test_mae': test_metrics[1:self.config['model']['cdim']+1],
                'test_nmae': test_metrics[self.config['model']['cdim']+1:],
            }
            
            self._save_results(results, output_dir, evaluate_all)
        else:
            # Run select evaluations
            results = []
            for indx in self.config['evaluation']['evalindices']:
                try:
                    # Get result from _eval_micro and append to list
                    result = self._eval_indx(
                        indx, data,
                        output_dir, imgs_subdir,
                        plot=True
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed evaluation for sample {indx}: {str(e)}")
        
            self._save_results(results, output_dir)
            
    def _eval_group(self, data, output_dir, imgs_subdir, mode='train'):
        
        metrics = []
        for indx in range(data.shape[0]):
            try:
                # Get result from _eval_micro and append to list
                result = self._eval_indx(
                    indx,
                    data,
                    output_dir, imgs_subdir,
                    plot=False,
                    mode=mode,
                )
                metrics.append(
                    np.hstack([result['log_likelihood'],
                                result['mae'],
                                result['nmae']])
                                )
            except Exception as e:
                logger.error(f"Failed test evaluation for sample {indx}: {str(e)}")

        metrics = list(np.mean(np.stack(metrics), axis=0))               
    
        return metrics
    
    def _eval_indx(self, indx, data, output_dir, imgs_subdir, plot=False, mode='train'):
        """Evaluate and return results for a single sample"""
        logger.info(f"Processing sample {indx}")
        
        y_target = data[f'output_{mode}'][indx,:]
        x_target = data[f'input_{mode}'][indx,:]
        
        # Generate samples
        traj = sample_model(
            self.model, y_target, 
            self.config['model']['cdim'],
            self.config['model']['ndim'],
            n_sample=self.config['evaluation']['n_eval_samples'],
            n_traj=self.config['evaluation']['n_traj_pts']
        )
        
        if plot:
            # Save plots        
            plot_trajectories(traj, data['input_scaler'], indx, x_target, save_path=imgs_subdir)
            plot_corner_theta(traj, data['input_scaler'], indx, x_target, save_path=imgs_subdir)
            plot_corner_prop(traj, self.forward_model, y_target, data['output_scaler'], indx, save_path=imgs_subdir)
        
        # Compute log likelihood
        log_prob = compute_log_likelihood(
            x=data[f'input_{mode}'][indx,:].unsqueeze(0).to(self.device),
            y=y_target.unsqueeze(0).to(self.device),
            flow_field=self.model.model,
            base_log_prob=base_log_prob,
            method=self.config['evaluation']['trace_method']
        )
        
        mae, nmae = eval_mae(traj, self.forward_model, y_target, data['output_scaler'])
                
        return {
            'sample_id': indx,
            'log_likelihood': log_prob.item(),
            'mae': mae.tolist()[0],
            'nmae': nmae.tolist()[0],
        }
    

    def _save_results(self, all_results, output_dir, evaluate_all=False):
        """Save numerical results"""
        
        results = {
            **({"indices_evaluated": self.config['evaluation']['evalindices']} if not evaluate_all else {}),
            "results": all_results,
            "config": self.config,
            "evaluation_date": datetime.now().isoformat()
        }
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="CNF Pipeline")
    parser.add_argument('--mode', default='eval', type=str,
                      choices=['train', 'eval'],
                      help='Operation mode: train or evaluate')
    parser.add_argument('--config', default='./config/config.yml', type=str,
                      help='Path to config YAML file')
    parser.add_argument('--model_path', default='./results/model.ckpt', type=str,
                      help='Path to saved model for evaluation')
    args = parser.parse_args()

    if args.mode == 'eval' and not args.model_path:
        parser.error("--model_path required in evaluate mode")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    pipeline = CNFPipeline(args.config)
    
    if args.mode == 'train':
        pipeline.initialize_models()
        pipeline.train()
    elif args.mode == 'eval':
        pipeline.initialize_models()
        pipeline.load_model(args.model_path)
        pipeline.evaluate_model(evaluate_all=False)