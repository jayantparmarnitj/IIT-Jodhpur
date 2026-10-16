"""
Noise Schedule Visualization and Analysis
Generates plots comparing native vs. ablated schedules for the report.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from EXPONENT import NOISE_SCHEDULE_EXPONENT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ScheduleVisualizer:
    """
    Visualize native vs. ablated noise schedules.
    """
    
    def __init__(self, exponent: float = NOISE_SCHEDULE_EXPONENT, num_timesteps: int = 1000):
        self.exponent = exponent
        self.num_timesteps = num_timesteps
        
        # ADDED: Compute schedules
        beta_start = 0.00085
        beta_end = 0.0120
        
        # ADDED: Native schedule
        betas = torch.linspace(
            np.sqrt(beta_start), np.sqrt(beta_end), num_timesteps, dtype=torch.float32
        ) ** 2
        self.betas_native = betas.numpy()
        
        # ADDED: Ablated schedule
        self.betas_ablated = (betas ** exponent).numpy()
        
        # ADDED: Cumulative quantities
        alphas_native = 1.0 - self.betas_native
        alphas_ablated = 1.0 - self.betas_ablated
        
        self.alphas_cumprod_native = np.cumprod(alphas_native)
        self.alphas_cumprod_ablated = np.cumprod(alphas_ablated)
        
        # ADDED: SNR
        self.snr_native = self.alphas_cumprod_native / (1.0 - self.alphas_cumprod_native)
        self.snr_ablated = self.alphas_cumprod_ablated / (1.0 - self.alphas_cumprod_ablated)
        
        # ADDED: Timestep axis
        self.timesteps = np.arange(1, num_timesteps + 1)
    
    def plot_schedule_comparison(self, output_dir: Path) -> None:
        """Generate comprehensive schedule comparison plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ADDED: Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Noise Schedule Comparison: Native vs. Ablated (x={self.exponent})', fontsize=16)
        
        # ADDED: Plot 1: β_t values
        ax = axes[0, 0]
        ax.plot(self.timesteps, self.betas_native, label='Native {β_t}', linewidth=2)
        ax.plot(self.timesteps, self.betas_ablated, label=f'Ablated {{β_t^{self.exponent}}}', linewidth=2)
        ax.set_xlabel('Diffusion Timestep t')
        ax.set_ylabel('Variance β_t')
        ax.set_title('Per-step Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # ADDED: Plot 2: α_bar,t (cumulative retention)
        ax = axes[0, 1]
        ax.plot(self.timesteps, self.alphas_cumprod_native, label='Native ᾱ_t', linewidth=2)
        ax.plot(self.timesteps, self.alphas_cumprod_ablated, label=f"Ablated ᾱ_t'", linewidth=2)
        ax.set_xlabel('Diffusion Timestep t')
        ax.set_ylabel('Signal Retention ᾱ_t')
        ax.set_title('Cumulative Signal Retention')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.985, color='r', linestyle='--', alpha=0.5, label='Ablated at t=1000: 0.985')
        
        # ADDED: Plot 3: SNR comparison
        ax = axes[1, 0]
        # MODIFIED: Log scale for SNR to see both high and low values
        ax.semilogy(self.timesteps, self.snr_native, label='Native SNR', linewidth=2)
        ax.semilogy(self.timesteps, self.snr_ablated, label=f'Ablated SNR (x={self.exponent})', linewidth=2)
        ax.set_xlabel('Diffusion Timestep t')
        ax.set_ylabel('Signal-to-Noise Ratio SNR(t)')
        ax.set_title('SNR Over Diffusion Steps')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        ax.axvline(x=1000, color='k', linestyle=':', alpha=0.3, label='t=T (pure noise)')
        ax.axvline(x=1, color='k', linestyle='-.', alpha=0.3, label='t=1 (signal-rich)')
        
        # ADDED: Plot 4: Noise scale (√(1 - ᾱ_t))
        ax = axes[1, 1]
        noise_scale_native = np.sqrt(1.0 - self.alphas_cumprod_native)
        noise_scale_ablated = np.sqrt(1.0 - self.alphas_cumprod_ablated)
        ax.plot(self.timesteps, noise_scale_native, label='Native noise scale', linewidth=2)
        ax.plot(self.timesteps, noise_scale_ablated, label=f"Ablated noise scale (x={self.exponent})", linewidth=2)
        ax.set_xlabel('Diffusion Timestep t')
        ax.set_ylabel('Noise Scale √(1 - ᾱ_t)')
        ax.set_title('Noise Contribution Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ADDED: Save figure
        plt.tight_layout()
        output_file = output_dir / 'schedule_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved schedule comparison to {output_file}")
        plt.close()
    
    def plot_distributional_mismatch(self, output_dir: Path) -> None:
        """Illustrate the distributional mismatch at t=T."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ADDED: Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Distributional Mismatch Hypothesis: Inference vs. Training', fontsize=14)
        
        # ADDED: Highlight critical timesteps
        critical_timesteps = [1, 250, 500, 750, 1000]
        
        # Plot 1: Training regime
        ax = axes[0]
        ax.fill_between(self.timesteps, 0, self.alphas_cumprod_native, 
                        label='Run A: Signal retained during training', alpha=0.3, color='blue')
        ax.plot(self.timesteps, self.alphas_cumprod_native, 'b-', linewidth=2)
        
        ax.fill_between(self.timesteps, 0, self.alphas_cumprod_ablated, 
                        label=f"Run B: Signal retained during training (x={self.exponent})", alpha=0.3, color='red')
        ax.plot(self.timesteps, self.alphas_cumprod_ablated, 'r-', linewidth=2)
        
        # Mark critical timesteps
        for t in critical_timesteps:
            idx = t - 1
            ax.scatter([t], [self.alphas_cumprod_native[idx]], color='blue', s=50, zorder=5)
            ax.scatter([t], [self.alphas_cumprod_ablated[idx]], color='red', s=50, zorder=5)
        
        ax.set_xlabel('Diffusion Timestep t')
        ax.set_ylabel('Signal Retention ᾱ_t')
        ax.set_title('Training: Forward Process\n(What the model learned)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Inference regime (both start from pure noise)
        ax = axes[1]
        
        # Pure noise start point
        ax.axhline(y=0, color='green', linewidth=3, label='Inference start: x_T ~ N(0,I)', alpha=0.7)
        
        # Highlight mismatch zone for Run B
        ax.fill_between([900, 1000], 0, 1, alpha=0.2, color='red', 
                        label=f"Run B mismatch zone\n(trained on α_bar≈0.985, now α_bar≈0)")
        
        # Highlight matching zone for Run A
        ax.fill_between([900, 1000], 0, 1, alpha=0.2, color='blue', 
                        label="Run A match zone\n(trained on α_bar≈0.0047)")
        
        ax.set_xlim(800, 1050)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('Diffusion Timestep t')
        ax.set_ylabel('Signal Retention ᾱ_t')
        ax.set_title('Inference: Reverse Process Start\n(Pure noise for both runs)')
        ax.legend(loc='upper left')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Pure Noise\n(α_bar=0)', 'All Signal\n(α_bar=1)'])
        
        # ADDED: Save figure
        plt.tight_layout()
        output_file = output_dir / 'distributional_mismatch.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved mismatch illustration to {output_file}")
        plt.close()
    
    def print_schedule_table(self) -> None:
        """Print schedule comparison table for report."""
        logger.info("\n" + "="*100)
        logger.info("NOISE SCHEDULE COMPARISON TABLE")
        logger.info("="*100)
        
        print(f"{'t':<6} {'α_bar (Native)':<20} {'α_bar (x={self.exponent})':<20} "
              f"{'SNR Native':<15} {'SNR Modified':<15}")
        print("-" * 100)
        
        for t in [1, 250, 500, 750, 1000]:
            idx = t - 1
            snr_native = self.snr_native[idx]
            snr_ablated = self.snr_ablated[idx]
            
            print(f"{t:<6} {self.alphas_cumprod_native[idx]:<20.6f} {self.alphas_cumprod_ablated[idx]:<20.6f} "
                  f"{snr_native:<15.4f} {snr_ablated:<15.4f}")
        
        logger.info("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize noise schedules")
    parser.add_argument('--exponent', type=float, default=NOISE_SCHEDULE_EXPONENT,
                       help="Noise schedule exponent")
    parser.add_argument('--output_dir', type=str, default='./schedule_plots',
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # ADDED: Create visualizer
    visualizer = ScheduleVisualizer(exponent=args.exponent)
    
    # ADDED: Generate plots
    visualizer.plot_schedule_comparison(Path(args.output_dir))
    visualizer.plot_distributional_mismatch(Path(args.output_dir))
    visualizer.print_schedule_table()
    
    logger.info(f"Visualizations saved to {args.output_dir}")


if __name__ == '__main__':
    main()
