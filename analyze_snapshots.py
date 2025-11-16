#!/usr/bin/env python3
"""
Script to analyze model snapshots saved during training
Specifically focuses on attention parameters and loss spikes
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime
import pandas as pd

def load_snapshots_from_dir(snapshot_dir="./model_snapshots/"):
    """Load all snapshots from directory and sort by step"""
    snapshot_files = glob.glob(os.path.join(snapshot_dir, "*.pt"))
    
    snapshots = []
    for file_path in snapshot_files:
        try:
            checkpoint = torch.load(file_path, map_location='cpu')
            checkpoint['file_path'] = file_path
            checkpoint['file_name'] = os.path.basename(file_path)
            snapshots.append(checkpoint)
        except Exception as e:
            print(f"⚠️  Could not load {file_path}: {e}")
    
    # Sort by step
    snapshots.sort(key=lambda x: x.get('step', 0))
    return snapshots

def analyze_attention_parameters(snapshots):
    """Analyze attention parameters across snapshots"""
    
    attention_params = [
        'decoder.attention.W_p.weight', 
        'decoder.attention.v_p.weight', 
        'decoder.attention.W_a.weight', 
        'decoder.attention.W_c.weight'
    ]
    
    analysis_data = []
    
    for snapshot in snapshots:
        step = snapshot.get('step', 0)
        epoch = snapshot.get('epoch', 0)
        loss = snapshot.get('loss', 0)
        snapshot_type = snapshot.get('snapshot_type', 'unknown')
        
        param_stats = {
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'snapshot_type': snapshot_type,
            'file_name': snapshot['file_name']
        }
        
        state_dict = snapshot['model_state_dict']
        
        for param_name in attention_params:
            if param_name in state_dict:
                param_tensor = state_dict[param_name]
                
                # Calculate statistics
                param_stats[f'{param_name}_norm'] = torch.norm(param_tensor).item()
                param_stats[f'{param_name}_mean'] = torch.mean(param_tensor).item()
                param_stats[f'{param_name}_std'] = torch.std(param_tensor).item()
                param_stats[f'{param_name}_min'] = torch.min(param_tensor).item()
                param_stats[f'{param_name}_max'] = torch.max(param_tensor).item()
        
        analysis_data.append(param_stats)
    
    return pd.DataFrame(analysis_data)

def check_position_prediction(snapshots):
    """Analyze position prediction behavior"""
    print("🎯 Position Prediction Analysis")
    print("=" * 50)
    
    for snapshot in snapshots:
        step = snapshot.get('step', 0)
        loss = snapshot.get('loss', 0)
        snapshot_type = snapshot.get('snapshot_type', 'unknown')
        state_dict = snapshot['model_state_dict']
        
        print(f"\nStep {step} ({snapshot_type}) - Loss: {loss:.4f}")
        
        if 'decoder.attention.v_p.weight' in state_dict and 'decoder.attention.W_p.weight' in state_dict:
            v_p = state_dict['decoder.attention.v_p.weight']
            W_p = state_dict['decoder.attention.W_p.weight']
            
            print(f"  v_p range: [{v_p.min():.4f}, {v_p.max():.4f}], var: {v_p.var():.6f}")
            
            # Simulate position prediction for source length 7 (like in your examples)
            dummy_input = torch.randn(32, 1000)  # batch_size=32, hidden_dim=1000
            with torch.no_grad():
                pos_intermediate = torch.tanh(torch.matmul(dummy_input, W_p.T))
                pos_output = torch.sigmoid(torch.matmul(pos_intermediate, v_p.T))
                # Scale to source length
                positions = 7 * pos_output  # Assuming source length 7
                
                print(f"  Predicted positions range: [{positions.min():.4f}, {positions.max():.4f}]")
                print(f"  Position variance: {positions.var():.6f}")
                print(f"  Mean position: {positions.mean():.4f}")
                
                # Check for position collapse
                if positions.var() < 0.001:
                    print("  ⚠️  POSITION COLLAPSE DETECTED!")
                elif positions.var() > 1.0:
                    print("  ✅ Good position diversity")
                else:
                    print("  ⚡ Moderate position diversity")

def plot_parameter_evolution(df, save_path="./snapshot_analysis.png"):
    """Create plots showing parameter evolution"""
    
    # Focus on position parameters
    position_params = [col for col in df.columns if 'v_p' in col or 'W_p' in col]
    
    if not position_params:
        print("⚠️  No position parameters found in data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Attention Parameter Evolution During Training', fontsize=16)
    
    # Plot 1: v_p weight statistics
    ax1 = axes[0, 0]
    if 'decoder.attention.v_p.weight_norm' in df.columns:
        ax1.plot(df['step'], df['decoder.attention.v_p.weight_norm'], 'bo-', markersize=4)
        ax1.set_title('v_p Weight Norm Evolution')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('L2 Norm')
        ax1.grid(True, alpha=0.3)
        
        # Highlight critical steps
        critical_steps = df[df['snapshot_type'] == 'critical_step']
        if not critical_steps.empty:
            ax1.scatter(critical_steps['step'], critical_steps['decoder.attention.v_p.weight_norm'], 
                       color='red', s=50, alpha=0.7, label='Critical Steps')
            ax1.legend()
    
    # Plot 2: v_p variance
    ax2 = axes[0, 1] 
    if 'decoder.attention.v_p.weight_std' in df.columns:
        ax2.plot(df['step'], df['decoder.attention.v_p.weight_std'], 'go-', markersize=4)
        ax2.set_title('v_p Weight Standard Deviation')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Std Dev')
        ax2.grid(True, alpha=0.3)
        
        # Mark loss spikes
        loss_spikes = df[df['snapshot_type'] == 'loss_spike']
        if not loss_spikes.empty:
            ax2.scatter(loss_spikes['step'], loss_spikes['decoder.attention.v_p.weight_std'],
                       color='orange', s=50, alpha=0.7, label='Loss Spikes')
            ax2.legend()
    
    # Plot 3: Loss evolution with snapshots
    ax3 = axes[1, 0]
    ax3.plot(df['step'], df['loss'], 'ko-', markersize=3, alpha=0.6)
    ax3.set_title('Loss Evolution with Snapshots')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    
    # Color by snapshot type
    snapshot_types = df['snapshot_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(snapshot_types)))
    for snapshot_type, color in zip(snapshot_types, colors):
        subset = df[df['snapshot_type'] == snapshot_type]
        if not subset.empty:
            ax3.scatter(subset['step'], subset['loss'], 
                       color=color, label=snapshot_type, s=30, alpha=0.8)
    ax3.legend(loc='upper right', fontsize=8)
    
    # Plot 4: Position parameter correlation
    ax4 = axes[1, 1]
    if 'decoder.attention.v_p.weight_norm' in df.columns and 'decoder.attention.W_p.weight_norm' in df.columns:
        scatter = ax4.scatter(df['decoder.attention.v_p.weight_norm'], 
                             df['decoder.attention.W_p.weight_norm'],
                             c=df['step'], cmap='viridis', alpha=0.7)
        ax4.set_title('v_p vs W_p Weight Norms')
        ax4.set_xlabel('v_p Norm')
        ax4.set_ylabel('W_p Norm')
        plt.colorbar(scatter, ax=ax4, label='Training Step')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Analysis plot saved to: {save_path}")

def compare_critical_snapshots(snapshots, critical_steps=[40, 50, 60]):
    """Compare snapshots at critical steps"""
    print(f"\n🔍 Comparing Critical Snapshots at steps: {critical_steps}")
    print("=" * 60)
    
    critical_snapshots = [s for s in snapshots if s.get('step', 0) in critical_steps]
    
    if len(critical_snapshots) < 2:
        print("⚠️  Not enough critical snapshots found for comparison")
        return
    
    # Compare position parameters
    for i, snapshot in enumerate(critical_snapshots):
        step = snapshot.get('step', 0)
        loss = snapshot.get('loss', 0)
        state_dict = snapshot['model_state_dict']
        
        print(f"\nSnapshot {i+1}: Step {step}, Loss: {loss:.4f}")
        
        if 'decoder.attention.v_p.weight' in state_dict:
            v_p = state_dict['decoder.attention.v_p.weight']
            print(f"  v_p: min={v_p.min():.4f}, max={v_p.max():.4f}, var={v_p.var():.6f}")
            
        if 'decoder.attention.W_a.weight' in state_dict:
            W_a = state_dict['decoder.attention.W_a.weight']
            print(f"  W_a: norm={torch.norm(W_a):.4f}, mean={torch.mean(W_a):.4f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model snapshots")
    parser.add_argument("--snapshot_dir", "-sd", default="./model_snapshots/",
                       help="Directory containing model snapshots")
    parser.add_argument("--output_plot", "-op", default="./snapshot_analysis.png",
                       help="Path to save analysis plot")
    parser.add_argument("--critical_steps", "-cs", nargs='+', type=int, 
                       default=[40, 50, 60], help="Critical steps to analyze")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.snapshot_dir):
        print(f"❌ Snapshot directory not found: {args.snapshot_dir}")
        return
    
    print(f"🔍 Loading snapshots from: {args.snapshot_dir}")
    snapshots = load_snapshots_from_dir(args.snapshot_dir)
    
    if not snapshots:
        print("❌ No snapshots found!")
        return
    
    print(f"✅ Loaded {len(snapshots)} snapshots")
    
    # Analyze attention parameters
    df = analyze_attention_parameters(snapshots)
    print(f"📊 Parameter analysis complete")
    
    # Check position prediction
    check_position_prediction(snapshots)
    
    # Plot evolution
    plot_parameter_evolution(df, args.output_plot)
    
    # Compare critical snapshots  
    compare_critical_snapshots(snapshots, args.critical_steps)
    
    # Save analysis data
    csv_path = args.output_plot.replace('.png', '_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"💾 Analysis data saved to: {csv_path}")
    
    print("\n🎉 Snapshot analysis complete!")
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"   Total snapshots: {len(snapshots)}")
    print(f"   Step range: {df['step'].min()} - {df['step'].max()}")
    print(f"   Snapshot types: {df['snapshot_type'].unique()}")

if __name__ == "__main__":
    main()