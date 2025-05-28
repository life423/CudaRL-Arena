#!/usr/bin/env python3
"""
Demo script showing the complete training pipeline with CPU fallback.
"""

import sys
import numpy as np

def test_training_pipeline():
    """Test the complete training pipeline."""
    try:
        print("=== CudaRL-Arena Training Demo ===\n")
        
        # Import the package
        import python.cudarl as cudarl
        print(f"✓ CudaRL-Arena {cudarl.__version__} loaded")
        print(f"  CUDA Available: {cudarl.CUDA_AVAILABLE}")
        print()
        
        # Create environment
        env = cudarl.Environment(width=6, height=6)
        print(f"✓ Created {env.width}x{env.height} environment")
        
        # Create a simple agent
        agent = cudarl.QTableAgent(
            action_space_size=4,
            observation_shape=(env.height, env.width),
            learning_rate=0.1,
            exploration_rate=0.9
        )
        print("✓ Created Q-learning agent")
        
        # Create trainer
        trainer = cudarl.Trainer(env, agent)
        print(f"✓ Created trainer")
        print()
        
        # Run short training
        print("Starting training...")
        metrics = trainer.train(
            num_episodes=50,
            max_steps_per_episode=100,
            verbose=True
        )
        
        print(f"\n✓ Training completed!")
        print(f"  Total episodes: {len(metrics['rewards'])}")
        print(f"  Average reward (last 10): {np.mean(metrics['rewards'][-10:]):.3f}")
        print(f"  Average length (last 10): {np.mean(metrics['lengths'][-10:]):.1f}")
        
        # Quick evaluation
        print("\nRunning evaluation...")
        eval_results = trainer.evaluate(num_episodes=3, render=False)
        print(f"✓ Evaluation completed!")
        print(f"  Average reward: {eval_results['avg_reward']:.3f}")
        print(f"  Average length: {eval_results['avg_length']:.1f}")
        
        # Show final state
        print("\n--- Final Environment State ---")
        env.render(mode='human')
        
        print("\n✅ Complete pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_pipeline()
    sys.exit(0 if success else 1)
