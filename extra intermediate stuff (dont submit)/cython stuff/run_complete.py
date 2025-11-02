#!/usr/bin/env python3
"""
Run all algorithms on all problems
Budget: 100,000 evaluations per run
Runs: 30 independent runs per problem
Problems: 2100-2103, 2200-2203
"""

import complete_suite
import time

if __name__ == "__main__":
    print("="*60)
    print("COMPLETE ALGORITHM SUITE")
    print("="*60)
    print(f"Budget: {complete_suite.BUDGET:,} evaluations")
    print(f"Runs: {complete_suite.RUNS} per problem")
    print(f"Problems: {complete_suite.PROBLEMS}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        complete_suite.run_all_algorithms()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print(f"\n{'='*60}")
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"{'='*60}")
    print("\n✅ Upload 'complete_logs/' folder to IOHanalyzer!")
