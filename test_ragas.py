"""
Run RAGAS evaluation
Usage: python test_ragas.py
"""
from src.evaluation.ragas_eval import run_evaluation, print_results

results = run_evaluation()
df = print_results(results)

# Save results to CSV
df.to_csv("ragas_results.csv", index=False)
print("\nResults saved to ragas_results.csv")