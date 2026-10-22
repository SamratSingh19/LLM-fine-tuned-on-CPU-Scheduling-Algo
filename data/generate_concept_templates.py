
import json

ALGORITHMS = {
    "fcfs": "First Come, First Served (FCFS)",
    "sjf": "Shortest Job First (SJF)",
    "srtf": "Shortest Remaining Time First (SRTF)",
    "priority": "Priority Scheduling",
    "rr": "Round Robin (RR)",
    "mlq": "Multilevel Queue",
    "mlfq": "Multilevel Feedback Queue (MLFQ)"
}

QUESTION_TYPES = [
    "Explain {algo} to a beginner in simple terms.",
    "What are the advantages of {algo}?",
    "What are the disadvantages or drawbacks of {algo}?",
    "Give a real-life, everyday analogy for how {algo} works.",
    "How is waiting time and turnaround time calculated in {algo}?",
    "In what situations is {algo} the best choice, and when is it a poor choice?",
    "What is a common mistake students make when solving {algo} numericals?",
    "What viva question might be asked about {algo}, and how would you answer it?"
]

CROSS_CUTTING = [
    "What is starvation in CPU scheduling, and which algorithms are most prone to it?",
    "What is the convoy effect, and which algorithm suffers from it most?",
    "What is context-switch overhead, and how does it affect Round Robin specifically?",
    "Compare preemptive and non-preemptive scheduling with examples of each.",
    "How do you choose a good time quantum for Round Robin, and what happens if it's too small or too large?",
    "What is the difference between Priority Scheduling and SJF?",
    "What is aging, and how does it solve starvation in Priority Scheduling?"
]


def generate_templates(out_path="data/processed/concepts_template.jsonl"):
    entries = []

    for key, full_name in ALGORITHMS.items():
        for q_template in QUESTION_TYPES:
            entries.append({
                "instruction": q_template.format(algo=full_name),
                "input": "",
                "output": "TODO: write 3-6 sentences in your own words, or your own teaching voice."
            })

    for q in CROSS_CUTTING:
        entries.append({
            "instruction": q,
            "input": "",
            "output": "TODO: write 3-6 sentences in your own words, or your own teaching voice."
        })

    with open(out_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    print(f"Generated {len(entries)} question templates -> {out_path}")
    print("Open this file and replace every 'TODO' output with a real answer.")
    print("Tip: draft answers faster by reading your raw_docs/ material "
          "(or asking an LLM for a first draft), but READ and EDIT every "
          "single one before keeping it — unlike numericals, nothing here "
          "is auto-verified.")


if __name__ == "__main__":
    generate_templates()