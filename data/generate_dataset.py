import json, random
from engine.scheduler import Process, ALGORITHMS

def random_process_table():
    n = random.randint(3, 5)
    return [Process(f"P{i+1}", random.randint(0, 6), random.randint(1, 10))
            for i in range(n)]

def format_gantt(gantt):
    return " | ".join(f"{pid}[{s}-{e}]" for pid, s, e in gantt)

def make_example(algo_name, algo_fn, quantum=None):
    procs = random_process_table()
    table_str = "\n".join(f"{p.pid}: Arrival={p.arrival}, Burst={p.burst}" for p in procs)
    gantt, rows = algo_fn(procs, quantum) if quantum else algo_fn(procs)

    lines = [f"Gantt chart: {format_gantt(gantt)}"]
    for r in rows:
        lines.append(f"{r['pid']}: Completion={r['completion']}, "
                      f"Turnaround={r['turnaround']}, Waiting={r['waiting']}")

    return {"instruction": f"Solve using {algo_name.upper()} scheduling.",
            "input": table_str, "output": "\n".join(lines)}

def generate(n_examples=3000, out_path="data/processed/finetune_data.jsonl"):
    examples = []
    for _ in range(n_examples):
        algo_name = random.choice(["fcfs", "sjf", "srtf", "priority", "rr"])
        quantum = random.choice([2, 3, 4]) if algo_name == "rr" else None
        examples.append(make_example(algo_name, ALGORITHMS[algo_name], quantum))

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} examples to {out_path}")

if __name__ == "__main__":
    generate()
    