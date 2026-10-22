from dataclasses import dataclass, field
from typing import List

@dataclass
class Process:
    pid: str
    arrival: int
    burst: int
    priority: int = 0
    remaining: int = field(init=False)

    def __post_init__(self):
        self.remaining = self.burst


def _metrics(processes, completion):
    rows = []
    for p in processes:
        ct = completion[p.pid]
        tat = ct - p.arrival
        wt = tat - p.burst
        rows.append({"pid": p.pid, "arrival": p.arrival, "burst": p.burst,
                      "completion": ct, "turnaround": tat, "waiting": wt})
    return rows


def fcfs(processes: List[Process]):
    procs = sorted(processes, key=lambda p: p.arrival)
    t, gantt, completion = 0, [], {}
    for p in procs:
        t = max(t, p.arrival)
        gantt.append((p.pid, t, t + p.burst))
        t += p.burst
        completion[p.pid] = t
    return gantt, _metrics(processes, completion)


def sjf_non_preemptive(processes: List[Process]):
    procs = sorted(processes, key=lambda p: p.arrival)
    remaining = procs[:]
    t, gantt, completion = 0, [], {}
    while remaining:
        ready = [p for p in remaining if p.arrival <= t]
        if not ready:
            t = min(p.arrival for p in remaining)
            continue
        nxt = min(ready, key=lambda p: p.burst)
        gantt.append((nxt.pid, t, t + nxt.burst))
        t += nxt.burst
        completion[nxt.pid] = t
        remaining.remove(nxt)
    return gantt, _metrics(processes, completion)


def srtf(processes: List[Process]):
    procs = [Process(p.pid, p.arrival, p.burst, p.priority) for p in processes]
    t, gantt, completion = 0, [], {}
    while any(p.remaining > 0 for p in procs):
        ready = [p for p in procs if p.arrival <= t and p.remaining > 0]
        if not ready:
            t += 1
            continue
        cur = min(ready, key=lambda p: p.remaining)
        if gantt and gantt[-1][0] == cur.pid:
            gantt[-1] = (cur.pid, gantt[-1][1], t + 1)
        else:
            gantt.append((cur.pid, t, t + 1))
        cur.remaining -= 1
        t += 1
        if cur.remaining == 0:
            completion[cur.pid] = t
    return gantt, _metrics(processes, completion)


def priority_non_preemptive(processes: List[Process]):
    procs = sorted(processes, key=lambda p: p.arrival)
    remaining = procs[:]
    t, gantt, completion = 0, [], {}
    while remaining:
        ready = [p for p in remaining if p.arrival <= t]
        if not ready:
            t = min(p.arrival for p in remaining)
            continue
        nxt = min(ready, key=lambda p: p.priority)
        gantt.append((nxt.pid, t, t + nxt.burst))
        t += nxt.burst
        completion[nxt.pid] = t
        remaining.remove(nxt)
    return gantt, _metrics(processes, completion)


def round_robin(processes: List[Process], quantum: int):
    procs = sorted([Process(p.pid, p.arrival, p.burst, p.priority) for p in processes],
                    key=lambda p: p.arrival)
    queue, t, gantt, completion = [], procs[0].arrival, [], {}
    i, n = 1, len(procs)
    queue.append(procs[0])
    while queue:
        cur = queue.pop(0)
        run = min(quantum, cur.remaining)
        gantt.append((cur.pid, t, t + run))
        t += run
        cur.remaining -= run
        while i < n and procs[i].arrival <= t:
            queue.append(procs[i]); i += 1
        if cur.remaining > 0:
            queue.append(cur)
        else:
            completion[cur.pid] = t
        if not queue and i < n:
            queue.append(procs[i]); t = max(t, procs[i].arrival); i += 1
    return gantt, _metrics(processes, completion)


ALGORITHMS = {"fcfs": fcfs, "sjf": sjf_non_preemptive, "srtf": srtf,
              "priority": priority_non_preemptive, "rr": round_robin}

procs = [Process("P1", 0, 5), Process("P2", 1, 3), Process("P3", 2, 2)]
gantt, rows = fcfs(procs)
print(gantt)
for r in rows:
    print(r)