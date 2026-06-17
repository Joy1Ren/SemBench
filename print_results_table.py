import csv
import json
import os
import statistics
import sys

METRICS_DIR = "agent_cost_model/metrics"
FINAL_ANSWER_DIR = "agent_cost_model/final_answer"
AGENT_TYPES = {"sampleCost_agent", "execute_agent", "customCost_agent"}

query_labels = {
    'Q1': 'F L', 'Q2': 'F L', 'Q3': 'F',  'Q4': 'F',
    'Q5': 'J L', 'Q6': 'J L', 'Q7': 'J',
    'Q8': 'C',   'Q9': 'R',   'Q10': 'R',
}

agent_titles = {
    'sampleCost_agent': 'Baseline 2 (SampleBasedCostModel)',
    'execute_agent': 'Baseline 1 (no cost model)',
    'customCost_agent': "Model 3 (agent-constructed cost model)"
}

all_queries = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']


def get_quality(q):
    if isinstance(q, dict):
        v = q.get('f1_score')
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
        v = q.get('relative_error')
        if v is not None:
            try:
                return 1.0 - float(v)
            except (ValueError, TypeError):
                pass
        v = q.get('spearman_correlation')
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    else:
        if 'f1_score' in q:
            return q['f1_score']
        elif 'relative_error' in q:
            return 1.0 - q['relative_error']
        elif 'spearman_correlation' in q:
            return q['spearman_correlation']
    return None


def fmt_cost(c):
    if c is None:
        return 'N/A'
    if c < 0.01:
        return f'${c*1000:.1f}·10⁻³'
    return f'${c:.2f}'


def fmt_quality(q):
    return 'N/A' if q is None else f'{q:.2f}'


def fmt_latency(t):
    return 'N/A' if t is None else f'{t:.1f} s'


def print_table(title, rows):
    costs = [c for _, _, c, _, _ in rows if c is not None]
    qualities = [ql for _, _, _, ql, _ in rows if ql is not None]
    latencies = [t for _, _, _, _, t in rows if t is not None]

    W = 58
    print(f'{title:^{W}}')
    print(f"{'':14} {'Cost':>12}  {'Quality':>10}  {'Latency':>12}")
    print('─' * W)

    for qid, label, c, ql, t in rows:
        q_label = f'{qid}: {label}'
        print(f'{q_label:<14} {fmt_cost(c):>12}  {fmt_quality(ql):>10}  {fmt_latency(t):>12}')

    print('─' * W)

    if costs:
        avg_c = sum(costs) / len(costs)
        avg_q = sum(qualities) / len(qualities) if qualities else None
        avg_t = sum(latencies) / len(latencies) if latencies else None
        std_c = statistics.stdev(costs) if len(costs) > 1 else 0.0
        std_q = statistics.stdev(qualities) if len(qualities) > 1 else 0.0
        std_t = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        print(f"{'Avg':<14} {fmt_cost(avg_c):>12}  {fmt_quality(avg_q):>10}  {fmt_latency(avg_t):>12}")
        print(f"{'Std Dev':<14} {'±'+fmt_cost(std_c):>12}  {'±'+fmt_quality(std_q):>10}  {'±'+fmt_latency(std_t):>12}")
        print('─' * W)


def print_agent_table(title, rows):
    costs = [c for _, _, c, _, _, _, _, _, _ in rows if c is not None]
    qualities = [ql for _, _, _, ql, _, _, _, _, _ in rows if ql is not None]
    latencies = [t for _, _, _, _, t, _, _, _, _ in rows if t is not None]
    n_execs = [n for _, _, _, _, _, n, _, _, _ in rows if n is not None]
    n_uniques = [u for _, _, _, _, _, _, u, _, _ in rows if u is not None]
    tot_costs = [tc for _, _, _, _, _, _, _, tc, _ in rows if tc is not None]
    n_writtens = [nw for _, _, _, _, _, _, _, _, nw in rows if nw is not None]

    W = 108
    print(f'{title:^{W}}')
    print(f"{'':14} {'Cost':>12}  {'Quality':>10}  {'Latency':>12}  {'#PlansExec':>10}  {'#UniquePlans':>12}  {'TotalCost':>12}  {'#PlansWritten':>13}")
    print('─' * W)

    for qid, label, c, ql, t, n_exec, n_unique, tot_c, n_written in rows:
        q_label = f'{qid}: {label}'
        n_exec_str = str(n_exec) if n_exec is not None else 'N/A'
        n_unique_str = str(n_unique) if n_unique is not None else 'N/A'
        n_written_str = str(n_written) if n_written is not None else 'N/A'
        print(f'{q_label:<14} {fmt_cost(c):>12}  {fmt_quality(ql):>10}  {fmt_latency(t):>12}  {n_exec_str:>10}  {n_unique_str:>12}  {fmt_cost(tot_c):>12}  {n_written_str:>13}')

    print('─' * W)

    if costs:
        avg_c = sum(costs) / len(costs)
        avg_q = sum(qualities) / len(qualities) if qualities else None
        avg_t = sum(latencies) / len(latencies) if latencies else None
        avg_exec_str = f'{sum(n_execs)/len(n_execs):.1f}' if n_execs else 'N/A'
        avg_unique_str = f'{sum(n_uniques)/len(n_uniques):.1f}' if n_uniques else 'N/A'
        avg_tot_c = sum(tot_costs) / len(tot_costs) if tot_costs else None
        avg_written_str = f'{sum(n_writtens)/len(n_writtens):.1f}' if n_writtens else 'N/A'
        std_c = statistics.stdev(costs) if len(costs) > 1 else 0.0
        std_q = statistics.stdev(qualities) if len(qualities) > 1 else 0.0
        std_t = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        print(f"{'Avg':<14} {fmt_cost(avg_c):>12}  {fmt_quality(avg_q):>10}  {fmt_latency(avg_t):>12}  {avg_exec_str:>10}  {avg_unique_str:>12}  {fmt_cost(avg_tot_c):>12}  {avg_written_str:>13}")
        print(f"{'Std Dev':<14} {'±'+fmt_cost(std_c):>12}  {'±'+fmt_quality(std_q):>10}  {'±'+fmt_latency(std_t):>12}  {'':>10}  {'':>12}  {'':>12}  {'':>13}")
        print('─' * W)


def load_agent_data(agent_type):
    selected_data, plans_executed, unique_plans_executed, total_costs = {}, {}, {}, {}
    for qid in all_queries:
        num = qid[1:]
        path = os.path.join(METRICS_DIR, f"Q{num}_{agent_type}_results.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline='') as f:
            all_rows = list(csv.DictReader(f))
        plans_executed[qid] = len(all_rows)
        unique_plans_executed[qid] = len({r['plan_name'] for r in all_rows})
        total_costs[qid] = sum(float(r['cost_usd']) for r in all_rows)
        candidates = [r for r in all_rows if r.get('final_selected', '').strip().lower() == 'true']
        if candidates:
            #temporary fix for storing quality = relative_error
            def _agg_quality_key(r):
                try:
                    return 1.0 - float(r.get('quality') or '')
                except (ValueError, TypeError):
                    return float('-inf')
            def _quality_key(r):
                try:
                    return float(r.get('quality') or '')
                except (ValueError, TypeError):
                    return float('-inf')
            selected_data[qid] = max(candidates, key=_quality_key if qid not in ["Q3", "Q4", "Q8"] else _agg_quality_key)
    return selected_data, plans_executed, unique_plans_executed, total_costs


def load_plans_written(agent_type):
    plans_written = {}
    for qid in all_queries:
        path = os.path.join(FINAL_ANSWER_DIR, agent_type, f"{qid}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        plans_written[qid] = len(d.get('plan_codes', []))
    return plans_written


def rows_from_agent_data(selected_data, plans_executed, unique_plans_executed, total_costs, plans_written):
    rows = []
    for qid in all_queries:
        label = query_labels.get(qid, '')
        if qid in selected_data:
            q = selected_data[qid]
            c = float(q['cost_usd'])
            t = float(q['latency_s'])
            ql = get_quality(q)
            rows.append((qid, label, c, ql, t,
                         plans_executed.get(qid), unique_plans_executed.get(qid),
                         total_costs.get(qid), plans_written.get(qid)))
        else:
            rows.append((qid, label, None, None, None, None, None, None, None))
    return rows


def rows_from_json(data):
    rows = []
    for qid in all_queries:
        label = query_labels.get(qid, '')
        if qid in data:
            q = data[qid]
            c = q['money_cost']
            t = q['execution_time']
            ql = get_quality(q)
            rows.append((qid, label, c, ql, t))
        else:
            rows.append((qid, label, None, None, None))
    return rows


arg = sys.argv[1] if len(sys.argv) > 1 else None

if arg in AGENT_TYPES:
    selected_data, plans_executed, unique_plans_executed, total_costs = load_agent_data(arg)
    plans_written = load_plans_written(arg)
    title = agent_titles[arg]
    print_agent_table(title, rows_from_agent_data(selected_data, plans_executed, unique_plans_executed, total_costs, plans_written))
else:
    metrics_file = arg or "palimpzest_physical_planner_agent.json"
    with open(metrics_file) as f:
        data = json.load(f)
    title = "Physical Planner Agent"
    print_table(title, rows_from_json(data))
