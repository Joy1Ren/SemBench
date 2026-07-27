import csv
import json
import os
import statistics
import sys

use_case = 'ecomm'
# Results are organized per scale factor: {dir}/{use_case}/sf_{scale_factor}/...
SCALE_FACTORS = {'ecomm': 500, 'movie': 2000}
scale_factor = SCALE_FACTORS.get(use_case, 500)
METRICS_DIR = f"agent_cost_model/metrics/{use_case}/sf_{scale_factor}"
FINAL_ANSWER_DIR = f"agent_cost_model/final_answer/{use_case}"
AGENT_TYPES = {"sampleCost_agent", "execute_agent", "customCost_agent"}
ORACLE_AGENT_TYPES = {"execute_oracle_agent", "customCost_oracle_agent", "customCost_oracle_helper_agent"}
ORACLE_METRICS_DIR = f"files/{use_case}/metrics"

# Per-use-case query metadata.
# Labels use abbreviated operator names: F=Filter, J=Join, M=Map, C=Classify, R=Rank, L=Limit
# agg_queries: query IDs where quality is stored as relative_error (needs inversion for max selection)
_QUERY_CONFIGS = {
    'movie': {
        'all_queries': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10'],
        'query_labels': {
            'Q1': 'F L', 'Q2': 'F L', 'Q3': 'F',  'Q4': 'F',
            'Q5': 'J L', 'Q6': 'J L', 'Q7': 'J',
            'Q8': 'L',   'Q9': 'R',   'Q10': 'R',
        },
        'agg_queries': {'Q3', 'Q4', 'Q8'},
    },
    'ecomm': {
        'all_queries': ['Q4', 'Q5', 'Q12', 'Q13'],
        # 'all_queries': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9',
        #                 'Q10', 'Q11', 'Q12', 'Q13', 'Q14'],
        'query_labels': {
            'Q1':  'F',     'Q2':  'F',     'Q3':  'M',     'Q4':  'M',
            'Q5':  'C',     'Q6':  'C',     'Q7':  'J',     'Q8':  'J',
            'Q9':  'J',     'Q10': 'F J',   'Q11': 'F J',   'Q12': 'M C',
            'Q13': 'F',     'Q14': 'J F R',
        },
        'agg_queries': set(),  # ecomm has no aggregation queries
    },
}

_cfg = _QUERY_CONFIGS.get(use_case, _QUERY_CONFIGS['movie'])
query_labels = _cfg['query_labels']
all_queries  = _cfg['all_queries']
agg_queries  = _cfg['agg_queries']

agent_titles = {
    'sampleCost_agent': 'Baseline 2 (SampleBasedCostModel)',
    'execute_agent': 'Baseline 1 (no cost model)',
    'customCost_agent': "Model 3 (agent-constructed cost model)"
}

oracle_agent_titles = {
    'execute_oracle_agent':    'Baseline 1 (w. oracle gt)',
    'customCost_oracle_agent': 'CustomCost Agent (w. oracle gt)',
    'customCost_oracle_helper_agent': 'CustomCost Agent (w. oracle gt, helper cost agent)',
}


def get_quality(q):
    if not isinstance(q, dict):
        return None
    # Try each metric in priority order; relative_error is inverted (lower = better → 1-x)
    for key in ('f1_score', 'spearman_correlation', 'accuracy'):
        v = q.get(key)
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


def load_oracle_data(agent_type):
    """Read files/{use_case}/metrics/{agent_type}.json and average across runs per query_id."""
    from collections import defaultdict
    path = os.path.join(ORACLE_METRICS_DIR, f"{agent_type}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)

    grouped = defaultdict(list)
    for entry in raw.values():
        # Metrics json may accumulate multiple scale factors; keep only this one.
        if entry.get('scale_factor') is not None and entry['scale_factor'] != scale_factor:
            continue
        grouped[str(entry['query_id'])].append(entry)

    def safe_avg(entries, field):
        vals = [e[field] for e in entries if field in e and e[field] is not None]
        return sum(vals) / len(vals) if vals else None

    result = {}
    for qid, entries in grouped.items():
        result[f'Q{qid}'] = {
            'cost':                  safe_avg(entries, 'cost'),
            'latency':               safe_avg(entries, 'latency'),
            'quality':               safe_avg(entries, 'quality'),
            'plans_written':         safe_avg(entries, 'plans_written'),
            'plans_executed':        safe_avg(entries, 'plans_executed'),
            'unique_plans_executed': safe_avg(entries, 'unique_plans_executed'),
            'agent_cost':            safe_avg(entries, 'agent_cost'),
            'helper_cost':           safe_avg(entries, 'helper_cost'),
            'subset_execution_cost': safe_avg(entries, 'subset_execution_cost'),
            'oracle_cost':           safe_avg(entries, 'oracle_cost'),
        }
    return result


def rows_from_oracle_data(data):
    rows = []
    for qid in all_queries:
        label = query_labels.get(qid, '')
        if qid in data:
            q = data[qid]
            rows.append((
                qid, label,
                q['cost'], q['latency'], q['quality'],
                q['plans_written'], q['plans_executed'], q['unique_plans_executed'],
                q['agent_cost'], q['helper_cost'], q['subset_execution_cost'], q['oracle_cost'],
            ))
        else:
            rows.append((qid, label, None, None, None, None, None, None, None, None, None, None))
    return rows


def print_oracle_table(title, rows):
    costs     = [r[2]  for r in rows if r[2]  is not None]
    latencies = [r[3]  for r in rows if r[3]  is not None]
    qualities = [r[4]  for r in rows if r[4]  is not None]
    p_written = [r[5]  for r in rows if r[5]  is not None]
    p_exec    = [r[6]  for r in rows if r[6]  is not None]
    agent_cs  = [r[8]  for r in rows if r[8]  is not None]
    helper_cs = [r[9]  for r in rows if r[9]  is not None]
    subset_cs = [r[10] for r in rows if r[10] is not None]
    oracle_cs = [r[11] for r in rows if r[11] is not None]

    # The 5 opt-cost sub-columns occupy: 12 + 2 + 12 + 2 + 12 + 2 + 14 + 2 + 12 = 70 chars
    OPT_GROUP = 'OptCost = (AgentCost + HelperCost + SubsetExecCost + OracleCost)'
    W = 154
    print(f'{title:^{W}}')
    print(f"{'':14} {'':12}  {'':10}  {'':12}    {'':12}  {'':10}    {OPT_GROUP:^70}")
    print(f"{'':14} {'Cost':>12}  {'Quality':>10}  {'Latency':>12}    "
          f"{'PlansWritten':>12}  {'PlansExec':>10}    "
          f"{'OptCost':>12}  {'AgentCost':>12}  {'HelperCost':>12}  {'SubsetExecCost':>14}  {'OracleCost':>12}")
    print('─' * W)

    for qid, label, c, t, ql, pw, pe, pu, ac, hc, sc, oc in rows:
        q_label = f'{qid}: {label}'
        pw_str  = f'{pw:.1f}' if pw is not None else 'N/A'
        pe_str  = f'{pe:.1f}' if pe is not None else 'N/A'
        tot     = (ac or 0) + (hc or 0) + (sc or 0) + (oc or 0) if ac is not None else None
        print(f'{q_label:<14} {fmt_cost(c):>12}  {fmt_quality(ql):>10}  {fmt_latency(t):>12}    '
              f'{pw_str:>12}  {pe_str:>10}    '
              f'{fmt_cost(tot):>12}  {fmt_cost(ac):>12}  {fmt_cost(hc):>12}  {fmt_cost(sc):>14}  {fmt_cost(oc):>12}')

    print('─' * W)

    if costs:
        avg_c  = sum(costs) / len(costs)
        avg_ql = sum(qualities) / len(qualities) if qualities else None
        avg_t  = sum(latencies) / len(latencies) if latencies else None
        avg_pw = sum(p_written) / len(p_written) if p_written else None
        avg_pe = sum(p_exec)    / len(p_exec)    if p_exec    else None
        avg_ac = sum(agent_cs)  / len(agent_cs)  if agent_cs  else None
        avg_hc = sum(helper_cs) / len(helper_cs) if helper_cs else None
        avg_sc = sum(subset_cs) / len(subset_cs) if subset_cs else None
        avg_oc = sum(oracle_cs) / len(oracle_cs) if oracle_cs else None
        avg_tot = (avg_ac or 0) + (avg_hc or 0) + (avg_sc or 0) + (avg_oc or 0) if avg_ac is not None else None

        std_c  = statistics.stdev(costs)     if len(costs)     > 1 else 0.0
        std_ql = statistics.stdev(qualities) if len(qualities) > 1 else 0.0
        std_t  = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        pw_str = f'{avg_pw:.1f}' if avg_pw is not None else 'N/A'
        pe_str = f'{avg_pe:.1f}' if avg_pe is not None else 'N/A'

        print(f"{'Avg':<14} {fmt_cost(avg_c):>12}  {fmt_quality(avg_ql):>10}  {fmt_latency(avg_t):>12}    "
              f"{pw_str:>12}  {pe_str:>10}    "
              f"{fmt_cost(avg_tot):>12}  {fmt_cost(avg_ac):>12}  {fmt_cost(avg_hc):>12}  {fmt_cost(avg_sc):>14}  {fmt_cost(avg_oc):>12}")
        print(f"{'Std Dev':<14} {'±'+fmt_cost(std_c):>12}  {'±'+fmt_quality(std_ql):>10}  {'±'+fmt_latency(std_t):>12}")
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
            def _agg_quality_key(r):
                # For aggregation queries quality may be stored as raw relative_error (lower = better)
                try:
                    return 1.0 - float(r.get('quality') or '')
                except (ValueError, TypeError):
                    return float('-inf')
            def _quality_key(r):
                try:
                    return float(r.get('quality') or '')
                except (ValueError, TypeError):
                    return float('-inf')
            selected_data[qid] = max(candidates, key=_agg_quality_key if qid in agg_queries else _quality_key)
    return selected_data, plans_executed, unique_plans_executed, total_costs


def load_plans_written(agent_type):
    plans_written = {}
    for qid in all_queries:
        path = os.path.join(FINAL_ANSWER_DIR, agent_type, f"sf_{scale_factor}", f"{qid}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        if 'plan_codes' in d:  # legacy flat format (single run)
            plans_written[qid] = len(d.get('plan_codes', []))
        else:  # nested by run number: {run_key: {plan_codes, ...}} — average across runs
            counts = [len(r.get('plan_codes', [])) for r in d.values() if isinstance(r, dict)]
            if counts:
                plans_written[qid] = sum(counts) / len(counts)
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

if arg in ORACLE_AGENT_TYPES:
    data = load_oracle_data(arg)
    title = oracle_agent_titles[arg]
    print_oracle_table(title, rows_from_oracle_data(data))
elif arg in AGENT_TYPES:
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
