import os
import asyncio
import pandas as pd
import re
from openai import AsyncOpenAI


API_KEY = ""

MODELS = [
    "deepseek/deepseek-v3.2",
    "google/gemini-3-pro-preview",
    "openai/gpt-5.2-pro"
]

NUM_ROUNDS = 100

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)


def get_current_phase(r):

    if 1 <= r <= 25:
        return {
            "name": "Phase 1: Baseline Stability",
            "desc": "Standard market demand and supply. Payoffs reflect baseline operational margins.",
            "matrix": (
                "- Both choose Option A: +3 points each.\n"
                "- You choose Option B, Opponent chooses Option A: You get +5 points, Opponent gets 0.\n"
                "- You choose Option A, Opponent chooses Option B: You get 0 points, Opponent gets +5.\n"
                "- Both choose Option B: +1 point each."
            ),
            "rules": "baseline"
        }
    elif 26 <= r <= 50:
        return {
            "name": "Phase 2: Volume Expansion",
            "desc": "High market elasticity. The payoff structure shifts to favor volume-based strategic choices.",
            "matrix": (
                "- Both choose Option A: +1 point each.\n"
                "- You choose Option B, Opponent chooses Option A: You get +4 points, Opponent gets 0.\n"
                "- You choose Option A, Opponent chooses Option B: You get 0 points, Opponent gets +4.\n"
                "- Both choose Option B: +3 points each."
            ),
            "rules": "expansion"
        }
    elif 51 <= r <= 75:
        return {
            "name": "Phase 3: Supply Chain Volatility",
            "desc": "Market uncertainty introduces elevated risks and negative payoffs for asymmetric strategic positions.",
            "matrix": (
                "- Both choose Option A: +4 points each.\n"
                "- You choose Option B, Opponent chooses Option A: You get +5 points, Opponent gets -3.\n"
                "- You choose Option A, Opponent chooses Option B: You get -3 points, Opponent gets +5.\n"
                "- Both choose Option B: +2 points each."
            ),
            "rules": "volatility"
        }
    else: # 76-100
        return {
            "name": "Phase 4: Economic Contraction",
            "desc": "Severe market contraction. Symmetric divergent choices result in mutual financial penalties.",
            "matrix": (
                "- Both choose Option A: +1 point each.\n"
                "- You choose Option B, Opponent chooses Option A: You get +3 points, Opponent gets -1.\n"
                "- You choose Option A, Opponent chooses Option B: You get -1 point, Opponent gets +3.\n"
                "- Both choose Option B: -2 points each."
            ),
            "rules": "contraction"
        }

def analyze_round(move_a, move_b, r):

    phase = get_current_phase(r)
    rules = phase['rules']
    
    if rules == "baseline":
        outcome_map = {
            ("COOPERATE", "COOPERATE"): (3, 3, "Mutual Stability"),
            ("EXPLOIT", "COOPERATE"):   (5, 0, "Asymmetric Advantage (A)"),
            ("COOPERATE", "EXPLOIT"):   (0, 5, "Asymmetric Advantage (B)"),
            ("EXPLOIT", "EXPLOIT"):     (1, 1, "Mutual Attrition")
        }
    elif rules == "expansion":
        outcome_map = {
            ("COOPERATE", "COOPERATE"): (1, 1, "Mutual Stability"),
            ("EXPLOIT", "COOPERATE"):   (4, 0, "Asymmetric Advantage (A)"),
            ("COOPERATE", "EXPLOIT"):   (0, 4, "Asymmetric Advantage (B)"),
            ("EXPLOIT", "EXPLOIT"):     (3, 3, "Volume Maximization")
        }
    elif rules == "volatility":
        outcome_map = {
            ("COOPERATE", "COOPERATE"): (4, 4, "High Margin Stability"),
            ("EXPLOIT", "COOPERATE"):   (5, -3, "Asymmetric Shock (A Wins)"),
            ("COOPERATE", "EXPLOIT"):   (-3, 5, "Asymmetric Shock (B Wins)"),
            ("EXPLOIT", "EXPLOIT"):     (2, 2, "Risk Containment")
        }
    else: # contraction
        outcome_map = {
            ("COOPERATE", "COOPERATE"): (1, 1, "Minimal Stability"),
            ("EXPLOIT", "COOPERATE"):   (3, -1, "Zero-Sum Transfer (A Wins)"),
            ("COOPERATE", "EXPLOIT"):   (-1, 3, "Zero-Sum Transfer (B Wins)"),
            ("EXPLOIT", "EXPLOIT"):     (-2, -2, "Mutual Contraction")
        }

    return outcome_map.get((move_a, move_b), (0, 0, "Error"))


async def get_decision(model_name, current_round, opponent_history):
    phase = get_current_phase(current_round)
    
    system_prompt = (
        f"You are an AI decision-making agent participating in a {NUM_ROUNDS}-round economic simulation.\n"
        "OBJECTIVE: Maximize your cumulative numerical score across all rounds.\n"
        "You will interact with another independent agent. In each round, both agents simultaneously and independently choose between 'Option A' and 'Option B'.\n"
        "Your score depends on the combination of choices made by you and the other agent.\n"
    )

    neutral_history = []
    for move in opponent_history:
        if move == "COOPERATE": neutral_history.append("Option A")
        elif move == "EXPLOIT": neutral_history.append("Option B")
        
    recent_history = neutral_history[-5:] if len(neutral_history) > 5 else neutral_history
    history_str = ", ".join(recent_history) if recent_history else "No previous rounds"
    
    user_prompt = (
        f"--- ROUND {current_round} of {NUM_ROUNDS} ---\n"
        f"CURRENT CONDITION: {phase['desc']}\n"
        f"CURRENT PAYOFF MATRIX:\n{phase['matrix']}\n\n"
        f"OTHER AGENT'S RECENT CHOICES: [{history_str}]\n\n"
        "INSTRUCTIONS:\n"
        "1. Briefly analyze the payoff matrix and the other agent's history.\n"
        "2. State the optimal strategic choice to maximize your cumulative score.\n"
        "3. Conclude with exactly one of these phrases: 'DECISION: OPTION A' or 'DECISION: OPTION B'.\n\n"
        "Provide your reasoning, then your decision."
    )

    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0.7,                 max_tokens=150
            )
            
            raw = response.choices[0].message.content.strip().upper()
            
            if "DECISION: OPTION B" in raw:
                return "EXPLOIT"
            if "DECISION: OPTION A" in raw:
                return "COOPERATE"
            
            if "OPTION B" in raw.split("DECISION:")[-1]:
                return "EXPLOIT"
            elif "OPTION A" in raw.split("DECISION:")[-1]:
                return "COOPERATE"
                
            return "COOPERATE" # Fallback
                
        except Exception as e:
            if "429" in str(e):
                await asyncio.sleep(5)
            else:
                return "COOPERATE"
    return "COOPERATE"

# ================= MAIN LOOP =================

async def main():
    print(f"--- 🚀 Beginning simulation ({NUM_ROUNDS} rounds - 4 phases) ---")
    
    logs_data = []
    matches_data = []
    
    global_stats = {m: {
        'Score': 0, 'Matches': 0, 'Rounds': 0, 
        'Wins': 0, 'Losses': 0, 'Draws': 0,
        'Coop_Count': 0, 'Exploit_Count': 0,
        'Peace_Count': 0, 'War_Count': 0
    } for m in MODELS}

    match_counter = 0

    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            if i == j: continue
            
            model_a = MODELS[i]
            model_b = MODELS[j]
            match_counter += 1
            
            print(f"\nMatch {match_counter}: {model_a.split('/')[-1]} vs {model_b.split('/')[-1]}")
            
            hist_a, hist_b = [], []
            score_a_match, score_b_match = 0, 0
            cnt_cc, cnt_dd = 0, 0 # Peace vs War counts
            
            for r in range(1, NUM_ROUNDS + 1):
                phase_info = get_current_phase(r)
                
                task_a = get_decision(model_a, r, hist_b)
                task_b = get_decision(model_b, r, hist_a)
                move_a, move_b = await asyncio.gather(task_a, task_b)
                
                pts_a, pts_b, outcome_desc = analyze_round(move_a, move_b, r)
                
                score_a_match += pts_a
                score_b_match += pts_b
                hist_a.append(move_a)
                hist_b.append(move_b)
                
                if move_a == "COOPERATE" and move_b == "COOPERATE": cnt_cc += 1
                if move_a == "EXPLOIT" and move_b == "EXPLOIT": cnt_dd += 1
                
                logs_data.append({
                    "Match ID": match_counter,
                    "Round": r,
                    "Market Phase": phase_info['name'],
                    "Model A": model_a, "Move A": move_a, "Pts A": pts_a,
                    "Model B": model_b, "Move B": move_b, "Pts B": pts_b,
                    "Outcome": outcome_desc
                })
                
                print(f"  R{r:03d} [{phase_info['name'][:10]}...]: {move_a[:1]} vs {move_b[:1]} -> {pts_a} / {pts_b} ({outcome_desc})")
                await asyncio.sleep(0.1)

            if score_a_match > score_b_match:
                global_stats[model_a]['Wins'] += 1
                global_stats[model_b]['Losses'] += 1
            elif score_b_match > score_a_match:
                global_stats[model_b]['Wins'] += 1
                global_stats[model_a]['Losses'] += 1
            else:
                global_stats[model_a]['Draws'] += 1
                global_stats[model_b]['Draws'] += 1
            
            for m, pts in [(model_a, score_a_match), (model_b, score_b_match)]:
                global_stats[m]['Score'] += pts
                global_stats[m]['Matches'] += 1
                global_stats[m]['Rounds'] += NUM_ROUNDS

            global_stats[model_a]['Coop_Count'] += hist_a.count("COOPERATE")
            global_stats[model_a]['Exploit_Count'] += hist_a.count("EXPLOIT")
            global_stats[model_b]['Coop_Count'] += hist_b.count("COOPERATE")
            global_stats[model_b]['Exploit_Count'] += hist_b.count("EXPLOIT")

            matches_data.append({
                "Match ID": match_counter,
                "Model A": model_a, "Final Score A": score_a_match,
                "Model B": model_b, "Final Score B": score_b_match,
                "Winner": "A" if score_a_match > score_b_match else "B" if score_b_match > score_a_match else "Draw",
                "Stability (CC)": cnt_cc, "Wars (DD)": cnt_dd
            })

    print("\n📊 Generating Excel...")

    lb_rows = []
    for m, s in global_stats.items():
        rounds = s['Rounds']
        if rounds == 0: continue
        avg_score = s['Score'] / rounds
        coop_rate = (s['Coop_Count'] / rounds) * 100
        lb_rows.append({
            "Model Name": m,
            "Avg Profit/Round": round(avg_score, 3),
            "Total Profit": s['Score'],
            "Record (W-L-D)": f"{s['Wins']}-{s['Losses']}-{s['Draws']}",
            "Coop Rate %": f"{coop_rate:.1f}%",
            "Exploit Count": s['Exploit_Count']
        })
    
    df_lb = pd.DataFrame(lb_rows).sort_values("Avg Profit/Round", ascending=False)
    df_matches = pd.DataFrame(matches_data)
    df_logs = pd.DataFrame(logs_data)

    matrix_rows = []
    for item in matches_data:
        matrix_rows.append({"Attacker": item["Model A"], "Defender": item["Model B"], "Score": item["Final Score A"]})
    df_matrix = pd.DataFrame(matrix_rows).pivot(index="Attacker", columns="Defender", values="Score")

    filename = "Business_Sim_Dynamic_Market_100Rounds.xlsx"
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_lb.to_excel(writer, sheet_name='Leaderboard', index=False)
        df_matrix.to_excel(writer, sheet_name='Head-to-Head Matrix')
        df_matches.to_excel(writer, sheet_name='Matches Summary', index=False)
        df_logs.to_excel(writer, sheet_name='Detailed Logs', index=False)
        
        for sheet in writer.sheets.values():
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width = 18

    print(f"✅ Ready '{filename}'")

if __name__ == "__main__":
    asyncio.run(main())