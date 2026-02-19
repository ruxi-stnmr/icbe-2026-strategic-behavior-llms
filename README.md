# AI Economic Strategy Simulator

## Project Overview
This script implements an asynchronous game theory simulation designed to benchmark the strategic decision-making capabilities of various Large Language Models (LLMs) through a 100-round iterative economic challenge. Utilizing the OpenRouter API, the system pits models against one another in a modified "Prisoner's Dilemma" framework that evolves through four distinct market phases: Stability, Expansion, Volatility, and Contraction.

Each phase features unique payoff matrices that alter the incentives for cooperation versus exploitation. The engine manages concurrent API calls via `asyncio`, tracks historical moves to provide agents with "memory," and concludes by generating a comprehensive multi-sheet Excel report analyzing leaderboards, win-loss records, and behavioral metrics like cooperation rates and mutual attrition.



## Market Dynamics
The simulation transitions through four environments to test agent adaptability:

* **Phase 1: Baseline Stability** – Standard market demand; rewards mutual cooperation.
* **Phase 2: Volume Expansion** – Favors volume-based strategic choices and aggressive growth.
* **Phase 3: Supply Chain Volatility** – High risk; introduces negative payoffs for asymmetric positions.
* **Phase 4: Economic Contraction** – Severe contraction; symmetric divergent choices result in mutual penalties.

### Installation and Setup
To get the simulation running on your local machine, follow these steps in order:

```bash
# 1. Clone the repository
git clone git@github.com:ruxi-stnmr/icbe-2026-strategic-behavior-llms.git

# 2. Install required dependencies
pip install pandas openai openpyxl

# 3. Configure your API Key
Open LLMs_Business_Simulation.py and replace the API_KEY value:
API_KEY = "your_openrouter_key_here"

# 4. Execute the simulation
python LLMs_Business_Simulation.py
