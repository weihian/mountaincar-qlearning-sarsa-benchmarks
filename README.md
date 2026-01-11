# Reinforcement Learning on Mountain Car: Hybrid Q-SARSA Innovation

This project provides a comprehensive analysis of Reinforcement Learning (RL) agents applied to the **MountainCar-v0** environment. While standard approaches often choose between on-policy and off-policy methods, this project innovates by **combining Q-Learning and SARSA** into a unified agent to balance stability and optimality.



## 🚀 The Core Innovation: Hybrid Q-SARSA

The standout feature of this repository is the `mountaincar_sarsa_qlearning_final.py` implementation. This hybrid model addresses the common pitfalls of individual TD-control methods:

1.  **SARSA (On-Policy) Stability:** By following the current policy to choose the next action ($A'$), the agent accounts for its own exploration risks, leading to safer and more stable convergence in early training.
2.  **Q-Learning (Off-Policy) Optimality:** By incorporating the maximum future reward ($\max Q$), the agent maintains a strong drive toward the theoretically optimal path, preventing the over-cautiousness sometimes seen in pure SARSA.



## 🛠️ Key Features

* **Hybrid Engine:** A custom implementation merging SARSA's updates with Q-Learning's greedy targets.
* **Genetic Algorithm (GA) Optimization:** The `ga_final.py` script treats hyperparameters (learning rate, explore rate, and bucket sizes) as DNA, evolving a population of agents to find the most efficient training configuration.
* **Simulated Annealing (SA):** An implementation that uses thermodynamic-inspired decay for exploration rates.
* **Deep Q-Learning (DQN):** A neural-network-based approach for handling the environment without manual discretization.
* **Multi-Strategy Benchmarking:** Compare Q-Learning, SARSA, Hybrid, and SA models through a single entry point (`main.py`).

## 📂 Project Structure

| File | Algorithm / Purpose |
| :--- | :--- |
| `mountaincar_sarsa_qlearning_final.py` | **Hybrid Q-SARSA Agent** |
| `ga_final.py` | Genetic Algorithm Hyperparameter Tuner |
| `mountaincar_qlearning_final.py` | Standard Q-Learning (Off-Policy) |
| `mountaincar_sarsa_final.py` | Standard SARSA (On-Policy) |
| `mountaincar_SA_qlearning_final.py` | Q-Learning with Simulated Annealing |
| `mountaincar-deepqlearning.py` | Deep Q-Network (DQN) implementation |
| `main.py` | Project Orchestrator |

## 🧠 Technical Methodology

### State Discretization
To handle the continuous state space of MountainCar, we employ a high-resolution bucketing system:
- **Position:** Discretized into 19 buckets.
- **Velocity:** Discretized into 15 to 29 buckets.

### The Genetic Evolution Loop
The GA component automates the "Trial and Error" of finding RL parameters. It evaluates the **Fitness** of an agent based on its "Streak"—how many times it can solve the environment consecutively within a 200-step limit.



## 💻 Installation & Usage

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/mountaincar-hybrid-q-sarsa.git](https://github.com/your-username/mountaincar-hybrid-q-sarsa.git)
    cd mountaincar-hybrid-q-sarsa
    ```
2.  **Install Dependencies:**
    ```bash
    pip install gym numpy mpmath
    ```
3.  **Run the Benchmark:**
    ```bash
    python main.py
    ```

## 📊 Evaluation
The hybrid model is evaluated against standard benchmarks using:
- **Success Streak:** Max consecutive successful runs.
- **Time-to-Goal:** Average steps per episode.
- **Learning Efficiency:** Number of episodes required to reach a stable policy.

## 📜 License
MIT License
