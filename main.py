import mountaincar_SA_qlearning_final
import mountaincar_qlearning_final
import mountaincar_sarsa_final
import mountaincar_sarsa_qlearning_final
from ga_final import *

if __name__ == '__main__':
    genetic_algorithm()
    mountaincar_qlearning_final.train()
    mountaincar_sarsa_final.train()
    mountaincar_SA_qlearning_final.train()
    mountaincar_sarsa_qlearning_final.train()
