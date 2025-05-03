# Active Inference Foraging and Trading Economy

## Overview
This project implements a multi-agent economic simulation where agents use active inference principles to make decisions about foraging resources and trading with other agents. The simulation explores how economic behaviors emerge from agents using Expected Free Energy (EFE) minimization as their decision-making framework.

## Key Features
1. **Grid-based world** with different resource types
2. **Probabilistic resource generation** with varying distributions
3. **Agents with specialized capabilities** creating natural comparative advantages
4. **Time-based constraints** through a turn-based system
5. **Dynamic network topology** for trading relationships

## Active Inference Framework
The agents in this simulation use Expected Free Energy (EFE) minimization to make decisions, which is a key principle in the active inference framework. This approach allows agents to balance:

- **Exploitation**: Maximizing expected utility by gathering resources and trading efficiently
- **Exploration**: Seeking new information about resource locations and trade partners

The implementation captures several core aspects of active inference:
- Generative models of the environment (beliefs about resources and other agents)
- Prediction error minimization through action selection
- Precision-weighted belief updating
- Information seeking through expected information gain

## Resource System
Resources in the economy include:
- **Basic resources**: Wood, stone, food with fixed values
- **Complementary resources**: Resources that provide bonus value when combined
- **Specialized resources**: Resources with different values to different agents

## Agent Specializations
Agents have four types of specializations:
1. **Movement efficiency** - affects the energy cost of movement
2. **Harvesting efficiency** - affects the probability of successfully gathering resources
3. **Trading advantage** - affects trade success and perceived resource values
4. **Resource preferences** - different utility functions for different resources

## Requirements
- Python 3.7+
- Mesa (agent-based modeling framework)
- NumPy
- Matplotlib
- NetworkX

## Installation
```bash
pip install mesa numpy matplotlib networkx
```

## Running the Simulation
There are two ways to run the simulation:

1. **Interactive mode with visualization server**:
```bash
python active_inference_economy.py --server
```

2. **Headless mode with post-simulation analysis**:
```bash
python active_inference_economy.py --steps 200 --analyze
```

## Command Line Arguments
- `--server`: Run with Mesa's interactive visualization server
- `--steps N`: Run the simulation for N steps (default: 100)
- `--analyze`: Run detailed analysis of agent behavior after simulation

## Implementation Details

### Active Inference Agent
The `ForagingAgent` class implements agents that use active inference principles:

```python
def expected_free_energy(self, action, current_state):
    """
    Calculate the expected free energy for an action
    Lower EFE is better (we're minimizing expected surprise)
    """
    # Components of EFE
    expected_utility = 0
    expected_information_gain = 0
    
    # Different action types: move, harvest, trade
    # ...
    
    # Combine components with a weighting parameter
    explore_exploit_balance = 0.7  # Higher values favor exploitation over exploration
    
    # Lower is better for EFE (minimizing expected surprise)
    efe = -(explore_exploit_balance * expected_utility + 
           (1 - explore_exploit_balance) * expected_information_gain)
    return efe
```

The EFE calculation accounts for:
- Expected utility gain (exploitation)
- Expected information gain (exploration)
- The balance between these components can be tuned

### Belief Updating
Agents continuously update their beliefs about the environment:

```python
def update_resource_location_belief(self, pos, resource_type, probability):
    """Update belief about a specific resource at a specific location"""
    if pos not in self.resource_location_beliefs:
        self.resource_location_beliefs[pos] = {}
    
    # Update with some memory decay
    if resource_type in self.resource_location_beliefs[pos]:
        current_belief = self.resource_location_beliefs[pos][resource_type]
        # Belief updates with learning rate
        learning_rate = 0.3
        self.resource_location_beliefs[pos][resource_type] = \
            current_belief * (1 - learning_rate) + probability * learning_rate
    else:
        self.resource_location_beliefs[pos][resource_type] = probability
```

### Dynamic Trading Network
The trading network evolves as agents interact:

```python
# Update trading history after successful trade
self.agent_trading_history[partner_id] = min(0.95, 
                                           self.agent_trading_history[partner_id] + 0.1)
partner.agent_trading_history[self.unique_id] = min(0.95, 
                                                  partner.agent_trading_history[self.unique_id] + 0.1)

# Add to trading partners
self.trading_partners.add(partner_id)
partner.trading_partners.add(self.unique_id)
```

## Visualization
The project includes comprehensive visualization tools:

1. **Grid Visualization**: Shows the position of agents and resources
2. **Network Visualization**: Displays the trading relationships between agents
3. **Wealth Distribution**: Shows the distribution of wealth across agents
4. **Resource Distribution**: Tracks the availability of different resource types

## Analysis
The analysis tools help understand the emergent economic behaviors:

1. **Wealth Correlations**: Analyzes how different specializations correlate with wealth
2. **Trading Patterns**: Examines the formation of trading networks
3. **EFE Breakdown**: Shows how different components of EFE influence decision-making
4. **Resource Flow**: Tracks how resources move through the economy

## Economic Principles Demonstrated
This simulation demonstrates several key economic principles:

1. **Comparative Advantage**: Agents specialize based on their abilities and trade for mutual benefit
2. **Market Formation**: Trading networks emerge organically based on agent interactions
3. **Price Discovery**: Resource values emerge based on supply, demand, and preferences
4. **Division of Labor**: Agents naturally focus on activities where they have advantages
5. **Information as Value**: Exploration has intrinsic value in the active inference framework

## Theoretical Background
This project is based on active inference theory, which is a framework for understanding action, perception, and learning that has roots in neuroscience and has recently been applied to artificial agents and economic modeling.

Key papers for further reading:
- Friston, K. (2010). "The free-energy principle: a unified brain theory?"
- Parr, T., & Friston, K. J. (2019). "Generalised free energy and active inference."
- Smith, R., Friston, K., & Whyte, C. (2022). "A step-by-step tutorial on active inference and its application to empirical data."

## Future Extensions
Possible extensions to this project:
1. Hierarchical generative models
2. Cultural transmission of knowledge
3. More complex resource production chains
4. Institution formation (markets, contracts)
5. Environmental changes and adaptation

## Contribution to Masters Applications
This project demonstrates:
1. Understanding of advanced AI architectures beyond standard machine learning
2. Implementation skills in agent-based modeling
3. Integration of economic principles with computational frameworks
4. Data analysis and visualization capabilities
5. Formal mathematical understanding of decision-making under uncertainty

## License
MIT License
