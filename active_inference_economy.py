"""
Active Inference Foraging and Trading Economy

A multi-agent economic simulation where agents use active inference principles (Expected Free Energy)
to make decisions about resource foraging and trading. Features grid-based world, probabilistic
resource generation, agent specializations, time constraints, and dynamic trading networks.

Created by: You (with guidance from Claude)
Date: May, 2025
"""

import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import math

# Resource Classes
class Resource:
    """Base resource class"""
    def __init__(self, resource_type, value, position=None):
        self.resource_type = resource_type
        self.base_value = value
        self.position = position
        
    def __repr__(self):
        return f"{self.resource_type} (Value: {self.base_value})"

class BasicResource(Resource):
    """Simple resource with fixed value"""
    def __init__(self, resource_type, value, position=None):
        super().__init__(resource_type, value, position)

class ComplementaryResource(Resource):
    """Resource that provides bonus value when combined with certain other resources"""
    def __init__(self, resource_type, value, complements, position=None):
        super().__init__(resource_type, value, position)
        # Dictionary of resources this complements and the bonus value
        self.complements = complements
        
    def get_combined_value(self, other_resource):
        """Calculate the combined value when paired with another resource"""
        if other_resource.resource_type in self.complements:
            return self.base_value + other_resource.base_value + self.complements[other_resource.resource_type]
        return self.base_value + other_resource.base_value

# Agent Class
class ForagingAgent(Agent):
    """
    Agent that uses Active Inference with Expected Free Energy minimization
    to make decisions about foraging and trading.
    """
    def __init__(self, unique_id, model, 
                 movement_efficiency=1.0, 
                 harvesting_efficiency=1.0,
                 trading_advantage=1.0,
                 resource_preferences=None):
        super().__init__(unique_id, model)
        
        # Specializations
        self.movement_efficiency = movement_efficiency  # Affects movement cost
        self.harvesting_efficiency = harvesting_efficiency  # Affects resource collection probability
        self.trading_advantage = trading_advantage  # Affects trade success and value perception
        
        # Resource preferences (utility multipliers)
        self.resource_preferences = resource_preferences if resource_preferences else {"wood": 1.0, "stone": 1.0, "food": 1.0, "metal": 1.0, "gem": 1.0}
        
        # Agent state
        self.inventory = defaultdict(int)
        self.energy = 100
        self.wealth = 0
        
        # Beliefs about the environment (generative model)
        self.resource_location_beliefs = {}  # Beliefs about where resources are
        self.resource_value_beliefs = {}     # Beliefs about resource values
        self.agent_trading_history = defaultdict(lambda: 0.5)  # Prior belief of 0.5 for unknown agents
        
        # Trading network
        self.trading_partners = set()
        self.trade_cooldown = 0
        
    def get_resource_utility(self, resource_type, quantity=1):
        """Calculate the utility of a resource to this agent based on preferences"""
        base_value = self.model.resource_values.get(resource_type, 1.0)
        preference_multiplier = self.resource_preferences.get(resource_type, 1.0)
        return base_value * preference_multiplier * quantity
    
    def expected_free_energy(self, action, current_state):
        """
        Calculate the expected free energy for an action
        Lower EFE is better (we're minimizing expected surprise)
        """
        # Components of EFE
        expected_utility = 0
        expected_information_gain = 0
        
        # Different action types
        if action[0] == "move":
            direction = action[1]
            # Calculate costs of movement
            movement_cost = 1.0 / self.movement_efficiency
            expected_utility -= movement_cost
            
            # Calculate potential gains from exploring new areas
            new_pos = self.get_new_position(direction)
            if new_pos in self.resource_location_beliefs:
                for res_type, belief in self.resource_location_beliefs[new_pos].items():
                    expected_util = belief * self.get_resource_utility(res_type)
                    expected_utility += expected_util
                    
                # Add information gain component - how much can we learn?
                uncertainty = sum([abs(0.5 - belief) for belief in self.resource_location_beliefs[new_pos].values()])
                expected_information_gain += (1.0 - uncertainty) * 2
            else:
                # Unknown position - high information gain potential
                expected_information_gain += 2.0
                
        elif action[0] == "harvest":
            resource_type = action[1]
            # Calculate utility from harvesting
            harvest_prob = min(0.9, self.harvesting_efficiency * 0.5)
            expected_utility += harvest_prob * self.get_resource_utility(resource_type)
            
            # Calculate information gain (learning about resource distribution)
            expected_information_gain += 0.5
            
        elif action[0] == "trade":
            partner_offer_utility = partner.get_resource_utility(request_resource)
                    partner_request_utility = partner.get_resource_utility(offer_resource)
                    
                    # Trade happens if partner also benefits or is influenced by our trading advantage
                    trade_threshold = 1.0
                    if self.trading_advantage > 1.0:
                        trade_threshold = 1.0 / self.trading_advantage
                        
                    if partner_request_utility >= partner_offer_utility * trade_threshold:
                        # Execute trade
                        self.inventory[offer_resource] -= 1
                        self.inventory[request_resource] += 1
                        partner.inventory[request_resource] -= 1
                        partner.inventory[offer_resource] += 1
                        
                        # Update trading history
                        self.agent_trading_history[partner_id] = min(0.95, self.agent_trading_history[partner_id] + 0.1)
                        partner.agent_trading_history[self.unique_id] = min(0.95, partner.agent_trading_history[self.unique_id] + 0.1)
                        
                        # Add to trading partners
                        self.trading_partners.add(partner_id)
                        partner.trading_partners.add(self.unique_id)
                        
                        # Update wealth
                        utility_change = self.get_resource_utility(request_resource) - self.get_resource_utility(offer_resource)
                        self.wealth += utility_change
                    else:
                        # Failed trade attempt
                        self.agent_trading_history[partner_id] = max(0.05, self.agent_trading_history[partner_id] - 0.05)
                
                # Set trade cooldown
                self.trade_cooldown = 3
    
    def update_resource_beliefs(self):
        """Update beliefs about resources based on current observations"""
        # What the agent can see in current position
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        resources_here = {obj.resource_type: 1.0 for obj in cell_contents if isinstance(obj, Resource)}
        
        # Update beliefs for current position
        if self.pos not in self.resource_location_beliefs:
            self.resource_location_beliefs[self.pos] = {}
            
        for res_type in self.model.resource_types:
            if res_type in resources_here:
                self.update_resource_location_belief(self.pos, res_type, 0.9)  # High confidence when we see it
            else:
                self.update_resource_location_belief(self.pos, res_type, 0.1)  # Low probability if we don't see it
    
    def update_resource_location_belief(self, pos, resource_type, probability):
        """Update belief about a specific resource at a specific location"""
        if pos not in self.resource_location_beliefs:
            self.resource_location_beliefs[pos] = {}
        
        # Update with some memory decay
        if resource_type in self.resource_location_beliefs[pos]:
            current_belief = self.resource_location_beliefs[pos][resource_type]
            # Belief updates with learning rate
            learning_rate = 0.3
            self.resource_location_beliefs[pos][resource_type] = current_belief * (1 - learning_rate) + probability * learning_rate
        else:
            self.resource_location_beliefs[pos][resource_type] = probability
    
    def step(self):
        """Agent's step function - select and execute action"""
        if self.energy <= 0:
            return  # Agent is exhausted
            
        # Decrease trade cooldown if active
        if self.trade_cooldown > 0:
            self.trade_cooldown -= 1
            
        # Select action based on expected free energy
        action = self.select_action()
        
        # Execute the selected action
        self.execute_action(action)
        
        # Basic energy regeneration
        self.energy = min(100, self.energy + 1)


# Model Class
class ForagingTradingEconomy(Model):
    """
    Model class for the Foraging and Trading Economy simulation.
    Handles resource distribution, agent interactions, and data collection.
    """
    def __init__(self, width=20, height=20, num_agents=10, 
                 resource_distribution=None, trade_radius=2):
        super().__init__()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.trade_radius = trade_radius
        
        # Resource types and values
        self.resource_types = ["wood", "stone", "food", "metal", "gem"]
        self.resource_values = {
            "wood": 2,
            "stone": 3,
            "food": 4,
            "metal": 6,
            "gem": 10
        }
        
        # Resource complementary relationships
        self.complementary_resources = {
            "wood": {"stone": 2},  # Wood + Stone = +2 bonus value
            "stone": {"metal": 3},  # Stone + Metal = +3 bonus value
            "metal": {"gem": 5}     # Metal + Gem = +5 bonus value
        }
        
        # Probabilistic resource distribution
        if resource_distribution is None:
            self.resource_distribution = {
                "wood": 0.1,   # 10% chance per cell
                "stone": 0.07,
                "food": 0.08,
                "metal": 0.04,
                "gem": 0.02
            }
        else:
            self.resource_distribution = resource_distribution
            
        # Initialize agents
        for i in range(num_agents):
            # Create agent with random specializations
            movement_eff = np.random.uniform(0.8, 1.6)
            harvesting_eff = np.random.uniform(0.8, 1.6)
            trading_adv = np.random.uniform(0.8, 1.6)
            
            # Generate resource preferences (each agent values resources differently)
            preferences = {}
            for res_type in self.resource_types:
                preferences[res_type] = np.random.uniform(0.5, 1.5)
            
            # Create the agent
            agent = ForagingAgent(i, self, movement_eff, harvesting_eff, 
                                 trading_adv, preferences)
            
            # Place agent randomly
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)
            
        # Initialize resources
        self.initialize_resources()
        
        # Data collection for visualization
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Wealth": self.calculate_total_wealth,
                "Resource_Distribution": self.get_resource_distribution,
                "Trade_Network": self.get_trade_network,
                "Agent_Specializations": self.get_agent_specializations
            },
            agent_reporters={
                "Wealth": lambda a: a.wealth,
                "Inventory": lambda a: dict(a.inventory),
                "Energy": lambda a: a.energy,
                "Position": lambda a: a.pos
            }
        )
        
    def initialize_resources(self):
        """Place resources on the grid according to probability distribution"""
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                for res_type in self.resource_types:
                    if self.random.random() < self.resource_distribution[res_type]:
                        # Determine if this is a complementary resource
                        if res_type in self.complementary_resources:
                            complements = self.complementary_resources[res_type]
                            resource = ComplementaryResource(res_type, self.resource_values[res_type], 
                                                           complements, (x, y))
                        else:
                            resource = BasicResource(res_type, self.resource_values[res_type], (x, y))
                        
                        self.grid.place_agent(resource, (x, y))
    
    def calculate_total_wealth(self):
        """Calculate total wealth across all agents"""
        return sum(agent.wealth for agent in self.schedule.agents)
    
    def get_resource_distribution(self):
        """Get current resource distribution"""
        distribution = {res_type: 0 for res_type in self.resource_types}
        
        for agent in self.grid.get_all_cell_contents():
            if isinstance(agent, Resource):
                distribution[agent.resource_type] += 1
                
        return distribution
    
    def get_trade_network(self):
        """Get the current trading network as list of edges"""
        edges = []
        for agent in self.schedule.agents:
            for partner_id in agent.trading_partners:
                edges.append((agent.unique_id, partner_id))
        return edges
    
    def get_agent_specializations(self):
        """Get agent specialization data for visualization"""
        specializations = []
        for agent in self.schedule.agents:
            spec = {
                "id": agent.unique_id,
                "movement": agent.movement_efficiency,
                "harvesting": agent.harvesting_efficiency,
                "trading": agent.trading_advantage,
                "preferences": agent.resource_preferences
            }
            specializations.append(spec)
        return specializations
    
    def resource_regrowth(self):
        """Handle probabilistic resource regrowth"""
        empty_cells = []
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell_contents = self.grid.get_cell_list_contents([(x, y)])
                if not any(isinstance(obj, Resource) for obj in cell_contents):
                    empty_cells.append((x, y))
        
        # Add new resources with low probability
        for cell in empty_cells:
            for res_type in self.resource_types:
                if self.random.random() < self.resource_distribution[res_type] * 0.2:  # Lower regrowth rate
                    if res_type in self.complementary_resources:
                        complements = self.complementary_resources[res_type]
                        resource = ComplementaryResource(res_type, self.resource_values[res_type], 
                                                       complements, cell)
                    else:
                        resource = BasicResource(res_type, self.resource_values[res_type], cell)
                    
                    self.grid.place_agent(resource, cell)
                    break  # Only one resource per cell during regrowth
    
    def step(self):
        """Model step function"""
        self.schedule.step()
        self.datacollector.collect(self)
        
        # Resource regrowth every 5 steps
        if self.schedule.steps % 5 == 0:
            self.resource_regrowth()


# Visualization and Analysis
class GridVisualization:
    """Custom visualization module for the grid-based economy"""
    def __init__(self, grid_width, grid_height, canvas_width=500, canvas_height=500):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Resource colors
        self.resource_colors = {
            "wood": "#8B4513",  # Brown
            "stone": "#A9A9A9",  # Gray
            "food": "#228B22",  # Green
            "metal": "#C0C0C0",  # Silver
            "gem": "#4B0082"   # Indigo
        }
        
    def render(self, model):
        """Render the model visualization dashboard"""
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Grid visualization
        ax1 = fig.add_subplot(221)
        self.draw_grid(model, ax1)
        
        # Network visualization
        ax2 = fig.add_subplot(222)
        self.draw_network(model, ax2)
        
        # Wealth distribution
        ax3 = fig.add_subplot(223)
        self.draw_wealth_distribution(model, ax3)
        
        # Resource distribution
        ax4 = fig.add_subplot(224)
        self.draw_resource_distribution(model, ax4)
        
        plt.tight_layout()
        return fig
        
    def draw_grid(self, model, ax):
        """Draw the grid with agents and resources"""
        # Create a matrix to represent the grid
        grid_matrix = np.zeros((model.grid.width, model.grid.height, 3))  # RGB values
        
        # Fill with resources
        for x in range(model.grid.width):
            for y in range(model.grid.height):
                cell_contents = model.grid.get_cell_list_contents([(x, y)])
                for obj in cell_contents:
                    if isinstance(obj, Resource):
                        color_hex = self.resource_colors.get(obj.resource_type, "#000000")
                        # Convert hex to RGB
                        r = int(color_hex[1:3], 16) / 255.0
                        g = int(color_hex[3:5], 16) / 255.0
                        b = int(color_hex[5:7], 16) / 255.0
                        grid_matrix[x, y] = [r, g, b]
        
        # Plot the grid
        ax.imshow(np.transpose(grid_matrix, (1, 0, 2)))
        
        # Add agents as scatter points
        agent_positions = [(agent.pos[0], agent.pos[1]) for agent in model.schedule.agents]
        x_pos = [pos[0] for pos in agent_positions]
        y_pos = [pos[1] for pos in agent_positions]
        
        # Color agents by their primary specialization
        colors = []
        for agent in model.schedule.agents:
            specs = [agent.movement_efficiency, agent.harvesting_efficiency, agent.trading_advantage]
            max_spec = max(specs)
            if max_spec == agent.movement_efficiency:
                colors.append('blue')  # Movement specialists
            elif max_spec == agent.harvesting_efficiency:
                colors.append('red')   # Harvesting specialists
            else:
                colors.append('yellow')  # Trading specialists
        
        ax.scatter(x_pos, y_pos, c=colors, edgecolors='white', s=50)
        ax.set_title("Economy Grid")
        ax.set_xticks([])
        ax.set_yticks([])
        
    def draw_network(self, model, ax):
        """Draw the trading network"""
        G = nx.Graph()
        
        # Add nodes
        for agent in model.schedule.agents:
            G.add_node(agent.unique_id)
        
        # Add edges from trading partners
        edges = model.get_trade_network()
        G.add_edges_from(edges)
        
        # Node colors based on specialization
        node_colors = []
        for agent in model.schedule.agents:
            specs = [agent.movement_efficiency, agent.harvesting_efficiency, agent.trading_advantage]
            max_spec = max(specs)
            if max_spec == agent.movement_efficiency:
                node_colors.append('blue')  # Movement specialists
            elif max_spec == agent.harvesting_efficiency:
                node_colors.append('red')   # Harvesting specialists
            else:
                node_colors.append('yellow')  # Trading specialists
        
        # Position nodes based on their grid position
        pos = {agent.unique_id: (agent.pos[0], agent.pos[1]) for agent in model.schedule.agents}
        
        nx.draw(G, pos, ax=ax, node_color=node_colors, with_labels=True, node_size=300, font_size=10)
        ax.set_title("Trading Network")
        
    def draw_wealth_distribution(self, model, ax):
        """Draw wealth distribution across agents"""
        agent_ids = [agent.unique_id for agent in model.schedule.agents]
        agent_wealth = [agent.wealth for agent in model.schedule.agents]
        
        # Sort by wealth for better visualization
        sorted_indices = np.argsort(agent_wealth)
        agent_ids = [agent_ids[i] for i in sorted_indices]
        agent_wealth = [agent_wealth[i] for i in sorted_indices]
        
        ax.bar(agent_ids, agent_wealth)
        ax.set_title("Agent Wealth Distribution")
        ax.set_xlabel("Agent ID")
        ax.set_ylabel("Wealth")
        
    def draw_resource_distribution(self, model, ax):
        """Draw resource distribution in the economy"""
        resource_dist = model.get_resource_distribution()
        resources = list(resource_dist.keys())
        counts = list(resource_dist.values())
        
        colors = [self.resource_colors.get(res, "#000000") for res in resources]
        
        ax.bar(resources, counts, color=colors)
        ax.set_title("Resource Distribution")
        ax.set_xlabel("Resource Type")
        ax.set_ylabel("Count")


# Simulation and Analysis Functions
def run_simulation(steps=100, visualize_results=True):
    """Run a simulation for a fixed number of steps and analyze results"""
    model = ForagingTradingEconomy(width=20, height=20, num_agents=10, trade_radius=2)
    
    # Store data for analysis
    wealth_history = []
    resource_history = []
    trade_counts = []
    
    for i in range(steps):
        model.step()
        
        # Collect data each step
        wealth_history.append(model.calculate_total_wealth())
        resource_history.append(model.get_resource_distribution())
        trade_counts.append(len(model.get_trade_network()))
        
        # Print progress
        if i % 10 == 0:
            print(f"Step {i} completed")
    
    if visualize_results:
        # Create visualization of simulation results
        fig = plt.figure(figsize=(15, 12))
        
        # Wealth over time
        ax1 = fig.add_subplot(221)
        ax1.plot(wealth_history)
        ax1.set_title("Total Economy Wealth Over Time")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Total Wealth")
        
        # Resource counts over time
        ax2 = fig.add_subplot(222)
        resource_data = {res_type: [res_hist[res_type] for res_hist in resource_history] 
                        for res_type in model.resource_types}
        
        for res_type, counts in resource_data.items():
            ax2.plot(counts, label=res_type)
        
        ax2.set_title("Resource Counts Over Time")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Count")
        ax2.legend()
        
        # Trading activity over time
        ax3 = fig.add_subplot(223)
        ax3.plot(trade_counts)
        ax3.set_title("Trading Activity Over Time")
        ax3.set_xlabel("Steps")
        ax3.set_ylabel("Number of Trading Relationships")
        
        # Final agent analysis
        ax4 = fig.add_subplot(224)
        
        # Analyze correlations between specializations and wealth
        movement_specs = [agent.movement_efficiency for agent in model.schedule.agents]
        harvest_specs = [agent.harvesting_efficiency for agent in model.schedule.agents]
        trading_specs = [agent.trading_advantage for agent in model.schedule.agents]
        wealth = [agent.wealth for agent in model.schedule.agents]
        
        # Calculate correlations
        corr_movement = np.corrcoef(movement_specs, wealth)[0, 1]
        corr_harvest = np.corrcoef(harvest_specs, wealth)[0, 1]
        corr_trading = np.corrcoef(trading_specs, wealth)[0, 1]
        
        # Plot correlations
        specs = ['Movement', 'Harvesting', 'Trading']
        corrs = [corr_movement, corr_harvest, corr_trading]
        
        ax4.bar(specs, corrs)
        ax4.set_title("Correlation between Specializations and Wealth")
        ax4.set_ylabel("Correlation Coefficient")
        
        plt.tight_layout()
        plt.savefig('simulation_results.png')
        plt.show()
    
    # Return the final model state for further analysis
    return model

def analyze_agent_behavior(model):
    """Analyze the behavior of agents to show active inference principles"""
    # Get agents sorted by wealth
    agents = sorted(model.schedule.agents, key=lambda a: a.wealth, reverse=True)
    
    # Analyze top 3 and bottom 3 agents
    top_agents = agents[:3]
    bottom_agents = agents[-3:]
    
    print("Analysis of Agent Behavior\n")
    print("Top 3 Agents by Wealth:")
    for agent in top_agents:
        print(f"Agent {agent.unique_id}:")
        print(f"  Wealth: {agent.wealth:.2f}")
        print(f"  Specializations: Movement={agent.movement_efficiency:.2f}, " + 
              f"Harvesting={agent.harvesting_efficiency:.2f}, " +
              f"Trading={agent.trading_advantage:.2f}")
        print(f"  Resource Preferences: {', '.join([f'{k}={v:.2f}' for k, v in agent.resource_preferences.items()])}")
        print(f"  Inventory: {dict(agent.inventory)}")
        print(f"  Trading Partners: {len(agent.trading_partners)}")
        print()
    
    print("Bottom 3 Agents by Wealth:")
    for agent in bottom_agents:
        print(f"Agent {agent.unique_id}:")
        print(f"  Wealth: {agent.wealth:.2f}")
        print(f"  Specializations: Movement={agent.movement_efficiency:.2f}, " + 
              f"Harvesting={agent.harvesting_efficiency:.2f}, " +
              f"Trading={agent.trading_advantage:.2f}")
        print(f"  Resource Preferences: {', '.join([f'{k}={v:.2f}' for k, v in agent.resource_preferences.items()])}")
        print(f"  Inventory: {dict(agent.inventory)}")
        print(f"  Trading Partners: {len(agent.trading_partners)}")
        print()
    
    # Analyze how different agent specializations affect EFE calculations
    print("Expected Free Energy Analysis:")
    test_agent = agents[0]  # Use the most successful agent
    
    # Create some test scenarios to demonstrate EFE calculations
    current_state = {
        "pos": test_agent.pos,
        "inventory": dict(test_agent.inventory),
        "energy": test_agent.energy
    }
    
    # Test a few action types
    test_actions = [
        ("move", "up"),
        ("harvest", "wood"),
        ("trade", 1, "wood", "gem")
    ]
    
    for action in test_actions:
        efe = test_agent.expected_free_energy(action, current_state)
        print(f"Action: {action}")
        print(f"EFE Value: {efe:.2f}")
        
        # Break down the components
        if action[0] == "move":
            move_cost = 1.0 / test_agent.movement_efficiency
            print(f"  Movement Cost Component: {move_cost:.2f}")
            print(f"  Exploration Value: Depends on agent's beliefs about what's in that direction")
        
        elif action[0] == "harvest":
            harvest_prob = min(0.9, test_agent.harvesting_efficiency * 0.5)
            resource_utility = test_agent.get_resource_utility(action[1])
            expected_utility = harvest_prob * resource_utility
            print(f"  Harvest Probability: {harvest_prob:.2f}")
            print(f"  Resource Utility: {resource_utility:.2f}")
            print(f"  Expected Utility: {expected_utility:.2f}")
        
        elif action[0] == "trade":
            partner_id, offer_res, request_res = action[1], action[2], action[3]
            trade_success_prob = test_agent.agent_trading_history.get(partner_id, 0.5)
            utility_gain = test_agent.get_resource_utility(request_res) - test_agent.get_resource_utility(offer_res)
            expected_utility = trade_success_prob * utility_gain * test_agent.trading_advantage
            
            print(f"  Trading Partner: Agent {partner_id}")
            print(f"  Trade Success Probability: {trade_success_prob:.2f}")
            print(f"  Utility Gain: {utility_gain:.2f}")
            print(f"  Trading Advantage: {test_agent.trading_advantage:.2f}")
            print(f"  Expected Utility: {expected_utility:.2f}")
        
        print()
    
    return

# Running the Model
def run_server():
    """Run the simulation with Mesa's visualization server"""
    from mesa.visualization.ModularVisualization import ModularServer
    from mesa.visualization.modules import CanvasGrid, ChartModule
    from mesa.visualization.UserParam import UserSettableParameter
    
    def agent_portrayal(agent):
        """Define how to portray each agent in the grid"""
        if isinstance(agent, ForagingAgent):
            portrayal = {
                "Shape": "circle",
                "Color": "blue",
                "Filled": "true",
                "Layer": 1,
                "r": 0.5
            }
            
            # Color based on specialization
            specs = [agent.movement_efficiency, agent.harvesting_efficiency, agent.trading_advantage]
            max_spec = max(specs)
            if max_spec == agent.movement_efficiency:
                portrayal["Color"] = "blue"
            elif max_spec == agent.harvesting_efficiency:
                portrayal["Color"] = "red"
            else:
                portrayal["Color"] = "yellow"
                
            return portrayal
        elif isinstance(agent, Resource):
            portrayal = {
                "Shape": "rect",
                "Color": "#8B4513",  # Default brown
                "Filled": "true",
                "Layer": 0,
                "w": 0.7,
                "h": 0.7
            }
            
            # Resource-specific colors
            resource_colors = {
                "wood": "#8B4513",
                "stone": "#A9A9A9",
                "food": "#228B22",
                "metal": "#C0C0C0",
                "gem": "#4B0082"
            }
            
            portrayal["Color"] = resource_colors.get(agent.resource_type, "#000000")
            
            return portrayal
        return None
    
    # Create a grid display
    grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)
    
    # Create a chart display
    chart = ChartModule([
        {"Label": "Total_Wealth", "Color": "Black"}
    ])
    
    # Model parameters
    model_params = {
        "width": 20,
        "height": 20,
        "num_agents": UserSettableParameter("slider", "Number of Agents", 10, 2, 30, 1),
        "trade_radius": UserSettableParameter("slider", "Trading Radius", 2, 1, 5, 1)
    }
    
    # Create the server
    server = ModularServer(
        ForagingTradingEconomy,
        [grid, chart],
        "Foraging and Trading Economy",
        model_params
    )
    
    server.port = 8521
    server.launch()

# Main function
def main():
    """Main function to run the simulation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Active Inference Economy Simulation')
    parser.add_argument('--server', action='store_true', help='Run with Mesa visualization server')
    parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps')
    parser.add_argument('--analyze', action='store_true', help='Run analysis after simulation')
    
    args = parser.parse_args()
    
    if args.server:
        run_server()
    else:
        print("Running Active Inference Foraging and Trading Economy Simulation...")
        model = run_simulation(steps=args.steps)
        
        if args.analyze:
            analyze_agent_behavior(model)
        
        print("\nSimulation complete!")

if __name__ == "__main__":
    main()id = action[1]
            offer_resource = action[2]
            request_resource = action[3]
            
            # Expected utility from trade
            trade_success_prob = self.agent_trading_history[partner_id]
            utility_gain = self.get_resource_utility(request_resource) - self.get_resource_utility(offer_resource)
            expected_utility += trade_success_prob * utility_gain * self.trading_advantage
            
            # Information gain about trading partner
            if abs(trade_success_prob - 0.5) < 0.1:  # Uncertain about partner
                expected_information_gain += 1.0
            else:
                expected_information_gain += 0.2  # Already know this partner well
        
        # Combine components with a weighting parameter (can be tuned)
        explore_exploit_balance = 0.7  # Higher values favor exploitation over exploration
        
        # Lower is better for EFE (minimizing expected surprise)
        efe = -(explore_exploit_balance * expected_utility + (1 - explore_exploit_balance) * expected_information_gain)
        return efe
    
    def get_new_position(self, direction):
        """Calculate new position based on direction"""
        x, y = self.pos
        if direction == "up":
            return (x, y + 1)
        elif direction == "down":
            return (x, y - 1)
        elif direction == "left":
            return (x - 1, y)
        elif direction == "right":
            return (x + 1, y)
        return (x, y)
    
    def get_possible_actions(self):
        """Get all possible actions the agent can take"""
        actions = []
        
        # Movement actions
        for direction in ["up", "down", "left", "right"]:
            actions.append(("move", direction))
            
        # Harvest actions if there are resources in cell
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        for obj in cell_contents:
            if isinstance(obj, Resource):
                actions.append(("harvest", obj.resource_type))
        
        # Trade actions with nearby agents
        if self.trade_cooldown == 0:
            neighbors = self.model.grid.get_neighbors(
                self.pos, moore=True, radius=self.model.trade_radius
            )
            for neighbor in neighbors:
                if isinstance(neighbor, ForagingAgent) and neighbor.unique_id != self.unique_id:
                    # Consider trading each resource type we have
                    for offer_res, amount in self.inventory.items():
                        if amount > 0:
                            # For each resource we don't have or want more of
                            for request_res in self.resource_preferences:
                                if self.resource_preferences[request_res] > self.resource_preferences[offer_res]:
                                    actions.append(("trade", neighbor.unique_id, offer_res, request_res))
        
        return actions
    
    def select_action(self):
        """Select the action with lowest expected free energy"""
        possible_actions = self.get_possible_actions()
        if not possible_actions:
            return None
            
        current_state = {
            "pos": self.pos,
            "inventory": dict(self.inventory),
            "energy": self.energy
        }
        
        # Calculate EFE for each action
        action_efes = [(action, self.expected_free_energy(action, current_state)) 
                        for action in possible_actions]
        
        # Select action with lowest EFE
        best_action = min(action_efes, key=lambda x: x[1])[0]
        return best_action
    
    def execute_action(self, action):
        """Execute the selected action"""
        if not action:
            return
            
        action_type = action[0]
        
        if action_type == "move":
            direction = action[1]
            new_pos = self.get_new_position(direction)
            
            # Check if new position is within grid bounds
            grid_width = self.model.grid.width
            grid_height = self.model.grid.height
            new_x, new_y = new_pos
            
            if 0 <= new_x < grid_width and 0 <= new_y < grid_height:
                self.model.grid.move_agent(self, new_pos)
                
                # Update beliefs based on what we see
                self.update_resource_beliefs()
                
                # Movement costs energy
                self.energy -= 1.0 / self.movement_efficiency
        
        elif action_type == "harvest":
            resource_type = action[1]
            cell_contents = self.model.grid.get_cell_list_contents([self.pos])
            
            for obj in cell_contents:
                if isinstance(obj, Resource) and obj.resource_type == resource_type:
                    # Probability of successful harvest based on efficiency
                    if np.random.random() < min(0.9, self.harvesting_efficiency * 0.5):
                        self.inventory[resource_type] += 1
                        self.model.grid.remove_agent(obj)
                        
                        # Update beliefs
                        self.update_resource_location_belief(self.pos, resource_type, 0.2)  # Reduce belief as we just harvested it
                        
                        # Harvesting costs energy
                        self.energy -= 2.0
                        
                        # Update wealth based on resource value and preference
                        self.wealth += self.get_resource_utility(resource_type)
                    break
        
        elif action_type == "trade":
            partner_id = action[1]
            offer_resource = action[2]
            request_resource = action[3]
            
            # Find the trading partner
            partner = None
            for agent in self.model.schedule.agents:
                if agent.unique_id == partner_id:
                    partner = agent
                    break
                    
            if partner and self.inventory[offer_resource] > 0:
                # Check if partner has the requested resource
                if partner.inventory[request_resource] > 0:
                    # Determine if trade is beneficial for partner
                    partner_offer_utility = partner.get_resource_utility(request_resource)
                    partner_request_utility = partner.get_resource_utility(offer_resource)
                    
                    # Trade happens if partner also benefits or