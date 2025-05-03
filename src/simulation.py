import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from environment import GridWorld
from agents import RandomAgent, GreedyAgent
import os

def run_sim(n_agents: int = 20, steps: int = 200):
    # Initialize environment
    env = GridWorld(10, 0.05)
    grid_size = env.size
    
    # Create agents (equal numbers of each type)
    agents = []
    n_per_type = n_agents // 3
    
    for i in range(n_per_type):
        agents.append(RandomAgent(i+1))
        agents.append(GreedyAgent(i+1+n_per_type))
        agents.append(ActiveInferenceAgent(i+1+2*n_per_type))
    
    # Initialize wealth tracking
    wealth = {agent.id: 0 for agent in agents}
    
    # Storage for animation frames
    frames = np.zeros((steps, grid_size, grid_size))
    
    # Run simulation
    for step in range(steps):
        # Store current state
        frames[step] = env.get_state()
        
        # Each agent takes action
        for agent in agents:
            action = agent.act(env)
            reward = env.step(agent.id, action)
            wealth[agent.id] += reward
            
    # Animation function
    def animate(frame):
        plt.clf()
        plt.imshow(frames[frame], cmap='YlOrRd')
        
        # Add agent markers
        for agent in agents:
            pos = np.where(frames[frame] == agent.id)
            if len(pos[0]) > 0:
                marker = '🤖' if isinstance(agent, RandomAgent) else '🧠' if isinstance(agent, GreedyAgent) else '🎯'
                plt.plot(pos[1], pos[0], marker=marker, markersize=15, color='none')
        
        plt.grid(True)
        plt.title(f'Step {frame}')
        
    # Create animation
    fig = plt.figure(figsize=(8, 8))
    anim = FuncAnimation(fig, animate, frames=steps, 
                        interval=100, repeat=False)
    
    # Save animation
    anim.save('run.gif', writer='pillow')
    
    # Display live animation
    plt.show()
    
    return frames, wealth
