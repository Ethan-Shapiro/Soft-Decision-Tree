import matplotlib.pyplot as plt
import numpy as np
import torch


def extract_path_from_tensors(tensor_list):
    # Initialize the path with the root decision
    path = [tensor_list[0].item()]

    # This index tracks which node (by its decision index) we're at, starting at 0
    node_index = 0

    # Iterate over the remaining tensors, starting from the second one
    for level, decisions in enumerate(tensor_list[1:], start=1):
        # Each level doubles the number of nodes, so the index in the decisions tensor
        # for the current node is doubled, and then we add 0 for left or 1 for right
        # based on the decision made at the previous level.
        node_index = node_index * 2 + path[-1]

        # Add the decision for the current node to the path
        # We need to check if node_index is within the bounds of the decisions tensor
        if node_index < len(decisions):
            path.append(decisions[node_index].item())
        else:
            break  # If we're out of bounds, exit the loop

    return path


# Example tensor array
# This is the format decision_path is in
# So passing it is easy, the issue is just how many and which ones
tensor_array = [
    torch.tensor([1]),
    torch.tensor([1, 1]),
    torch.tensor([1, 0, 0, 1]),
    torch.tensor([0, 0, 0, 0, 0, 0, 1, 0]),
    torch.tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
]

path = extract_path_from_tensors(tensor_array)
path = [[i] for i in path]
print("Extracted Path:", path)


def plot_complete_tree_with_decision_path(decision_path):
    depth = len(decision_path)  # Determine the tree depth
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    # Function to plot nodes and edges
    def plot_node(pos, is_decision=False):
        color = 'red' if is_decision else 'blue'
        ax.plot(pos[0], -pos[1], 'o', color=color, markersize=4)  # Invert y-axis

    def plot_edge(start, end, is_decision=False):
        color = 'red' if is_decision else 'black'
        ax.plot([start[0], end[0]], [-start[1], -end[1]], color=color)  # Invert y-axis

    # Function to plot the complete tree
    def plot_tree(pos, level=0, spacing=2.0):
        if level < depth:
            # Plot left child
            x_delta = spacing / 2 ** level
            left_pos = (pos[0] - x_delta, pos[1] + 1)
            plot_edge(pos, left_pos)
            plot_node(left_pos)
            plot_tree(left_pos, level + 1, spacing)

            # Plot right child
            right_pos = (pos[0] + x_delta, pos[1] + 1)
            plot_edge(pos, right_pos)
            plot_node(right_pos)
            plot_tree(right_pos, level + 1, spacing)

    # Function to highlight the decision path
    def highlight_decision_path(pos, decision_path, level=0, spacing=2.0):
        if level < len(decision_path):
            decision = decision_path[level][0]  # Get the decision for the current level
            # Determine the next position based on the decision
            x_delta = spacing / 2 ** level
            next_pos = (pos[0] - x_delta, pos[1] + 1, pos[1] + 1) if decision == 0 else \
                       (pos[0] + x_delta, pos[1] + 1, pos[1] + 1)
            plot_node(pos, is_decision=True)  # Highlight the current node
            plot_edge(pos, next_pos, is_decision=True)  # Highlight the edge to the next node
            # Continue to the next level
            if level + 1 < len(decision_path):
                highlight_decision_path(next_pos, decision_path, level + 1, spacing)

    # Start plotting from the root
    root_pos = (0, 0)
    plot_node(root_pos)  # Plot the root node
    plot_tree(root_pos)  # Plot the entire tree
    highlight_decision_path(root_pos, decision_path)  # Highlight the decision path

    plt.xlim(-4, 4)
    plt.ylim(-depth, 1)
    plt.show()

# Example decision path structure
# Assuming each sublist represents the decision taken at each level (0 for left, 1 for right)
plot_complete_tree_with_decision_path(path)

# [0], [1, 1], [[1, 0], [1, 0]], [[[0, 1], [0, 0]], [[0, 1], [1, 0]]]