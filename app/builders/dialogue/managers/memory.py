"""
Initialization by ChatGPT -> unverified
"""
import time
import math


# Initialize constants
EPSILON = 0.1           # Threshold below which a memory is forgotten
DECAY_RATE = 0.05       # Controls how fast the memory decays
UPDATE_INTERVAL = 1     # Time interval to update weights (seconds)


class Memory:

    def __init__(self, content, initial_weight, timestamp=None):
        self.content = content
        self.initial_weight = initial_weight
        self.timestamp = timestamp if timestamp else time.time()
        self.current_weight = initial_weight

    def decay_weight(self, current_time):
        """
        Exponential decay of memory weight.
        """
        time_diff = current_time - self.timestamp
        self.current_weight = self.initial_weight * math.exp(-DECAY_RATE * time_diff)

    def is_forgotten(self):
        """
        Check if the memory weight is below the threshold.
        """
        return self.current_weight < EPSILON


class MemoryStorage:

    def __init__(self):
        self.memories = []

    def add_memory(self, memory_content, initial_weight):
        """
        Add a new memory to the storage.
        """
        memory = Memory(memory_content, initial_weight)
        self.memories.append(memory)
        print(f"Memory added: {memory_content}")

    def update_memories(self):
        """
        Update the weights of all memories and remove forgotten ones.
        """
        current_time = time.time()
        for memory in self.memories[:]:
            memory.decay_weight(current_time)
            if memory.is_forgotten():
                print(f"Memory forgotten: {memory.content}")
                self.memories.remove(memory)

    def display_memories(self):
        """
        Display current memories and their weights.
        """
        for memory in self.memories:
            print(f"Memory: {memory.content}, Weight: {memory.current_weight:.2f}")


if __name__ == "__main__":

    storage = MemoryStorage()

    # Add some initial memories
    storage.add_memory("Had lunch with friends", initial_weight=1.0)
    storage.add_memory("Learned Python", initial_weight=0.9)
    storage.add_memory("Attended family event", initial_weight=0.8)

    # Simulate memory decay over time
    for i in range(30):
        time.sleep(UPDATE_INTERVAL)  # Simulate the passage of time
        storage.update_memories()  # Update memory weights
        storage.display_memories()
        print("---")
