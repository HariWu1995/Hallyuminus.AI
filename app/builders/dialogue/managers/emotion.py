"""
Initialization by ChatGPT -> unverified
"""
import time
import random


class Talker:

    def __init__(self, name, initial_emotion="neutral"):
        self.name = name
        self.emotion = initial_emotion
        self.emotion_intensity = 0.5    # Intensity ranges from 0 to 1
        self.history = []               # Keeps track of past emotions and interactions
        self.last_update = time.time()

    def update_emotion(self, new_emotion, modifier=0.2):
        """
        Update the emotional state of the talker based on the new input.
        """
        current_time = time.time()
        time_diff = current_time - self.last_update

        # Apply a decay factor to emotion intensity
        self.emotion_intensity = max(self.emotion_intensity - (time_diff * 0.01), 0)

        # Modify emotion intensity
        if new_emotion == self.emotion:
            # Intensify the current emotion
            self.emotion_intensity = min(self.emotion_intensity + modifier, 1.0)

        else:
            # Switch emotion and reset intensity
            self.emotion = new_emotion
            self.emotion_intensity = min(modifier, 1.0)

        self.history.append((self.emotion, self.emotion_intensity, time.ctime()))
        self.last_update = current_time

    def get_temperament(self):
        """
        Return the talker's current emotion and intensity.
        """
        return (self.emotion, self.emotion_intensity)


class EmotionManager:

    def __init__(self):
        self.talkers = {}

    def add_talker(self, name, initial_emotion="neutral"):
        """
        Add a new talker to the conversation.
        """
        talker = Talker(name, initial_emotion)
        self.talkers[name] = talker

    def update_talker_emotion(self, name, new_emotion, modifier=0.2):
        """
        Update the emotion of a specific talker.
        """
        if name in self.talkers:
            self.talkers[name].update_emotion(new_emotion, modifier)

    def manage_interaction(self, talker1, talker2, action):
        """
        Manage emotions between two talkers based on interaction type.
        """
        talker1_emotion, t1_intensity = self.talkers[talker1].get_temperament()
        talker2_emotion, t2_intensity = self.talkers[talker2].get_temperament()

        # Define some basic interaction rules for emotion changes
        if action == "argument":
            self.talkers[talker1].update_emotion("angry", modifier=0.3)
            self.talkers[talker2].update_emotion("defensive", modifier=0.3)

        elif action == "comfort":
            self.talkers[talker1].update_emotion("calm", modifier=0.4)
            self.talkers[talker2].update_emotion("relieved", modifier=0.4)

        elif action == "praise":
            self.talkers[talker1].update_emotion("happy", modifier=0.5)
            self.talkers[talker2].update_emotion("appreciated", modifier=0.5)

        # Log the interaction
        print(f"{talker1} ({talker1_emotion}, {t1_intensity:.2f}) and {talker2} ({talker2_emotion}, {t2_intensity:.2f}) engaged in a '{action}'.")

    def get_talker_emotion(self, name):
        """
        Retrieve the current emotion and intensity of a talker.
        """
        if name in self.talkers:
            return self.talkers[name].get_temperament()

    def decay_emotions(self):
        """
        Apply decay to all talkers' emotions over time.
        """
        for talker in self.talkers.values():
            talker.update_emotion(talker.emotion, modifier=-0.05)  # Decay the intensity gradually

    def display_talker_states(self):
        """
        Display the current emotional states of all talkers.
        """
        for name, talker in self.talkers.items():
            emotion, intensity = talker.get_temperament()
            print(f"{name} is feeling {emotion} with intensity {intensity:.2f}")


if __name__ == "__main__":
    
    emotion_manager = EmotionManager()
    emotion_manager.add_talker("Alice")
    emotion_manager.add_talker("Bob", initial_emotion="happy")

    emotion_manager.manage_interaction("Alice", "Bob", "argument")
    emotion_manager.manage_interaction("Alice", "Bob", "comfort")
    
    # Simulate time passing and decaying emotions
    time.sleep(2)
    emotion_manager.decay_emotions()
    emotion_manager.display_talker_states()

