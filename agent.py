import torch  # Importe la bibliothèque PyTorch pour le calcul tensoriel
import random  # Importe le module pour la génération de nombres aléatoires
import numpy as np  # Importe le module NumPy pour les opérations sur les tableaux
from collections import deque  # Importe la structure de données deque pour stocker les expériences de l'agent
from game import Game, Direction, Point  # Importe les classes et les énumérations liées au jeu Snake
from model import Linear_QNet, QTrainer  # Importe les classes de réseau neuronal et d'entraîneur Q
from helper import plot  # Importe la fonction de traçage pour afficher les résultats

MAX_MEMORY = 100_000  # Définit la taille maximale de la mémoire pour stocker les expériences
BATCH_SIZE = 1000  # Définit la taille du lot pour l'apprentissage par lots
LR = 0.001  # Définit le taux d'apprentissage pour l'optimiseur

class Agent:

    def __init__(self):
        self.n_games = 0  # Initialise le nombre de jeux joués par l'agent
        self.epsilon = 0  # Initialise le paramètre epsilon pour l'exploration/exploitation
        self.gamma = 0.9  # Initialise le facteur de réduction pour les récompenses futures
        self.memory = deque(maxlen=MAX_MEMORY)  # Initialise la mémoire de l'agent avec une structure de données deque
        self.model = Linear_QNet(11, 256, 3)  # Initialise le modèle de réseau neuronal pour l'agent
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Initialise l'entraîneur Q pour le modèle


    def get_state(self, game):
        # Fonction pour obtenir l'état actuel du jeu
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Crée un vecteur représentant l'état actuel du jeu
        state = [
            # Danger en ligne droite
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger à droite
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger à gauche
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Direction de déplacement
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Emplacement de la nourriture
            game.food.x < game.head.x,  # nourriture à gauche
            game.food.x > game.head.x,  # nourriture à droite
            game.food.y < game.head.y,  # nourriture en haut
            game.food.y > game.head.y  # nourriture en bas
            ]

        return np.array(state, dtype=int)  # Retourne l'état sous forme de tableau NumPy

    def remember(self, state, action, reward, next_state, done):
        # Fonction pour enregistrer l'expérience dans la mémoire de l'agent
        self.memory.append((state, action, reward, next_state, done))  # Ajoute l'expérience à la mémoire

    def train_long_memory(self):
        # Fonction pour entraîner le modèle de l'agent sur des expériences mémorisées
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Échantillonne un mini-lot à partir de la mémoire si plus grand que BATCH_SIZE
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)  # Décompacte le mini-lot
        self.trainer.train_step(states, actions, rewards, next_states, dones)  # Entraîne le modèle avec le mini-lot

    def train_short_memory(self, state, action, reward, next_state, done):
        # Fonction pour entraîner le modèle de l'agent sur des expériences récentes
        self.trainer.train_step(state, action, reward, next_state, done)  # Entraîne le modèle avec une expérience récente

    def get_action(self, state):
        # Fonction pour choisir une action à partir de l'état actuel du jeu
        self.epsilon = 80 - self.n_games  # Met à jour le paramètre epsilon
        final_move = [0,0,0]  # Initialise le vecteur pour l'action finale

        if random.randint(0, 200) < self.epsilon:
            # Exploration : choisit une action aléatoire avec une probabilité epsilon
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation : choisit l'action avec la plus grande valeur prédite par le modèle
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move  # Retourne l'action finale


def train():
    # Fonction principale pour entraîner l'agent à jouer au jeu Snake
    plot_scores = []  # Initialise la liste pour stocker les scores de jeu
    plot_mean_scores = []  # Initialise la liste pour stocker les moyennes des scores de jeu
    total_score = 0  # Initialise le score total
    record = 0  # Initialise le record de score
    agent = Agent()  # Initialise l'agent
    game = Game()  # Initialise le jeu Snake
    while True:
        # Boucle de jeu principale
        state_old = agent.get_state(game)  # Obtient l'état actuel du jeu

        final_move = agent.get_action(state_old)  # Choix de l'action à effectuer

        reward, done, score = game.play_step(final_move)  # Effectue l'action dans le jeu

        state_new = agent.get_state(game)  # Obtient le nouvel état du jeu

        agent.train_short_memory(state_old, final_move, reward, state_new, done)  # Entraîne le modèle avec une expérience récente

        agent.remember(state_old, final_move, reward, state_new, done)  # Enregistre l'expérience dans la mémoire de l'agent

        if done:
            # Si le jeu est terminé
            game.reset()  # Réinitialise le jeu
            agent.n_games += 1  # Incrémente le nombre de jeux joués par l'agent
            agent.train_long_memory()  # Entraîne le modèle avec les expériences mémorisées

            if score > record:
                # Si le score actuel est supérieur au record
                record = score  # Met à jour le record de score
                agent.model.save()  # Sauvegarde le modèle

            print('Game', agent.n_games, 'Score', score, 'Record:', record)  # Affiche les informations sur le jeu

            plot_scores.append(score)  # Ajoute le score à la liste des scores de jeu
            total_score += score  # Met à jour le score total
            mean_score = total_score / agent.n_games  # Calcule la moyenne des scores de jeu
            plot_mean_scores.append(mean_score)  # Ajoute la moyenne des scores à la liste

            plot(plot_scores, plot_mean_scores)  # Affiche les graphiques de scores

if __name__ == '__main__':
    train()  # Lance la fonction principale d'entraînement de l'agent au jeu Snake