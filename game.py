import pygame
import random
from enum import Enum 
#permet de créer des énumérations en Python. Les énumérations sont des ensembles de noms symboliques associés à des valeurs constantes
from collections import namedtuple 
#permet de créer des tuples nommés en Python. Les tuples sont des structures de données immuables
import numpy as np 
#fournit des fonctionnalités pour le calcul numérique


pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Définition d'une énumération pour représenter les directions possibles du serpent
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Définition d'un tuple nommé Point pour représenter les coordonnées x et y d'un point sur l'écran
Point = namedtuple('Point', 'x, y')

# Définition de quelques constantes de couleur RGB et de paramètres du jeu
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN = (50, 205, 50)

BLOCK_SIZE = 20  # Taille d'un bloc sur l'écran
SPEED = 40  # Vitesse de déplacement du serpent

# Définition de la classe Game pour gérer le jeu
class Game:

    def __init__(self, w=640, h=480):
        self.w = w  # Largeur de la fenêtre du jeu
        self.h = h  # Hauteur de la fenêtre du jeu
        self.display = pygame.display.set_mode((self.w, self.h))  # Initialisation de la fenêtre du jeu
        pygame.display.set_caption('Snake')  # Définition du titre de la fenêtre
        self.clock = pygame.time.Clock()  # Initialisation de l'horloge de jeu
        self.reset()  # Réinitialisation du jeu


    def reset(self):
        self.direction = Direction.RIGHT  # Direction initiale du serpent
        self.head = Point(self.w / 2, self.h / 2)  # Position initiale de la tête du serpent
        # Initialisation du serpent avec une tête et deux blocs de corps
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0  # Initialisation du score à zéro
        self.food = None  # Initialisation de la nourriture
        self._place_food()  # Placement de la nourriture sur l'écran
        self.frame_iteration = 0  # Initialisation du compteur d'itération de jeu


    def _place_food(self):
        # Placement aléatoire de la nourriture sur l'écran
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        # Vérification pour s'assurer que la nourriture ne se trouve pas sur le serpent
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1  # Incrémentation du compteur d'itération à chaque étape de jeu
        for event in pygame.event.get():  # Collecte des événements utilisateur
            if event.type == pygame.QUIT:  # Si l'utilisateur quitte le jeu
                pygame.quit()
                quit()
        
        self._move(action)  # Déplacement du serpent en fonction de l'action donnée
        self.snake.insert(0, self.head)  # Ajout de la nouvelle position de la tête du serpent
        
        reward = 0  # Initialisation de la récompense à zéro
        game_over = False  # Initialisation de l'état de fin de jeu à False

        # Vérification des collisions avec les bords de l'écran ou avec le serpent lui-même ou si trop de temps est passé sans qu'il prenne la pomme
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True  # Le jeu est terminé
            reward = -10  # La récompense est négative
            return reward, game_over, self.score  # Retour de la récompense, de l'état de fin de jeu et du score

        # Placement de nouvelle nourriture ou simple déplacement du serpent
        if self.head == self.food:  # Si la tête du serpent atteint la nourriture
            self.score += 1  # Incrémentation du score
            reward = 10  # Récompense positive
            self._place_food()  # Placement d'une nouvelle nourriture
        else:
            self.snake.pop()  # Suppression du dernier bloc du serpent
        
        self._update_ui()  # Mise à jour de l'interface utilisateur
        self.clock.tick(SPEED)  # Attente pour contrôler la vitesse du jeu
        return reward, game_over, self.score  # Retour de la récompense, de l'état de fin de jeu et du score


    def is_collision(self, pt=None):
        # Vérification des collisions avec les bords de l'écran ou avec le serpent lui-même
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True  # Collision avec les bords de l'écran
        if pt in self.snake[1:]:
            return True  # Collision avec le corps du serpent
        return False  # Aucune collision


    def _update_ui(self):
        self.display.fill(GREEN)  # Remplissage de l'écran avec la couleur noire
        # Dessin du serpent et de ses blocs
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        # Dessin de la nourriture
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # Affichage du score à l'écran
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()  # Rafraîchissement de l'écran


    def _move(self, action):
        # Mise à jour de la direction du serpent en fonction de l'action donnée
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):  # Aucun changement de direction
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # Rotation vers la droite
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # Rotation vers la gauche
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir  # Mise à jour de la direction du serpent

        # Calcul de la nouvelle position de la tête du serpent
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)  # Mise à jour de la position de la tête du serpent